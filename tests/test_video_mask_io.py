"""以真實 CPU ffmpeg／OpenCV 檢查影片、alpha 與高精度序列，不啟動 ComfyUI。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
folders = ModuleType("folder_paths")
spec = importlib.util.spec_from_file_location("logic_lite_video_mask_io_test", ROOT / "nodes/video_mask_io.py")
io = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"folder_paths": folders}):
    spec.loader.exec_module(io)

PRORES, XQ, VP9, PNG, EXR, H264 = io.FORMATS


def synthetic(frames=3, height=24, width=32):
    t, y, x = torch.meshgrid(torch.arange(frames), torch.arange(height), torch.arange(width), indexing="ij")
    images = torch.stack((0.15 + x / width * 0.5, 0.2 + y / height * 0.5, 0.3 + t / max(frames, 1) * 0.2), -1)
    alpha = ((x + y * width + t * 71) % 1009).float() / 1008
    return images.float(), alpha


class VideoMaskIOTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.version.cuda is not None or torch.version.hip is not None:
            raise RuntimeError("I/O 測試必須使用 CPU-only PyTorch")
        torch.set_num_threads(1)
        cls.ffmpeg = io.find_ffmpeg()
        cls.ffprobe = io.find_ffprobe(cls.ffmpeg)
        result = subprocess.run([cls.ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
        cls.encoders = {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) > 2}
        print(f"I/O CPU ffmpeg: {cls.ffmpeg}; ffprobe: {cls.ffprobe}", flush=True)

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="logic_lite_io_test_")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        for name in ("input", "output", "temp"):
            (self.root / name).mkdir()
        folders.get_input_directory = lambda: str(self.root / "input")
        folders.get_output_directory = lambda: str(self.root / "output")
        folders.get_temp_directory = lambda: str(self.root / "temp")
        folders.get_annotated_filepath = lambda value: str(self.root / "input" / value.removesuffix(" [input]"))

    def require_codec(self, name):
        if name not in self.encoders:
            self.skipTest(f"本機 ffmpeg 缺少 CPU encoder {name}")

    def save(self, images, alpha=None, format=PNG, **kwargs):
        result = io.VideoCombineWithMask().combine_video(images, mask=alpha, format=format, **kwargs)
        return json.loads(result["result"][0]), result["ui"]

    def load(self, path, **kwargs):
        return io.LoadVideoWithMask().load_video(str(path), **kwargs)

    def roundtrip(self, format, rgb_tolerance, alpha_tolerance):
        images, alpha = synthetic()
        paths, ui = self.save(images, alpha, format=format)
        rgb, mask, count, fps, info = self.load(paths[0], missing_alpha="error")
        self.assertEqual(count, 3)
        self.assertEqual(fps, 24)
        self.assertTrue(json.loads(info)["has_alpha"])
        self.assertEqual(rgb.dtype, torch.float32)
        self.assertEqual(mask.dtype, torch.float32)
        self.assertEqual(rgb.device.type, "cpu")
        rgb_error, alpha_error = (rgb - images).abs().max().item(), (mask - alpha).abs().max().item()
        print(f"{format}: RGB max={rgb_error:.9g}; alpha max={alpha_error:.9g}", flush=True)
        self.assertLessEqual(rgb_error, rgb_tolerance)
        self.assertLessEqual(alpha_error, alpha_tolerance)
        self.assertTrue((self.root / "output" / ui["images"][0]["subfolder"] / "preview.png").is_file())
        return paths, rgb, mask

    def test_roundtrip_prores_4444(self):
        self.require_codec("prores_ks")
        paths, _, _ = self.roundtrip(PRORES, 0.012, 0.002)
        self.assertEqual(io._probe(paths[0], self.ffmpeg)["profile"], "4444")

    def test_roundtrip_prores_4444_xq(self):
        self.require_codec("prores_ks")
        paths, _, _ = self.roundtrip(XQ, 0.012, 0.002)
        self.assertEqual(io._probe(paths[0], self.ffmpeg)["profile"], "XQ")

    def test_prores_requests_rgba16_to_12bit_without_forced_color_conversion(self):
        self.require_codec("prores_ks")
        import numpy as np
        images, alpha = synthetic()
        for format in (PRORES, XQ):
            with patch.object(io.subprocess, "Popen", wraps=subprocess.Popen) as popen:
                paths, _ = self.save(images, alpha, format=format, bit_depth="12")
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("-pixel_format") + 1], "rgba64le")
            self.assertEqual(command[command.index("-pix_fmt") + 1], "yuva444p12le")
            self.assertEqual(command[command.index("-alpha_bits") + 1], "16")
            for option in ("-vf", "-colorspace", "-color_primaries", "-color_trc"):
                self.assertNotIn(option, command)
            metadata = json.loads(Path(paths[-1]).read_text())
            self.assertEqual(metadata["bit_depth"], "12")
            # 另外以 warning 等級重跑同一參數，記錄本機是否自動回退；不假設所有版本都只支援 10-bit。
            command = command.copy()
            command[command.index("-v") + 1] = "warning"
            command[-1] = str(Path(paths[0]).with_name("negotiation.mov"))
            raw = np.rint(np.concatenate((images.numpy(), alpha.numpy()[..., None]), -1) * 65535).astype("<u2")
            result = subprocess.run(command, input=raw.tobytes(), capture_output=True, check=True)
            print(f"{format} requested rgba64le -> yuva444p12le; encoder: "
                  + (result.stderr.decode().strip() or "無像素格式回退警告"), flush=True)
            self.assertEqual(io._probe(paths[0], self.ffmpeg)["pix_fmt"], "yuva444p12le")

    def test_sequence_uses_opencv_unchanged_without_ffmpeg_decode(self):
        import cv2
        images, alpha = synthetic()
        for format, codec in ((PNG, "png"), (EXR, "exr")):
            self.require_codec(codec)
            paths, _ = self.save(images, alpha, format=format)
            with (patch.object(io, "find_ffmpeg", side_effect=AssertionError("序列不可呼叫 ffmpeg")),
                  patch.object(io, "_probe", side_effect=AssertionError("序列不可呼叫 ffprobe")),
                  patch.object(io, "_decode", side_effect=AssertionError("序列不可走 ffmpeg 解碼")),
                  patch.object(cv2, "imread", wraps=cv2.imread) as imread):
                rgb, mask, _, _, info = self.load(paths[0])
            self.assertEqual(imread.call_count, 3)
            self.assertTrue(all(call.args[1] == cv2.IMREAD_UNCHANGED for call in imread.call_args_list))
            self.assertEqual(os.environ["OPENCV_IO_ENABLE_OPENEXR"], "1")
            self.assertEqual(json.loads(info)["decoder"], "opencv_imread_unchanged")
            tolerance = 0 if format == EXR else 0.5 / 65535 + 1e-7
            torch.testing.assert_close(rgb, images, atol=tolerance, rtol=0)
            torch.testing.assert_close(mask, alpha, atol=tolerance, rtol=0)

    def test_sequence_missing_opencv_fails_without_fallback(self):
        self.require_codec("png")
        images, alpha = synthetic()
        paths, _ = self.save(images, alpha)
        with (patch.dict(sys.modules, {"cv2": None}),
              patch.object(io, "_decode", side_effect=AssertionError("不可回退 ffmpeg"))):
            with self.assertRaisesRegex(RuntimeError, "需要 OpenCV"):
                self.load(paths[0])

    def test_sequence_mask_is_alpha_without_inversion(self):
        self.require_codec("png")
        self.require_codec("exr")
        images, _ = synthetic()
        alpha = torch.tensor([0., 0.25, 1.]).view(3, 1, 1).expand(3, 24, 32)
        for format in (PNG, EXR):
            paths, _ = self.save(images, alpha, format=format)
            _, actual, *_ = self.load(paths[0], missing_alpha="error")
            torch.testing.assert_close(actual, alpha, atol=0.5 / 65535 + 1e-7, rtol=0)
            self.assertEqual(actual[0, 0, 0].item(), 0)
            self.assertEqual(actual[-1, 0, 0].item(), 1)

    def test_sequence_rate_skip_stride_and_cap(self):
        self.require_codec("png")
        import cv2
        images = torch.full((6, 24, 32, 3), 0.4)
        alpha = torch.arange(6).float().view(6, 1, 1).expand(6, 24, 32) / 8
        paths, _ = self.save(images, alpha)
        _, actual, count, fps, _ = self.load(paths[0], sequence_fps=12, force_rate=6)
        self.assertEqual((count, fps), (3, 6))
        torch.testing.assert_close(actual[:, 0, 0], torch.tensor([1., 3., 5.]) / 8, atol=0.5 / 65535 + 1e-7, rtol=0)
        _, actual, count, fps, _ = self.load(paths[0], sequence_fps=12, force_rate=6,
                                          skip_first_frames=1, select_every_nth=2, frame_load_cap=1)
        self.assertEqual((count, fps), (1, 3))
        self.assertAlmostEqual(actual[0, 0, 0].item(), 3 / 8, delta=0.5 / 65535 + 1e-7)
        with patch.object(cv2, "imread", wraps=cv2.imread) as imread:
            _, _, count, fps, _ = self.load(paths[0], sequence_fps=12, force_rate=24)
        self.assertEqual((count, fps), (12, 24))
        self.assertEqual(imread.call_count, 6)

    def test_roundtrip_vp9_alpha(self):
        self.require_codec("libvpx-vp9")
        self.roundtrip(VP9, 0.04, 1 / 255 + 0.0001)

    def test_roundtrip_png16(self):
        self.require_codec("png")
        self.roundtrip(PNG, 0.5 / 65535 + 1e-7, 0.5 / 65535 + 1e-7)

    def test_roundtrip_exr_float(self):
        self.require_codec("exr")
        self.roundtrip(EXR, 0, 0)

    def test_png16_retains_all_uint16_codes_and_beats_8bit_baseline(self):
        self.require_codec("png")
        codes = torch.arange(3 * 24 * 32 * 3, dtype=torch.float32).reshape(3, 24, 32, 3) * 7
        images = codes / 65535
        alpha = (codes[..., 0] + 1) / 65535
        paths, _ = self.save(images, alpha)
        rgb, mask, *_ = self.load(paths[0])
        self.assertTrue(torch.equal(rgb, images))
        self.assertTrue(torch.equal(mask, alpha))
        self.assertGreater((images - (images * 255).round() / 255).abs().max().item(), 0.001)

    def test_exr_preserves_float32_hdr_negative_and_alpha_exactly(self):
        self.require_codec("exr")
        images, alpha = synthetic()
        images = images * 19.12345 - 4.23456
        alpha = alpha * 0.9333333 + 0.01234567
        paths, _ = self.save(images, alpha, format=EXR)
        rgb, mask, *_ = self.load(paths[0])
        self.assertTrue(torch.equal(rgb, images))
        self.assertTrue(torch.equal(mask, alpha))
        self.assertLess(rgb.min().item(), 0)
        self.assertGreater(rgb.max().item(), 1)

    def test_exr_half_matches_rounded_float16(self):
        self.require_codec("exr")
        images, alpha = synthetic()
        images = images * 4.12345 - 0.73333
        paths, _ = self.save(images, alpha, format=EXR, bit_depth="half")
        rgb, mask, *_ = self.load(paths[0])
        self.assertTrue(torch.equal(rgb, images.half().float()))
        self.assertTrue(torch.equal(mask, alpha.half().float()))

    def test_rgb_and_grayscale_exr_without_alpha_keep_float_range(self):
        self.require_codec("exr")
        import numpy as np
        for grayscale in (False, True):
            images, _ = synthetic(frames=1)
            images = images * 21.12345 - 5.33333
            if grayscale:
                images = images[..., :1].expand(-1, -1, -1, 3).clone()
                raw = np.ascontiguousarray(images.numpy()[..., 0], dtype="<f4")
                pixel_format = "grayf32le"
            else:
                raw = np.ascontiguousarray(images.numpy()[..., [1, 2, 0]].transpose(0, 3, 1, 2), dtype="<f4")
                pixel_format = "gbrpf32le"
            source = self.root / "input" / f"source_{grayscale}.exr"
            subprocess.run([self.ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", pixel_format,
                            "-s", "32x24", "-i", "pipe:0", "-c:v", "exr", "-format", "float",
                            "-threads", "2", "-frames:v", "1", str(source)], input=raw.tobytes(), check=True)
            rgb, mask, *_ = self.load(source)
            self.assertTrue(torch.equal(rgb, images))
            self.assertTrue(torch.equal(mask, torch.ones_like(mask)))
            with self.assertRaisesRegex(ValueError, "missing_alpha"):
                self.load(source, missing_alpha="error")

    def test_exr_half_overflow_is_explicit(self):
        self.require_codec("exr")
        images, alpha = synthetic()
        with self.assertRaisesRegex(ValueError, "float16"):
            self.save(images * 200000, alpha, format=EXR, bit_depth="half")

    def test_prores_has_more_than_256_alpha_levels(self):
        self.require_codec("prores_ks")
        alpha = torch.arange(1024).reshape(1, 32, 32).float() / 1023
        images = alpha.unsqueeze(-1).expand(-1, -1, -1, 3)
        paths, _ = self.save(images, alpha, format=XQ)
        _, decoded, *_ = self.load(paths[0])
        self.assertGreater(decoded.unique().numel(), 256)
        paths8, _ = self.save((images * 255).round() / 255, (alpha * 255).round() / 255, format=XQ)
        _, baseline, *_ = self.load(paths8[0])
        high_error = (decoded - alpha).square().mean().item()
        low_error = (baseline - alpha).square().mean().item()
        self.assertLess(high_error, low_error)

    def test_missing_alpha_ones_or_error(self):
        self.require_codec("libx264")
        images, _ = synthetic()
        paths, _ = self.save(images, format=H264)
        _, alpha, *_ = self.load(paths[0], missing_alpha="ones")
        self.assertTrue(torch.equal(alpha, torch.ones_like(alpha)))
        with self.assertRaisesRegex(ValueError, "missing_alpha"):
            self.load(paths[0], missing_alpha="error")

    def test_nonalpha_mask_mode_error_separate_and_ignore(self):
        self.require_codec("libx264")
        self.require_codec("png")
        images, alpha = synthetic()
        with self.assertRaisesRegex(ValueError, "mask_mode"):
            self.save(images, alpha, format=H264)
        for mode in ("separate_png", "ignore"):
            paths, ui = self.save(images, alpha, format=H264, mask_mode=mode)
            self.assertIn("mask_videos", ui)
            self.assertEqual(len(paths), 3 if mode == "separate_png" else 2)
            if mode == "separate_png":
                _, mask, *_ = self.load(paths[1], missing_alpha="error")
                torch.testing.assert_close(mask, alpha, atol=0.5 / 65535 + 1e-7, rtol=0)

    def test_always_save_mask_even_for_alpha_video_or_ignore_mode(self):
        self.require_codec("libvpx-vp9")
        self.require_codec("libx264")
        self.require_codec("png")
        images, alpha = synthetic()
        for format in (VP9, H264):
            paths, _ = self.save(images, alpha, format=format, mask_mode="ignore", save_mask_sequence=True)
            self.assertEqual(len(paths), 3)
            _, result, *_ = self.load(paths[1], missing_alpha="error")
            torch.testing.assert_close(result, alpha, atol=0.5 / 65535 + 1e-7, rtol=0)

    def test_skip_stride_and_cap_on_short_video(self):
        self.require_codec("prores_ks")
        images = torch.ones(8, 24, 32, 3) * 0.4
        alpha = torch.arange(8).float().view(8, 1, 1).expand(8, 24, 32) / 8
        paths, _ = self.save(images, alpha, format=PRORES, frame_rate=12)
        _, mask, count, fps, _ = self.load(paths[0], skip_first_frames=1, select_every_nth=2, frame_load_cap=3)
        self.assertEqual(count, 3)
        self.assertEqual(fps, 6)
        torch.testing.assert_close(mask[:, 0, 0], torch.tensor([1., 3., 5.]) / 8, atol=0.002, rtol=0)

    def test_force_rate_changes_frame_count_and_reported_fps(self):
        self.require_codec("libvpx-vp9")
        images, alpha = synthetic(frames=6)
        paths, _ = self.save(images, alpha, format=VP9, frame_rate=12)
        _, _, count, fps, _ = self.load(paths[0], force_rate=6)
        self.assertEqual((count, fps), (3, 6))

    def test_video_rotation_metadata_keeps_pixels_and_correct_dimensions(self):
        self.require_codec("libx264")
        images, _ = synthetic()
        paths, _ = self.save(images, format=H264)
        rotated = self.root / "input" / "rotated.mp4"
        subprocess.run([self.ffmpeg, "-v", "error", "-display_rotation", "90", "-i", paths[0],
                        "-c", "copy", str(rotated)], check=True)
        baseline, *_ = self.load(paths[0])
        actual, _, _, _, info = self.load(rotated)
        self.assertEqual(actual.shape, (3, 32, 24, 3))
        self.assertEqual(json.loads(info)["source_rotation"], 90)
        self.assertTrue(torch.equal(actual, torch.rot90(baseline, 1, dims=(1, 2))))

    def test_sequence_directory_first_frame_and_patterns(self):
        self.require_codec("png")
        images, alpha = synthetic(frames=4)
        paths, _ = self.save(images, alpha)
        directory = Path(paths[0]).parent
        for source in (directory, paths[0], directory / "########.png", directory / "*.png"):
            rgb, _, count, fps, _ = self.load(source, sequence_fps=15)
            self.assertEqual((count, fps), (4, 15))
            torch.testing.assert_close(rgb, images, atol=0.5 / 65535 + 1e-7, rtol=0)
        rgb, _, count, *_ = self.load(directory / "00000002.png")
        self.assertEqual(count, 2)
        torch.testing.assert_close(rgb, images[2:], atol=0.5 / 65535 + 1e-7, rtol=0)

    def test_sequence_natural_order_and_gaps(self):
        self.require_codec("png")
        images, alpha = synthetic()
        paths, _ = self.save(images, alpha)
        directory = Path(paths[0]).parent
        for index, number in enumerate((1, 2, 10)):
            (directory / f"{index:08d}.png").rename(directory / f"frame{number}.png")
        rgb, *_ = self.load(directory)
        torch.testing.assert_close(rgb, images, atol=0.5 / 65535 + 1e-7, rtol=0)

    def test_mixed_sequence_dimensions_or_precision_fail_before_decode(self):
        for format, codec in ((PNG, "png"), (EXR, "exr")):
            self.require_codec(codec)
            images, alpha = synthetic(frames=2)
            paths, _ = self.save(images, alpha, format=format)
            small, _ = self.save(images[:1, :12], alpha[:1, :12], format=format)
            import shutil
            extension = Path(paths[0]).suffix
            shutil.copyfile(Path(small[0]).parent / f"00000000{extension}",
                            Path(paths[0]).parent / f"00000001{extension}")
            with self.assertRaisesRegex(ValueError, "相同尺寸、位元深度與通道"):
                self.load(paths[0])
        images, alpha = synthetic(frames=2)
        full, _ = self.save(images, alpha, format=EXR)
        half, _ = self.save(images[:1], alpha[:1], format=EXR, bit_depth="half")
        shutil.copyfile(Path(half[0]).parent / "00000000.exr", Path(full[0]).parent / "00000001.exr")
        with self.assertRaisesRegex(ValueError, "位元深度"):
            self.load(full[0])

    def test_unicode_and_apostrophes_in_sequence_path(self):
        self.require_codec("png")
        images, alpha = synthetic()
        paths, _ = self.save(images, alpha, filename_prefix="測試/joe's take")
        rgb, *_ = self.load(paths[0])
        torch.testing.assert_close(rgb, images, atol=0.5 / 65535 + 1e-7, rtol=0)

    def test_size_options_and_float_resize_preserve_hdr_range(self):
        self.require_codec("exr")
        images, alpha = synthetic()
        paths, _ = self.save(images * 20 - 5, alpha, format=EXR)
        rgb, mask, _, _, info = self.load(paths[0], custom_width=16)
        self.assertEqual(rgb.shape, (3, 12, 16, 3))
        self.assertEqual(mask.shape, (3, 12, 16))
        self.assertEqual(json.loads(info)["height"], 12)
        self.assertLess(rgb.min().item(), 0)
        self.assertGreater(rgb.max().item(), 1)

    def test_shape_length_validation_never_broadcasts(self):
        images, _ = synthetic()
        for shape in ((2, 24, 32), (3, 23, 32), (3, 24, 31), (24, 32)):
            with self.assertRaisesRegex(ValueError, "相同 T,H,W"):
                self.save(images, torch.ones(shape))
        with self.assertRaisesRegex(ValueError, "images"):
            self.save(torch.empty(0, 24, 32, 3))

    def test_rgba_alpha_and_explicit_mask_override(self):
        self.require_codec("png")
        images, alpha = synthetic()
        rgba = torch.cat((images, alpha.unsqueeze(-1)), -1)
        for supplied, expected in ((None, alpha), (1 - alpha, 1 - alpha)):
            paths, _ = self.save(rgba, supplied)
            _, actual, *_ = self.load(paths[0])
            torch.testing.assert_close(actual, expected, atol=0.5 / 65535 + 1e-7, rtol=0)
        with self.assertRaisesRegex(ValueError, "mask_mode"):
            self.save(rgba, format=H264)

    def test_loop_count_temp_output_and_manifest(self):
        self.require_codec("png")
        images, alpha = synthetic()
        paths, ui = self.save(images, alpha, loop_count=1, save_output=False, filename_prefix="shots/take")
        rgb, _, count, *_ = self.load(paths[0])
        self.assertEqual(count, 6)
        torch.testing.assert_close(rgb, images.repeat(2, 1, 1, 1), atol=0.5 / 65535 + 1e-7, rtol=0)
        self.assertEqual(ui["images"][0]["type"], "temp")
        self.assertTrue(Path(paths[0]).is_relative_to(self.root / "temp"))
        self.assertEqual(json.loads(Path(paths[-1]).read_text())["frame_count"], 6)

    def test_output_prefix_and_depth_validation(self):
        images, alpha = synthetic()
        for prefix in ("../escape", "/absolute", "bad%03d", "C:\\bad", ""):
            with self.assertRaisesRegex(ValueError, "filename_prefix"):
                self.save(images, alpha, filename_prefix=prefix)
        for format, depth in ((PRORES, "16"), (PNG, "8"), (VP9, "10"), (EXR, "16")):
            with self.assertRaisesRegex(ValueError, "bit_depth"):
                self.save(images, alpha, format=format, bit_depth=depth)
        with self.assertRaisesRegex(ValueError, "偶數"):
            self.save(images[:, :, :-1], format=H264)

    def test_missing_tools_and_explicit_override(self):
        with patch.dict(os.environ, {"LOGIC_LITE_FFMPEG": "/nonexistent/ffmpeg"}):
            with self.assertRaisesRegex(RuntimeError, "LOGIC_LITE_FFMPEG"):
                io.find_ffmpeg()
        with patch.dict(os.environ, {"LOGIC_LITE_FFMPEG": self.ffmpeg}):
            self.assertEqual(io.find_ffmpeg(), self.ffmpeg)
        with patch.dict(os.environ, {"LOGIC_LITE_FFPROBE": "/nonexistent/ffprobe"}):
            with self.assertRaisesRegex(RuntimeError, "LOGIC_LITE_FFPROBE"):
                io.find_ffprobe(self.ffmpeg)

    def test_pipe_failure_returns_ffmpeg_error(self):
        self.require_codec("png")
        images, alpha = synthetic(frames=32, height=64, width=64)
        destination = self.root / "does_not_exist" / "frame%08d.png"
        with self.assertRaisesRegex(RuntimeError, "ffmpeg 編碼失敗"):
            io._encode(destination, "png", "16", images, alpha, 24, 1, self.ffmpeg)

    def test_empty_selection_and_invalid_options(self):
        self.require_codec("png")
        images, alpha = synthetic()
        paths, _ = self.save(images, alpha)
        with self.assertRaisesRegex(ValueError, "沒有影格"):
            self.load(paths[0], skip_first_frames=10)
        for options in ({"force_rate": -1}, {"frame_load_cap": -1}, {"select_every_nth": 0}, {"missing_alpha": "bad"}):
            with self.assertRaises(ValueError):
                self.load(paths[0], **options)

    def test_streaming_is_bounded_to_chunks(self):
        self.require_codec("png")
        images, alpha = synthetic(frames=7)
        with patch.object(io, "_BUFFER_BYTES", 24 * 32 * 4 * 4 * 2):
            chunks = list(io._rgba_chunks(images, alpha, repetitions=1))
            self.assertEqual([len(chunk) for chunk in chunks], [2, 2, 2, 1])
            paths, _ = self.save(images, alpha)
            rgb, *_ = self.load(paths[0])
        torch.testing.assert_close(rgb, images, atol=0.5 / 65535 + 1e-7, rtol=0)


class VideoIORegistrationTests(unittest.TestCase):
    def test_package_maps_two_independent_nodes_without_vhs(self):
        from test_mask_registration import load_package
        package = load_package()
        self.assertEqual({key: value.__name__ for key, value in package._VIDEO_IO_CLASS.items()},
                         {io.LOAD_ID: "LoadVideoWithMask", io.SAVE_ID: "VideoCombineWithMask"})
        for key in (io.LOAD_ID, io.SAVE_ID):
            node = package.NODE_CLASS_MAPPINGS[key]
            self.assertEqual(node.CATEGORY, "video/mask")
            self.assertIsInstance(node.INPUT_TYPES(), dict)
            self.assertTrue(callable(getattr(node, node.FUNCTION)))
        formats = io.VideoCombineWithMask.INPUT_TYPES()["required"]["format"][0]
        self.assertEqual(sum(name.startswith("[mask]") for name in formats), 5)
        self.assertEqual(sum(name.startswith("[no mask]") for name in formats), 1)


if __name__ == "__main__":
    unittest.main()
