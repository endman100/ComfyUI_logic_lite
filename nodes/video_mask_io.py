"""獨立實作的高精度影片／遮罩 I/O，不依賴 VideoHelperSuite。"""

from fractions import Fraction
import glob
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import folder_paths


FORMATS = {
    "[mask] ProRes 4444 (.mov)": ("prores", "mov", "12"),
    "[mask] ProRes 4444 XQ (.mov)": ("prores_xq", "mov", "12"),
    "[mask] VP9 alpha (.webm)": ("vp9", "webm", "8"),
    "[mask] PNG16 RGBA sequence": ("png", "png", "16"),
    "[mask] EXR float sequence": ("exr", "exr", "float"),
    "[no mask] H.264 (.mp4)": ("h264", "mp4", "8"),
}
LOAD_ID = "Load Video With Mask (Upload) (endman100)"
SAVE_ID = "Video Combine With Mask (endman100)"
_BUFFER_BYTES = 16 * 1024 * 1024


def find_ffmpeg():
    """依明確覆寫、imageio、VHS 已有設定、PATH 的順序尋找，不匯入 VHS。"""
    override = os.environ.get("LOGIC_LITE_FFMPEG")
    if override:
        found = shutil.which(override)
        if not found:
            raise RuntimeError("LOGIC_LITE_FFMPEG 指向不存在或不可執行的 ffmpeg")
        return found
    candidates = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        candidates.append(get_ffmpeg_exe())
    except (ImportError, RuntimeError, OSError):
        pass
    candidates.append(os.environ.get("VHS_FORCE_FFMPEG_PATH"))
    candidates.extend(getattr(module, "ffmpeg_path", None) for name, module in tuple(sys.modules.items())
                      if name.endswith("videohelpersuite.utils"))
    candidates.append(shutil.which("ffmpeg"))
    for candidate in candidates:
        found = shutil.which(candidate) if candidate else None
        if found:
            return found
    raise RuntimeError("找不到 ffmpeg；請安裝 CPU ffmpeg，或設定 LOGIC_LITE_FFMPEG")


def find_ffprobe(ffmpeg):
    override = os.environ.get("LOGIC_LITE_FFPROBE")
    if override:
        found = shutil.which(override)
        if not found:
            raise RuntimeError("LOGIC_LITE_FFPROBE 指向不存在或不可執行的 ffprobe")
        return found
    companion = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    found = shutil.which(str(companion)) or shutil.which("ffprobe")
    if not found:
        raise RuntimeError("找不到 ffprobe；請安裝 ffmpeg 工具組，或設定 LOGIC_LITE_FFPROBE")
    return found


def _probe(path, ffmpeg):
    result = subprocess.run([find_ffprobe(ffmpeg), "-v", "error", "-select_streams", "v:0",
                             "-show_streams", "-of", "json", str(path)], capture_output=True)
    if result.returncode:
        raise RuntimeError("無法讀取媒體資訊：" + result.stderr.decode("utf-8", errors="replace"))
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"檔案沒有影像軌：{path}")
    stream = streams[0]
    pixel_format = stream.get("pix_fmt", "")
    tags = {key.lower(): str(value) for key, value in stream.get("tags", {}).items()}
    stream["has_alpha"] = (any(part in pixel_format for part in ("rgba", "bgra", "argb", "abgr", "yuva", "gbrap"))
                           or pixel_format.startswith("ya") or tags.get("alpha_mode") == "1")
    try:
        stream["fps"] = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        stream["fps"] = 0.0
    return stream


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必須是大於或等於 {minimum} 的整數")
    return value


def _rate(value, name, allow_zero=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} 必須是{'非負' if allow_zero else '正'}有限數值")
    return float(value)


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def _sequence_signature(path):
    """只讀取影格標頭，檢查整批序列的尺寸、精度與通道一致。"""
    with path.open("rb") as stream:
        if path.suffix.lower() == ".png":
            header = stream.read(29)
            if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise ValueError(f"無效的 PNG 標頭：{path}")
            return struct.unpack(">IIBB", header[16:26])
        if stream.read(4) != struct.pack("<I", 20000630):
            raise ValueError(f"無效的 EXR 標頭：{path}")
        version = stream.read(4)
        if len(version) != 4 or struct.unpack("<I", version)[0] & 0x1A00:
            raise ValueError("僅支援一般單層 scanline EXR，不支援 tiled／deep／multipart EXR")

        def string():
            value = bytearray()
            while True:
                byte = stream.read(1)
                if not byte or len(value) > 255:
                    raise ValueError(f"無效的 EXR 屬性：{path}")
                if byte == b"\0":
                    return bytes(value)
                value.extend(byte)

        attributes = {}
        while True:
            name = string()
            if not name:
                break
            string()
            length = stream.read(4)
            if len(length) != 4:
                raise ValueError(f"不完整的 EXR 標頭：{path}")
            size = struct.unpack("<I", length)[0]
            if size > 16 * 1024 * 1024:
                raise ValueError(f"EXR 屬性過大：{path}")
            value = stream.read(size)
            if len(value) != size:
                raise ValueError(f"不完整的 EXR 屬性：{path}")
            if name in (b"dataWindow", b"channels"):
                attributes[name] = value
        if b"dataWindow" not in attributes or b"channels" not in attributes:
            raise ValueError(f"EXR 缺少 dataWindow 或 channels：{path}")
        return attributes[b"dataWindow"], attributes[b"channels"]


def _resolve_source(video):
    if not isinstance(video, str) or not video.strip():
        raise ValueError("video 請填入已上傳檔名、影片路徑或 PNG／EXR 序列路徑")
    text = video.strip()
    if text.endswith((" [input]", " [output]", " [temp]")):
        text = folder_paths.get_annotated_filepath(text)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(folder_paths.get_input_directory()) / path
    path = path.resolve()
    if path.is_dir():
        files = [item for item in path.iterdir() if item.suffix.lower() in (".png", ".exr") and item.is_file()]
    elif any(char in path.name for char in ("*", "?", "#", "%")):
        pattern = re.sub(r"%0?\d*d|#+", "*", str(path))
        files = [Path(item) for item in glob.glob(pattern) if Path(item).is_file()]
    elif path.suffix.lower() in (".png", ".exr") and path.is_file():
        match = re.fullmatch(r"(.*?)(\d+)", path.stem)
        if match:
            prefix, number = match.groups()
            files = [item for item in path.parent.iterdir()
                     if item.suffix.lower() == path.suffix.lower()
                     and re.fullmatch(re.escape(prefix) + r"\d+", item.stem)
                     and int(re.search(r"\d+$", item.stem).group()) >= int(number)]
        else:
            files = [path]
    elif path.is_file():
        return path, None
    else:
        raise ValueError(f"找不到 video：{path}")
    files = sorted(files, key=_natural_key)
    if not files or any(item.suffix.lower() not in (".png", ".exr") for item in files):
        raise ValueError("序列必須包含 PNG 或 EXR 影格")
    if len({item.suffix.lower() for item in files}) != 1:
        raise ValueError("同一序列不可混用 PNG 與 EXR；請指定檔名 pattern")
    return files[0], files


def _load_sequence(first, files, sequence_fps, force_rate, skip, nth, cap, missing_alpha):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    try:
        import cv2
    except (ImportError, OSError) as error:
        raise RuntimeError("PNG／EXR 序列需要 OpenCV（opencv-python）；不會降精度或改用 ffmpeg 解碼") from error
    signature = _sequence_signature(first)
    for path in files:
        if _sequence_signature(path) != signature:
            raise ValueError(f"序列影格必須具有相同尺寸、位元深度與通道：{path}")

    def read(path):
        try:
            frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except cv2.error as error:
            raise RuntimeError(f"OpenCV 無法讀取序列影格；EXR 需要啟用 OpenEXR 的版本：{path}") from error
        if frame is None:
            hint = "；請確認 OpenCV 編譯時包含 OpenEXR 支援" if path.suffix.lower() == ".exr" else ""
            raise ValueError(f"OpenCV 無法讀取序列影格{hint}：{path}")
        if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[-1] not in (3, 4)):
            raise ValueError(f"序列僅支援灰階、RGB 或 RGBA：{path}")
        if frame.dtype not in (np.uint8, np.uint16, np.float32):
            raise ValueError(f"不支援的序列像素型別 {frame.dtype}：{path}")
        return frame

    first_frame = read(first)
    has_alpha = first_frame.ndim == 3 and first_frame.shape[-1] == 4
    if not has_alpha and missing_alpha == "error":
        raise ValueError("來源沒有 alpha；若需要全白遮罩，請將 missing_alpha 設為 ones")
    # 以影格中心的時間做最近鄰重採樣，再套用跳格、抽格與 cap；不插值原始像素。
    rate = force_rate or sequence_fps
    count = math.floor(len(files) * rate / sequence_fps + 0.5)
    stop = min(count, skip + cap * nth) if cap else count
    positions = np.arange(skip, stop, nth, dtype=np.int64)
    indices = np.minimum(np.floor((positions + 0.5) * sequence_fps / rate).astype(np.int64), len(files) - 1)
    if not len(indices):
        raise ValueError("選取後沒有影格；請檢查 skip_first_frames 與 frame_load_cap")
    # 每個檔案僅 imread 一次；整批 stack、通道重排與數值轉換均向量化。
    unique, inverse = np.unique(indices, return_inverse=True)
    frames = [first_frame if index == 0 else read(files[index]) for index in unique]
    if any(frame.shape != first_frame.shape or frame.dtype != first_frame.dtype for frame in frames):
        raise ValueError("序列影格必須具有相同尺寸、位元深度與通道")
    array = np.stack(frames)[inverse].astype(np.float32)
    if first_frame.dtype in (np.uint8, np.uint16):
        array /= np.iinfo(first_frame.dtype).max
    if first_frame.ndim == 2:
        rgb = np.repeat(array[..., None], 3, axis=-1)
    else:
        rgb = array[..., [2, 1, 0]]
    alpha = array[..., 3:] if has_alpha else np.ones((*array.shape[:3], 1), dtype=np.float32)
    rgba = torch.from_numpy(np.concatenate((rgb, alpha), axis=-1))
    info = {"width": first_frame.shape[1], "height": first_frame.shape[0], "has_alpha": has_alpha,
            "pix_fmt": f"opencv_{'BGRA' if has_alpha else ('gray' if first_frame.ndim == 2 else 'BGR')}_{first_frame.dtype}"}
    return rgba, info


def _process_error(log, operation):
    log.seek(0, os.SEEK_END)
    log.seek(max(0, log.tell() - 6000))
    return RuntimeError(f"ffmpeg {operation}失敗：" + log.read().decode("utf-8", errors="replace"))


def _decode(args, width, height, floating, ffmpeg, has_alpha=True, grayscale=False):
    channels = (1 if grayscale else (4 if has_alpha else 3)) if floating else 4
    dtype = np.dtype("<f4" if floating else "<u2")
    frame_bytes = width * height * channels * dtype.itemsize
    chunk_size = frame_bytes * max(1, _BUFFER_BYTES // frame_bytes)
    pixel_format = ("grayf32le" if grayscale else ("gbrapf32le" if has_alpha else "gbrpf32le")) if floating else "rgba64le"
    command = [ffmpeg, "-v", "error", "-nostdin", "-hwaccel", "none", "-noautorotate", "-threads", "2",
               *args, "-map", "0:v:0", "-an", "-sn", "-dn", "-pix_fmt", pixel_format,
               "-threads", "2", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            while True:
                raw = process.stdout.read(chunk_size)
                if not raw:
                    break
                if len(raw) % frame_bytes:
                    raise RuntimeError("ffmpeg 傳回不完整影格，請確認序列尺寸與格式一致")
                if floating:
                    planes = np.frombuffer(raw, dtype=dtype).reshape(-1, channels, height, width)
                    order = [0, 0, 0] if grayscale else ([2, 0, 1, 3] if has_alpha else [2, 0, 1])
                    array = planes[:, order].transpose(0, 2, 3, 1).copy()
                    if not has_alpha:
                        array = np.concatenate((array, np.ones((*array.shape[:-1], 1), dtype=np.float32)), axis=-1)
                else:
                    array = np.frombuffer(raw, dtype=dtype).reshape(-1, height, width, 4).astype(np.float32)
                    array /= 65535.0
                yield torch.from_numpy(array)
            if process.wait():
                raise _process_error(log, "解碼")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


class LoadVideoWithMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("STRING", {"default": "", "tooltip": "上傳檔名、絕對路徑、序列資料夾、首格或 %08d／####／* pattern"}),
            "frame_load_cap": ("INT", {"default": 0, "min": 0}),
            "skip_first_frames": ("INT", {"default": 0, "min": 0}),
            "select_every_nth": ("INT", {"default": 1, "min": 1}),
            "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0}),
            "custom_width": ("INT", {"default": 0, "min": 0}),
            "custom_height": ("INT", {"default": 0, "min": 0}),
            "sequence_fps": ("FLOAT", {"default": 24.0, "min": 0.01}),
            "missing_alpha": (["ones", "error"], {"default": "ones"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "mask", "frame_count", "fps", "video_info")
    FUNCTION = "load_video"
    CATEGORY = "video/mask"
    DESCRIPTION = "CPU 高精度影片載入；PNG16／EXR 序列使用 OpenCV IMREAD_UNCHANGED。MASK 直接等於 alpha（白色不透明），不反相；無 alpha 可選全白或報錯。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 路徑可能指向可持續更新的整批序列，避免只依首格時間戳快取。
        return float("nan")

    def load_video(self, video, frame_load_cap=0, skip_first_frames=0, select_every_nth=1,
                   force_rate=0.0, custom_width=0, custom_height=0, sequence_fps=24.0, missing_alpha="ones"):
        cap = _integer(frame_load_cap, "frame_load_cap", 0)
        skip = _integer(skip_first_frames, "skip_first_frames", 0)
        nth = _integer(select_every_nth, "select_every_nth", 1)
        width = _integer(custom_width, "custom_width", 0)
        height = _integer(custom_height, "custom_height", 0)
        force_rate = _rate(force_rate, "force_rate", allow_zero=True)
        sequence_fps = _rate(sequence_fps, "sequence_fps")
        if missing_alpha not in ("ones", "error"):
            raise ValueError("missing_alpha 必須是 ones 或 error")
        first, files = _resolve_source(video)
        if files is not None:
            rgba, info = _load_sequence(first, files, sequence_fps, force_rate, skip, nth, cap, missing_alpha)
        else:
            ffmpeg = find_ffmpeg()
            info = _probe(first, ffmpeg)
        if not info["has_alpha"] and missing_alpha == "error":
            raise ValueError("來源沒有 alpha；若需要全白遮罩，請將 missing_alpha 設為 ones")
        source_fps = sequence_fps if files is not None else info["fps"]
        if source_fps <= 0:
            raise ValueError("無法取得來源 fps；請先轉為有有效影格率的影片或影像序列")
        source_w, source_h = info["width"], info["height"]
        rotation = next((item["rotation"] for item in info.get("side_data_list", []) if "rotation" in item), 0)
        if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 1e-6:
            raise ValueError("目前僅支援 90 度倍數的影片旋轉資訊")
        turns = round(rotation / 90) % 4
        display_w, display_h = (source_h, source_w) if turns % 2 else (source_w, source_h)
        width = width or (max(1, round(display_w * height / display_h)) if height else display_w)
        height = height or max(1, round(display_h * width / display_w))
        fps = (force_rate or source_fps) / nth
        floating = first.suffix.lower() == ".exr"
        filters = ([f"fps={force_rate}"] if force_rate else [])
        filters += [f"trim=start_frame={skip}", f"select=not(mod(n\\,{nth}))", f"setpts=N/({fps}*TB)"]
        batches = []
        if files is not None:
            if (width, height) != (display_w, display_h):
                rgba = F.interpolate(rgba.movedim(-1, 1), size=(height, width), mode="bilinear",
                                     align_corners=False).movedim(1, -1)
            batches.append(rgba)
        else:
            inputs = ["-i", str(first)]
            # VP9 原生解碼器可能忽略 WebM alpha；明確使用 libvpx 解碼。
            if info.get("codec_name") == "vp9" and info["has_alpha"]:
                inputs = ["-c:v", "libvpx-vp9", *inputs]
            args = [*inputs, "-vf", ",".join(filters)]
            if cap:
                args += ["-frames:v", str(cap)]
            for batch in _decode(args, source_w, source_h, floating, ffmpeg,
                                 has_alpha=info["has_alpha"], grayscale=info.get("pix_fmt", "").startswith("gray")):
                if turns:
                    batch = torch.rot90(batch, turns, dims=(1, 2))
                if (width, height) != (display_w, display_h):
                    batch = F.interpolate(batch.movedim(-1, 1), size=(height, width), mode="bilinear",
                                          align_corners=False).movedim(1, -1)
                batches.append(batch)
        if not batches:
            raise ValueError("選取後沒有影格；請檢查 skip_first_frames 與 frame_load_cap")
        rgba = torch.cat(batches, dim=0)
        alpha = rgba[..., 3].contiguous() if info["has_alpha"] else torch.ones_like(rgba[..., 0])
        metadata = {"source": str(first), "source_fps": source_fps, "fps": fps,
                    "frame_count": len(rgba), "width": width, "height": height,
                    "has_alpha": info["has_alpha"], "source_pix_fmt": info.get("pix_fmt"),
                    "source_rotation": rotation,
                    "decoder": "opencv_imread_unchanged" if files is not None else "ffmpeg",
                    "alpha_convention": "mask=alpha", "sequence_files": len(files) if files else None}
        return rgba[..., :3].contiguous(), alpha, len(rgba), fps, json.dumps(metadata, ensure_ascii=False)


def _validate_images(images, mask):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] not in (3, 4) or min(images.shape) < 1:
        raise ValueError("images 必須是非空的 [T,H,W,3] 或 [T,H,W,4]")
    if images.is_complex():
        raise ValueError("images 必須是實數")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.is_complex():
            raise ValueError("mask 必須是實數張量")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.shape != images.shape[:3]:
            raise ValueError(f"images 與 mask 必須具有相同 T,H,W；images={tuple(images.shape[:3])}，mask={tuple(mask.shape)}")
    return mask


def _rgba_chunks(images, mask, repetitions, half=False, mask_only=False):
    count, height, width = images.shape[:3]
    batch_size = max(1, _BUFFER_BYTES // (height * width * 4 * 4))
    # 只依有界 I/O 批次與播放次數迭代；轉換在整個批次上向量化，不逐像素處理。
    for _ in range(repetitions):
        for start in range(0, count, batch_size):
            rgb = images[start:start + batch_size, ..., :3].detach().to(device="cpu", dtype=torch.float32)
            if mask is not None:
                alpha = mask[start:start + batch_size].detach().to(device="cpu", dtype=torch.float32).unsqueeze(-1)
            elif images.shape[-1] == 4:
                alpha = images[start:start + batch_size, ..., 3:].detach().to(device="cpu", dtype=torch.float32)
            else:
                alpha = torch.ones_like(rgb[..., :1])
            if not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all():
                raise ValueError("images 與 mask 不可包含 NaN 或 Infinity")
            alpha = alpha.clamp(0, 1)
            rgba = torch.cat((alpha.expand_as(rgb) if mask_only else rgb, alpha), dim=-1).numpy()
            if half and np.any(np.abs(rgba) > np.finfo(np.float16).max):
                raise ValueError("EXR half 數值超過 float16 範圍；請改用 bit_depth=float")
            yield rgba.astype(np.float16).astype(np.float32) if half else rgba


def _encode(destination, kind, depth, images, mask, frame_rate, repetitions, ffmpeg, mask_only=False):
    height, width = images.shape[1:3]
    floating = kind == "exr"
    raw_format = "gbrapf32le" if floating else "rgba64le"
    command = [ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pixel_format", raw_format,
               "-video_size", f"{width}x{height}", "-framerate", str(frame_rate), "-i", "pipe:0", "-an"]
    if kind in ("prores", "prores_xq"):
        command += ["-c:v", "prores_ks", "-profile:v", "4444xq" if kind == "prores_xq" else "4444",
                    "-pix_fmt", "yuva444p12le", "-alpha_bits", "16"]
    elif kind == "vp9":
        command += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-lossless", "1", "-auto-alt-ref", "0"]
    elif kind == "png":
        command += ["-c:v", "png", "-pix_fmt", "rgba64be", "-start_number", "0"]
    elif kind == "exr":
        command += ["-c:v", "exr", "-pix_fmt", "gbrapf32le", "-format", "half" if depth == "half" else "float",
                    "-compression", "zip16", "-start_number", "0"]
    elif kind == "h264":
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart"]
    command += ["-threads", "2", str(destination)]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            try:
                for rgba in _rgba_chunks(images, mask, repetitions, half=depth == "half", mask_only=mask_only):
                    if floating:
                        data = np.ascontiguousarray(rgba[..., [1, 2, 0, 3]].transpose(0, 3, 1, 2), dtype="<f4")
                    else:
                        data = np.rint(np.clip(rgba, 0, 1) * 65535).astype("<u2")
                    process.stdin.write(data.tobytes())
                process.stdin.close()
            except BrokenPipeError:
                process.wait()
                raise _process_error(log, "編碼") from None
            if process.wait():
                raise _process_error(log, "編碼")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass


def _output_directory(prefix, save_output):
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("filename_prefix 不可為空")
    root = Path(folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()).resolve()
    relative = Path(prefix)
    if relative.is_absolute() or ".." in relative.parts or any(char in prefix for char in ("%", "\\", ":")):
        raise ValueError("filename_prefix 必須是 output 內的相對路徑，不可包含 ..、%、反斜線或磁碟代號")
    parent = (root / relative.parent).resolve()
    if not parent.is_relative_to(root):
        raise ValueError("filename_prefix 不可超出 ComfyUI 輸出目錄")
    parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=relative.name + "_", dir=parent))
    return root, directory


class VideoCombineWithMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.01}),
            "loop_count": ("INT", {"default": 0, "min": 0}),
            "filename_prefix": ("STRING", {"default": "mask_video"}),
            "format": (list(FORMATS), {"default": "[mask] ProRes 4444 (.mov)"}),
            "save_output": ("BOOLEAN", {"default": True}),
            "mask_mode": (["error", "separate_png", "ignore"], {"default": "error", "tooltip": "不支援 alpha 的格式：報錯／另存 PNG16 遮罩序列／忽略遮罩"}),
            "save_mask_sequence": ("BOOLEAN", {"default": False, "tooltip": "不論主格式是否支援 alpha，額外保存可重載 MASK 的 PNG16 序列"}),
            "bit_depth": (["auto", "8", "12", "16", "half", "float"], {"default": "auto", "tooltip": "auto：ProRes 要求 12-bit（實際精度受編碼器限制）、VP9/H.264 8、PNG 16、EXR float；EXR 另可選 half"}),
        }, "optional": {"mask": ("MASK",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filenames",)
    FUNCTION = "combine_video"
    OUTPUT_NODE = True
    CATEGORY = "video/mask"
    DESCRIPTION = "CPU 串流輸出影片／高精度序列。MASK 直接作 alpha；可另存 PNG16 遮罩序列。ProRes 以 RGBA16 輸入並要求 yuva444p12le，實際精度受編碼器限制；EXR 保留 float 數值。"

    def combine_video(self, images, frame_rate=24.0, loop_count=0, filename_prefix="mask_video",
                      format="[mask] ProRes 4444 (.mov)", save_output=True, mask_mode="error",
                      save_mask_sequence=False, bit_depth="auto", mask=None):
        mask = _validate_images(images, mask)
        frame_rate = _rate(frame_rate, "frame_rate")
        repetitions = _integer(loop_count, "loop_count", 0) + 1
        if format not in FORMATS:
            raise ValueError(f"不支援的 format：{format}")
        if mask_mode not in ("error", "separate_png", "ignore"):
            raise ValueError("mask_mode 必須是 error、separate_png 或 ignore")
        kind, extension, default_depth = FORMATS[format]
        depth = default_depth if bit_depth == "auto" else bit_depth
        if depth not in (("half", "float") if kind == "exr" else (default_depth,)):
            raise ValueError(f"{format} 不支援 bit_depth={bit_depth}；有效選項為 auto 或 "
                             + ("half／float" if kind == "exr" else default_depth))
        has_mask = mask is not None or images.shape[-1] == 4
        if kind == "h264" and has_mask and mask_mode == "error":
            raise ValueError("H.264 不支援 alpha；mask_mode 請選 separate_png 或 ignore")
        if kind in ("h264", "vp9") and any(size % 2 for size in images.shape[1:3]):
            raise ValueError("H.264／VP9 的 4:2:0 格式需要偶數寬高；不會自動裁切或補邊")
        ffmpeg = find_ffmpeg()
        root, directory = _output_directory(filename_prefix, save_output)
        sequence = kind in ("png", "exr")
        if sequence:
            (directory / "frames").mkdir()
            destination = directory / "frames" / f"%08d.{extension}"
        else:
            destination = directory / f"video.{extension}"
        _encode(destination, kind, depth, images, mask, frame_rate, repetitions, ffmpeg)
        paths = [str(destination)]
        if has_mask and (save_mask_sequence or (kind == "h264" and mask_mode == "separate_png")):
            (directory / "mask").mkdir()
            mask_destination = directory / "mask" / "%08d.png"
            _encode(mask_destination, "png", "16", images, mask, frame_rate, repetitions, ffmpeg, mask_only=True)
            paths.append(str(mask_destination))
        metadata = {"format": format, "bit_depth": depth, "fps": frame_rate,
                    "frame_count": len(images) * repetitions, "alpha_convention": "mask=alpha",
                    "mask_sequence": len(paths) > 1, "files": paths}
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(str(manifest))
        # 預覽專用 8-bit 縮圖不參與母檔編碼；MOV／EXR 的瀏覽器支援不一致。
        from PIL import Image
        thumbnail = directory / "preview.png"
        preview = images[0, ..., :3].detach().to(device="cpu", dtype=torch.float32).numpy()
        Image.fromarray(np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8)).save(thumbnail)
        storage_type = "output" if save_output else "temp"
        subfolder = directory.relative_to(root).as_posix()
        ui = {"images": [{"filename": thumbnail.name, "subfolder": subfolder, "type": storage_type}],
              "mask_video_files": paths}
        if kind in ("vp9", "h264"):
            ui["mask_videos"] = [{"filename": destination.name, "subfolder": subfolder, "type": storage_type}]
        return {"ui": ui, "result": (json.dumps(paths, ensure_ascii=False),)}


NODE_CLASS_MAPPINGS = {LOAD_ID: LoadVideoWithMask, SAVE_ID: VideoCombineWithMask}
NODE_DISPLAY_NAME_MAPPINGS = {LOAD_ID: LOAD_ID, SAVE_ID: SAVE_ID}
