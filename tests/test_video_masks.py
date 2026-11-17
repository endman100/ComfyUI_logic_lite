"""影片遮罩的 CPU 測試；參考實作可用迴圈，正式節點的像素運算須向量化。"""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("logic_lite_video_masks", ROOT / "nodes/mask.py")
mask = importlib.util.module_from_spec(spec)
stubs = {} if importlib.util.find_spec("cv2") else {"cv2": ModuleType("cv2")}
with patch.dict(sys.modules, stubs):
    spec.loader.exec_module(mask)

LOGIC_NODES = (mask.VideoMaskOr, mask.VideoMaskAnd)


def invoke(node, *inputs, **options):
    return node().combine(**{f"mask{i}": value for i, value in enumerate(inputs, 1)}, **options)[0]


def morphology(value, **options):
    return mask.VideoMaskMorphology().morphology(value, **options)[0]


def reference_vote(value, radius, threshold):
    return torch.stack([
        (value[max(0, t - radius):t + radius + 1].float().mean(0) > threshold).float()
        for t in range(value.shape[0])
    ])


def reference_spatial(value, method, kernel=3, iterations=1):
    result = (value > 0.5).float().unsqueeze(1)

    def pool(value, erosion):
        sign = -1 if erosion else 1
        padded = F.pad(sign * value, (kernel // 2,) * 4, mode="replicate")
        return sign * F.max_pool2d(padded, kernel, stride=1)

    stages = {"erosion": (True,), "dilation": (False,), "open": (True, False), "close": (False, True)}
    for erosion in stages[method]:
        for _ in range(iterations):
            result = pool(result, erosion)
    return result.squeeze(1)


class VideoLogicTests(unittest.TestCase):
    def assert_output(self, result, expected):
        self.assertEqual(result.dtype, torch.float32)
        self.assertEqual(result.device, expected.device)
        self.assertEqual(result.ndim, 3)
        self.assertTrue(torch.equal(result, expected.float()), (result, expected))
        self.assertTrue(torch.all((result == 0) | (result == 1)))

    def test_one_two_and_many_inputs(self):
        generator = torch.Generator(device="cpu").manual_seed(903)
        for node in LOGIC_NODES:
            for count in (1, 2, 5):
                inputs = [torch.rand((7, 3, 4), generator=generator) for _ in range(count)]
                binary = torch.stack(inputs) > 0.5
                expected = binary.any(0) if node is mask.VideoMaskOr else binary.all(0)
                with self.subTest(node=node.__name__, count=count):
                    self.assert_output(invoke(node, *inputs), expected)

    def test_temporal_radius_five_known_pattern_and_clipped_edges(self):
        # 13 個影格；前 6 格與末格為 1。首格 6/6，中央第 6 格 5/11，末格 1/6。
        source = torch.tensor([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(13, 1, 1)
        averages = torch.tensor([6/6, 6/7, 6/8, 6/9, 6/10, 6/11, 5/11,
                                 5/11, 4/10, 3/9, 2/8, 1/7, 1/6]).reshape(13, 1, 1)
        for node in LOGIC_NODES:
            for threshold in (0.15, 0.5, 0.9):
                result = invoke(node, source, temporal_radius=5, temporal_threshold=threshold)
                self.assert_output(result, averages > threshold)
            self.assertEqual(invoke(node, source, temporal_radius=5)[0].item(), 1)
            self.assertEqual(invoke(node, source, temporal_radius=5)[6].item(), 0)
            self.assertEqual(invoke(node, source, temporal_radius=5, temporal_threshold=0.15)[-1].item(), 1)

    def test_vote_follows_combine(self):
        first = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32).view(5, 1, 1)
        second = 1 - first
        for node in LOGIC_NODES:
            expected = torch.ones_like(first) if node is mask.VideoMaskOr else torch.zeros_like(first)
            self.assert_output(invoke(node, first, second, temporal_radius=1), expected)

    def test_random_temporal_vote_matches_direct_reference(self):
        generator = torch.Generator(device="cpu").manual_seed(515)
        inputs = [torch.rand((19, 3, 5), generator=generator) for _ in range(3)]
        for node in LOGIC_NODES:
            stack = torch.stack(inputs) > 0.4
            combined = stack.any(0) if node is mask.VideoMaskOr else stack.all(0)
            for radius in (1, 5, 100000000000000000000):
                expected = reference_vote(combined, radius, 0.3)
                result = invoke(node, *inputs, threshold=0.4, temporal_radius=radius, temporal_threshold=0.3)
                self.assert_output(result, expected)

    def test_radius_zero_skips_vote_even_at_threshold_one(self):
        source = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float16).view(3, 1, 1)
        for node in LOGIC_NODES:
            self.assert_output(invoke(node, source, temporal_radius=0, temporal_threshold=1), source > 0.5)

    def test_shape_mismatch_t_h_w_reports_both_shapes(self):
        for node in LOGIC_NODES:
            for shape in ((2, 3, 4), (3, 2, 4), (3, 3, 2)):
                with self.subTest(node=node.__name__, shape=shape):
                    with self.assertRaisesRegex(ValueError, r"相同 T,H,W；mask1=\(3, 3, 4\)，mask3="):
                        invoke(node, torch.ones(3, 3, 4), mask3=torch.ones(shape))

    def test_single_frame_tensors_lists_and_channel_axis(self):
        frame = torch.tensor([[0.5, 0.6], [1, 0]], dtype=torch.float64)
        video = torch.stack((frame, 1 - frame))
        for node in LOGIC_NODES:
            for source, expected in (
                (frame, (frame > 0.5).unsqueeze(0)),
                (video.unsqueeze(-1), video > 0.5),
                ([video[:1], video[1:].unsqueeze(-1)], video > 0.5),
                (list(video), video > 0.5),
                (video.tolist(), video > 0.5),
                (frame.tolist(), (frame > 0.5).unsqueeze(0)),
            ):
                self.assert_output(invoke(node, source), expected)

    def test_inverse_is_last_and_inputs_unchanged(self):
        first = torch.tensor([0.9, 0.1, 0.9, 0.1]).reshape(4, 1, 1)
        before = first.clone()
        for node in LOGIC_NODES:
            normal = invoke(node, first, temporal_radius=1)
            self.assert_output(invoke(node, first, temporal_radius=1, mask_inverse=True), 1 - normal)
            self.assertTrue(torch.equal(first, before))

    def test_strict_temporal_threshold_boundary(self):
        source = torch.tensor([1., 0.]).reshape(2, 1, 1)
        for node in LOGIC_NODES:
            self.assert_output(invoke(node, source, temporal_radius=5, temporal_threshold=0.5), torch.zeros_like(source))
            self.assert_output(invoke(node, source, temporal_radius=5, temporal_threshold=0.49), torch.ones_like(source))

    def test_dynamic_ports_allow_gaps_and_validate_names_and_values(self):
        source = torch.ones(2, 3, 4)
        for node in LOGIC_NODES:
            self.assert_output(invoke(node, source, mask10=source, mask3=source), source)
            with self.assertRaisesRegex(ValueError, "mask1"):
                invoke(node, None)
            for name in ("mask0", "mask03", "masks", "merge_all"):
                with self.assertRaisesRegex(ValueError, name):
                    invoke(node, source, **{name: source})
            for value in (None, [], torch.empty(0, 3, 4), torch.ones(2, 3, 4, 2), torch.ones(1, dtype=torch.complex64)):
                with self.assertRaisesRegex((ValueError, TypeError), "mask3"):
                    invoke(node, source, mask3=value)
            with self.assertRaisesRegex(ValueError, "H,W"):
                invoke(node, [torch.ones(3, 4), torch.ones(2, 4)])

    def test_rejects_invalid_temporal_params(self):
        source = torch.ones(2, 1, 1)
        for node in LOGIC_NODES:
            for radius in (-1, 1.5, True):
                with self.assertRaisesRegex(ValueError, "temporal_radius"):
                    invoke(node, source, temporal_radius=radius)
            for name in ("threshold", "temporal_threshold"):
                for value in (-0.1, 1.1, float("nan"), float("inf")):
                    with self.assertRaisesRegex(ValueError, name):
                        invoke(node, source, **{name: value})

    def test_long_video_uses_prefix_sum_without_window_expansion(self):
        source = (torch.arange(10001) % 3 == 0).view(-1, 1, 1)
        for node in LOGIC_NODES:
            result = invoke(node, source, temporal_radius=1000000, temporal_threshold=0.3)
            self.assert_output(result, torch.ones_like(source))


class VideoMorphologyTests(unittest.TestCase):
    def test_temporal_kernel_one_matches_independent_2d_morphology(self):
        generator = torch.Generator(device="cpu").manual_seed(845)
        source = torch.rand((5, 7, 9), generator=generator)
        for method in ("erosion", "dilation", "open", "close"):
            for kernel in (1, 3, 5):
                for iterations in (1, 2):
                    expected = reference_spatial(source, method, kernel, iterations)
                    result = morphology(source, method=method, kernel_size=kernel, iterations=iterations)
                    self.assertTrue(torch.equal(result, expected), (method, kernel, iterations))

    def test_known_3d_dilation_and_erosion_patterns(self):
        impulse = torch.zeros(5, 5, 5)
        impulse[2, 2, 2] = 1
        cube = torch.zeros_like(impulse)
        cube[1:4, 1:4, 1:4] = 1
        self.assertTrue(torch.equal(morphology(impulse, temporal_kernel_size=3), cube))
        self.assertTrue(torch.equal(morphology(cube, method="erosion", temporal_kernel_size=3), impulse))

    def test_known_open_removes_speck_and_preserves_cube(self):
        cube = torch.zeros(7, 9, 9)
        cube[2:5, 3:6, 3:6] = 1
        source = cube.clone()
        source[0, 0, 0] = 1
        self.assertTrue(torch.equal(morphology(source, method="open", temporal_kernel_size=3), cube))

    def test_3d_close_bridges_exactly_one_missing_frame(self):
        # 5 個全白影格的中央影格全黑：[1,1,0,1,1]；時間核心 3 的 close 應補滿。
        source = torch.tensor([1., 1., 0., 1., 1.]).reshape(5, 1, 1).expand(5, 3, 3).clone()
        self.assertTrue(torch.equal(morphology(source, method="close", temporal_kernel_size=3), torch.ones_like(source)))
        self.assertTrue(torch.equal(morphology(source, method="close", temporal_kernel_size=1), source))

    def test_uniform_volume_boundaries_do_not_change(self):
        for value in (0., 1.):
            source = torch.full((2, 3, 4), value)
            for method in ("erosion", "dilation", "open", "close"):
                result = morphology(source, method=method, kernel_size=5, temporal_kernel_size=5)
                self.assertTrue(torch.equal(result, source), (value, method))

    def test_first_last_frame_and_spatial_border_use_replicated_edges(self):
        for last in (False, True):
            frame = -1 if last else 0
            source = torch.zeros(5, 5, 5)
            source[frame, 0, 0] = 1
            expected = torch.zeros_like(source)
            expected[-2 if last else 0:None if last else 2, :2, :2] = 1
            self.assertTrue(torch.equal(morphology(source, temporal_kernel_size=3), expected))
            self.assertTrue(torch.equal(morphology(1 - source, method="erosion", temporal_kernel_size=3), 1 - expected))

    def test_iterations_expand_dilation_and_erosion(self):
        source = torch.zeros(7, 7, 7)
        source[3, 3, 3] = 1
        expected = torch.zeros_like(source)
        expected[1:6, 1:6, 1:6] = 1
        self.assertTrue(torch.equal(morphology(source, temporal_kernel_size=3, iterations=2), expected))
        self.assertTrue(torch.equal(morphology(1 - source, method="erosion", temporal_kernel_size=3, iterations=2), 1 - expected))

    def test_one_pool_call_per_morphology_stage_and_iteration(self):
        source = torch.zeros(3, 4, 5)
        for method, stages in (("dilation", 1), ("erosion", 1), ("open", 2), ("close", 2)):
            with patch.object(mask.F, "max_pool3d", wraps=F.max_pool3d) as pool:
                morphology(source, method=method, temporal_kernel_size=3, iterations=2)
                self.assertEqual(pool.call_count, stages * 2)

    def test_threshold_normalization_dtype_device_and_inverse(self):
        source = torch.tensor([[0.4, 0.5], [0.7, 1.]], dtype=torch.float16)
        for value in (source, [source], source.unsqueeze(0).unsqueeze(-1)):
            result = morphology(value, kernel_size=1, threshold=0.5, mask_inverse=True)
            self.assertTrue(torch.equal(result, (source <= 0.5).float().unsqueeze(0)))
            self.assertEqual(result.dtype, torch.float32)
            self.assertEqual(result.device.type, "cpu")
        volume = torch.zeros(3, 5, 5)
        volume[1, 2, 2] = 1
        before = volume.clone()
        for method in ("erosion", "dilation", "open", "close"):
            options = dict(method=method, temporal_kernel_size=3)
            self.assertTrue(torch.equal(morphology(volume, **options, mask_inverse=True), 1 - morphology(volume, **options)))
        self.assertTrue(torch.equal(volume, before))

    def test_validates_odd_positive_kernels_and_iterations(self):
        for name in ("kernel_size", "temporal_kernel_size"):
            for value in (0, -1, 2, 4, 1.5, True):
                with self.assertRaisesRegex(ValueError, name):
                    morphology(torch.ones(1, 1, 1), **{name: value})
        for value in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, "iterations"):
                morphology(torch.ones(1, 1, 1), iterations=value)
        for options in ({"method": "erode"}, {"kernel_shape": "ellipse"}, {"threshold": float("nan")}):
            with self.assertRaises(ValueError):
                morphology(torch.ones(1, 1, 1), **options)

    def test_single_input_contract_rejects_vote_or_extra_masks(self):
        for name in ("mask2", "temporal_radius", "temporal_threshold"):
            with self.assertRaises(TypeError):
                morphology(torch.ones(1, 1, 1), **{name: 1})


class VideoSchemaTests(unittest.TestCase):
    def test_registrations_and_defaults(self):
        for title, node in (("Video Mask Or", mask.VideoMaskOr), ("Video Mask And", mask.VideoMaskAnd),
                            ("Video Mask Morphology", mask.VideoMaskMorphology)):
            name = f"{title} (endman100)"
            self.assertIs(mask.NODE_CLASS_MAPPINGS[name], node)
            self.assertEqual(mask.NODE_DISPLAY_NAME_MAPPINGS[name], name)
            self.assertEqual(node.CATEGORY, "mask/video")
            self.assertEqual(node.RETURN_TYPES, ("MASK",))
            self.assertFalse(getattr(node, "INPUT_IS_LIST", False))
        for node in LOGIC_NODES:
            schema = node.INPUT_TYPES()
            self.assertEqual(schema["optional"], {"mask2": ("MASK",)})
            defaults = {name: info[1]["default"] for name, info in schema["required"].items() if name != "mask1"}
            self.assertEqual(defaults, dict(threshold=0.5, temporal_radius=0, temporal_threshold=0.5, mask_inverse=False))
        schema = mask.VideoMaskMorphology.INPUT_TYPES()
        self.assertNotIn("optional", schema)
        defaults = {name: info[1]["default"] for name, info in schema["required"].items() if name != "mask"}
        self.assertEqual(defaults, dict(method="dilation", kernel_size=3, temporal_kernel_size=1,
                                        kernel_shape="rect", iterations=1, threshold=0.5, mask_inverse=False))


if __name__ == "__main__":
    unittest.main()
