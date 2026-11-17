"""Real CPU tensor tests for the four dynamic mask nodes."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("logic_lite_mask_operations", ROOT / "nodes/mask.py")
mask = importlib.util.module_from_spec(spec)
# OpenCV is unused by these four nodes; keep their tensor tests runnable alone.
stubs = {} if importlib.util.find_spec("cv2") else {"cv2": ModuleType("cv2")}
with patch.dict(sys.modules, stubs):
    spec.loader.exec_module(mask)

LOGIC_NODES = {"or": mask.MaskOrMask, "and": mask.MaskAndMask, "sub": mask.MaskSubMask}
ALL_NODES = (*LOGIC_NODES.values(), mask.MaskConcatLongImage)


def invoke(node, inputs, **options):
    kwargs = {f"mask{i}": value for i, value in enumerate(inputs, 1)}
    return node().method(**kwargs, **options)[0]


def reference(operation, inputs, merge_all, threshold=0.5):
    binary = [value > threshold for value in inputs]
    if merge_all:
        flat = torch.cat(binary)
        if operation == "or":
            return flat.any(dim=0, keepdim=True)
        if operation == "and":
            return flat.all(dim=0, keepdim=True)
        return flat[:1] & ~flat[1:].any(dim=0, keepdim=True)
    stacked = torch.stack(binary)
    if operation == "or":
        return stacked.any(dim=0)
    if operation == "and":
        return stacked.all(dim=0)
    return stacked[0] & ~stacked[1:].any(dim=0)


class MaskOperationTests(unittest.TestCase):
    def test_one_two_and_many_ports_match_boolean_reference(self):
        generator = torch.Generator(device="cpu").manual_seed(173)
        for operation, node in LOGIC_NODES.items():
            for count in (1, 2, 5):
                inputs = [torch.rand((3, 4, 7), generator=generator) for _ in range(count)]
                for merge_all in (False, True):
                    with self.subTest(operation=operation, count=count, merge_all=merge_all):
                        result = invoke(node, inputs, merge_all=merge_all)
                        expected = reference(operation, inputs, merge_all).float()
                        self.assertTrue(torch.equal(result, expected))
                        self.assertEqual(result.shape, (1 if merge_all else 3, 4, 7))

    def test_merge_all_accepts_different_batch_lengths(self):
        inputs = [torch.full((size, 3, 5), value) for size, value in ((2, 0.8), (1, 0.2), (4, 0.7))]
        for operation, node in LOGIC_NODES.items():
            with self.subTest(operation=operation):
                result = invoke(node, inputs, merge_all=True)
                self.assertTrue(torch.equal(result, reference(operation, inputs, True).float()))
                self.assertEqual(result.shape, (1, 3, 5))

    def test_single_batch_input_false_is_thresholded_passthrough(self):
        value = torch.tensor([[[0.1, 0.5, 0.9]], [[0.7, 0.2, 1.0]]])
        for node in LOGIC_NODES.values():
            self.assertTrue(torch.equal(node().method(value)[0], (value > 0.5).float()))

    def test_sub_true_subtracts_within_single_input_batch(self):
        value = torch.tensor([[[1., 1., 1.]], [[0., 1., 0.]], [[0., 0., 1.]]])
        result, = mask.MaskSubMask().method(value, merge_all=True)
        self.assertTrue(torch.equal(result, torch.tensor([[[1., 0., 0.]]])))
        identical = torch.ones((2, 3, 5))
        self.assertFalse(mask.MaskSubMask().method(identical, merge_all=True)[0].any())

    def test_sub_true_uses_first_frame_not_union_of_first_port(self):
        first = torch.tensor([[[1., 1., 0.]], [[0., 1., 1.]]])
        second = torch.tensor([[[0., 0., 0.]]])
        result, = mask.MaskSubMask().method(first, second, True)
        self.assertTrue(torch.equal(result, torch.tensor([[[1., 0., 0.]]])))
        swapped, = mask.MaskSubMask().method(second, first, True)
        self.assertFalse(swapped.any())

    def test_sub_false_uses_matching_batch_indices(self):
        first = torch.ones((2, 1, 3))
        second = torch.tensor([[[0., 1., 0.]], [[1., 0., 0.]]])
        third = torch.tensor([[[0., 0., 1.]], [[0., 1., 0.]]])
        result, = mask.MaskSubMask().method(first, second, False, mask3=third)
        self.assertTrue(torch.equal(result, torch.tensor([[[1., 0., 0.]], [[0., 0., 1.]]])))

    def test_sub_with_missing_optional_port_keeps_mask1_as_base(self):
        first = torch.ones((1, 2, 3))
        third = torch.zeros_like(first)
        self.assertTrue(torch.equal(mask.MaskSubMask().method(first, mask3=third)[0], first))
        with self.assertRaisesRegex(ValueError, "mask1 is required"):
            mask.MaskSubMask().method(None, mask3=first)

    def test_false_rejects_batch_mismatch_including_singleton(self):
        for node in LOGIC_NODES.values():
            for first_b, second_b in ((2, 1), (1, 2), (2, 3)):
                with self.subTest(node=node.__name__, batches=(first_b, second_b)):
                    with self.assertRaisesRegex(ValueError, "same batch size"):
                        node().method(torch.ones((first_b, 3, 5)), torch.ones((second_b, 3, 5)))

    def test_spatial_mismatch_never_broadcasts(self):
        for node in LOGIC_NODES.values():
            for merge_all in (False, True):
                for shape in ((2, 1, 5), (2, 3, 1), (2, 4, 5), (2, 3, 6)):
                    with self.subTest(node=node.__name__, merge_all=merge_all, shape=shape):
                        with self.assertRaisesRegex(ValueError, "same height and width"):
                            node().method(torch.ones((2, 3, 5)), torch.ones(shape), merge_all)

    def test_mixed_list_and_tensor_inputs_normalize_without_losing_frames(self):
        first = [torch.zeros((3, 5)), torch.ones((1, 3, 5)), torch.full((1, 3, 5, 1), 0.25)]
        normalized = torch.cat((first[0].unsqueeze(0), first[1], first[2].squeeze(-1)))
        second = torch.full((3, 3, 5), 0.6)
        third = [torch.ones((3, 5)) for _ in range(3)]
        for operation, node in LOGIC_NODES.items():
            for merge_all in (False, True):
                result = invoke(node, [first, second, third], merge_all=merge_all)
                expected = reference(operation, [normalized, second, torch.stack(third)], merge_all)
                self.assertIsInstance(result, torch.Tensor)
                self.assertTrue(torch.equal(result, expected.float()))

    def test_list_dtype_promotion_does_not_turn_soft_values_into_booleans(self):
        value = [torch.zeros((2, 3), dtype=torch.bool), torch.full((2, 3), 0.2)]
        result, = mask.MaskOrMask().method(value)
        self.assertEqual(result.shape, (2, 2, 3))
        self.assertFalse(result.any())

    def test_unequal_list_lengths_raise_instead_of_zip_truncation(self):
        value = torch.ones((3, 5))
        for node in LOGIC_NODES.values():
            with self.assertRaisesRegex(ValueError, "same batch size"):
                node().method([value, value], [value])
            self.assertEqual(node().method([value, value], [value], True)[0].shape, (1, 3, 5))

    def test_2d_and_single_channel_4d_inputs(self):
        for node in LOGIC_NODES.values():
            result = node().method(torch.ones((3, 5)), torch.zeros((1, 3, 5, 1)))[0]
            self.assertEqual(result.shape, (1, 3, 5))

    def test_rejects_zero_inputs_and_empty_batches(self):
        for node in ALL_NODES:
            with self.subTest(node=node.__name__):
                with self.assertRaises(TypeError):
                    node().method()
                with self.assertRaisesRegex(ValueError, "mask1 is required"):
                    node().method(None)
                for value in ([], torch.empty((0, 3, 5)), torch.empty((1, 0, 5)), torch.empty((1, 3, 0))):
                    with self.assertRaisesRegex(ValueError, "empty|positive"):
                        node().method(value)

    def test_all_extra_ports_validate_type_and_shape(self):
        valid = torch.ones((1, 3, 5))
        bad_values = (None, 42, "mask", {}, [valid, "bad"], torch.ones((5,)),
                      torch.ones((1, 3, 5, 2)), torch.ones((1, 1, 3, 5, 1)),
                      torch.empty((0, 3, 5)), [], torch.ones((1, 3, 5), dtype=torch.complex64))
        for node in ALL_NODES:
            for name in ("mask3", "mask10"):
                for value in bad_values:
                    with self.subTest(node=node.__name__, name=name, value=type(value).__name__):
                        with self.assertRaisesRegex((TypeError, ValueError), name):
                            node().method(valid, **{name: value})

    def test_extra_ports_validate_spatial_and_batch_sizes(self):
        for node in LOGIC_NODES.values():
            for merge_all in (False, True):
                with self.assertRaisesRegex(ValueError, "same height and width"):
                    node().method(torch.ones((2, 3, 5)), merge_all=merge_all, mask3=torch.ones((2, 1, 5)))
            with self.assertRaisesRegex(ValueError, "same batch size"):
                node().method(torch.ones((2, 3, 5)), mask10=torch.ones((1, 3, 5)))

    def test_list_items_validate_shapes_individually(self):
        for node in ALL_NODES:
            with self.assertRaisesRegex(ValueError, "mask3 list items.*same height and width"):
                node().method(torch.ones((1, 3, 5)), mask3=[torch.ones((3, 5)), torch.ones((1, 5))])

    def test_threshold_and_inverse_are_applied_in_order(self):
        first = torch.tensor([[[0.3, 0.4, 0.8]], [[0.9, 0.1, 0.4]]], dtype=torch.float16)
        second = torch.full_like(first, 0.2)
        for operation, node in LOGIC_NODES.items():
            for merge_all in (False, True):
                expected = reference(operation, [first, second], merge_all, threshold=0.4)
                result, = node().method(first, second, merge_all, True, 0.4)
                self.assertTrue(torch.equal(result, (~expected).to(first.dtype)))
                self.assertEqual(result.dtype, first.dtype)
                self.assertEqual(result.device.type, "cpu")

    def test_boolean_inverse_and_input_immutability(self):
        first = torch.tensor([[[True, False]], [[False, True]]])
        before = first.clone()
        for node in LOGIC_NODES.values():
            result, = node().method(first, mask_inverse=True)
            self.assertEqual(result.dtype, torch.bool)
            self.assertTrue(torch.equal(result, ~first))
            self.assertTrue(torch.equal(first, before))

    def test_schema_preserves_optional_mask2_and_widget_order(self):
        for node in ALL_NODES:
            schema = node.INPUT_TYPES()
            self.assertEqual(schema["required"]["mask1"], ("MASK",))
            self.assertNotIn("mask2", schema["required"])
            self.assertEqual(schema["optional"]["mask2"], ("MASK",))
            self.assertFalse(getattr(node, "INPUT_IS_LIST", False))
        for node in LOGIC_NODES.values():
            self.assertEqual(list(node.INPUT_TYPES()["required"]), ["mask1", "merge_all", "mask_inverse", "threshold"])


class LongConcatTests(unittest.TestCase):
    def test_single_input_is_soft_mask_and_rgb_preview(self):
        value = torch.tensor([[[0.1, 0.3, 0.7], [0.2, 0.5, 0.9]]])
        result, preview = mask.MaskConcatLongImage().method(value)
        self.assertTrue(torch.equal(result, value))
        self.assertTrue(torch.equal(preview, value.unsqueeze(-1).expand(-1, -1, -1, 3)))
        self.assertEqual(preview.shape, (1, 2, 3, 3))

    def test_two_inputs_join_width_preserving_batches_and_values(self):
        first = torch.full((2, 3, 2), 0.25)
        second = torch.full((2, 3, 4), 0.75)
        result, preview = mask.MaskConcatLongImage().method(first, second)
        self.assertTrue(torch.equal(result, torch.cat((first, second), dim=2)))
        self.assertEqual(preview.shape, (2, 3, 6, 3))

    def test_many_inputs_sort_numeric_suffixes_and_allow_gaps(self):
        parts = {name: torch.full((2, 3, 1), value) for name, value in
                 (("mask1", 0.1), ("mask2", 0.2), ("mask3", 0.3), ("mask4", 0.4), ("mask10", 1.0))}
        result, _ = mask.MaskConcatLongImage().method(parts["mask1"], parts["mask2"],
                    mask10=parts["mask10"], mask4=parts["mask4"], mask3=list(parts["mask3"]))
        self.assertTrue(torch.equal(result, torch.cat(list(parts.values()), dim=2)))
        result, _ = mask.MaskConcatLongImage().method(parts["mask1"], mask10=parts["mask10"])
        self.assertTrue(torch.equal(result, torch.cat((parts["mask1"], parts["mask10"]), dim=2)))

    def test_resize_preserves_nearest_neighbor_values_and_preview(self):
        first = torch.linspace(0, 1, 2 * 4 * 6).reshape(2, 4, 6)
        second = torch.full((2, 4, 4), 0.35)
        result, preview = mask.MaskConcatLongImage().method(first, second, 0.5)
        expected = torch.cat((first[:, ::2, ::2], second[:, ::2, ::2]), dim=2)
        self.assertTrue(torch.equal(result, expected))
        self.assertTrue(torch.equal(preview, result.unsqueeze(-1).expand(-1, -1, -1, 3)))

    def test_height_is_checked_after_resize(self):
        result, _ = mask.MaskConcatLongImage().method(torch.ones((1, 3, 2)), torch.ones((1, 4, 2)), 0.5)
        self.assertEqual(result.shape, (1, 2, 2))

    def test_rejects_batch_or_resized_height_mismatch(self):
        for shape, message in (((1, 3, 5), "same batch size"), ((2, 4, 5), "same height after resize")):
            with self.assertRaisesRegex(ValueError, message):
                mask.MaskConcatLongImage().method(torch.ones((2, 3, 5)), mask3=torch.ones(shape))

    def test_rejects_invalid_resize(self):
        for scale in (0, -0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                mask.MaskConcatLongImage().method(torch.ones((1, 3, 5)), downscale_resize=scale)

    def test_schema_has_no_merge_all_and_keeps_existing_registration(self):
        schema = mask.MaskConcatLongImage.INPUT_TYPES()
        self.assertEqual(list(schema["required"]), ["mask1", "downscale_resize"])
        self.assertNotIn("merge_all", schema["required"] | schema["optional"])
        self.assertIs(mask.NODE_CLASS_MAPPINGS["Mask Concat (endman100)"], mask.MaskConcatLongImage)
        self.assertEqual(len(mask.NODE_CLASS_MAPPINGS), 17)


if __name__ == "__main__":
    unittest.main()
