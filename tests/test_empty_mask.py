import importlib.util
from pathlib import Path
import unittest

import torch

from comfy_extras.nodes_mask import MaskToImage, SolidMask


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "image.py"
spec = importlib.util.spec_from_file_location("logic_lite_image", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class EmptyMaskTests(unittest.TestCase):
    def test_single_mask_matches_original(self):
        for value in (0.0, 0.35, 1.0):
            with self.subTest(value=value):
                mask, = module.emptyMask().solid(value, 17, 9)
                original, = SolidMask.execute(value, 17, 9).result

                self.assertTrue(torch.equal(mask, original))
                self.assertEqual(mask.dtype, original.dtype)
                self.assertEqual(mask.device, original.device)


    def test_batch_masks_have_independent_storage(self):
        mask, = module.emptyMask().solid(0.25, 13, 7, batch_size=4)

        self.assertEqual(mask.shape, (4, 7, 13))
        self.assertTrue(torch.all(mask == 0.25))
        mask[0, 0, 0] = 1.0
        self.assertTrue(torch.all(mask[1:] == 0.25))


    def test_batch_connects_to_core_mask_to_image(self):
        mask, = module.emptyMask().solid(0.5, 13, 7, batch_size=3)
        images, = MaskToImage.execute(mask).result

        self.assertEqual(images.shape, (3, 7, 13, 3))
        self.assertTrue(torch.all(images == 0.5))


    def test_registers_node_and_preserves_solid_mask_inputs(self):
        node_id = "logic emptyMask"
        required = module.emptyMask.INPUT_TYPES()["required"]
        original = SolidMask.INPUT_TYPES()["required"]

        self.assertIs(module.NODE_CLASS_MAPPINGS[node_id], module.emptyMask)
        self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS[node_id], "Empty Mask")
        self.assertEqual(module.emptyMask.RETURN_TYPES, ("MASK",))
        for name in ("value", "width", "height"):
            self.assertEqual(required[name], original[name])
        self.assertEqual(required["batch_size"][1], {"default": 1, "min": 1, "max": 4096})
