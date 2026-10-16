import importlib.util
from pathlib import Path
import unittest

import torch

from comfy_extras.nodes_images import RepeatImageBatch
from comfy_extras.nodes_mask import MaskToImage


MODULE_PATH = Path(__file__).parents[1] / "nodes" / "image.py"
spec = importlib.util.spec_from_file_location("logic_lite_repeat_image", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RepeatNodeTests(unittest.TestCase):
    def test_image_batch_matches_core_repeat_order(self):
        image = torch.arange(2 * 3 * 5 * 3, dtype=torch.float16).reshape(2, 3, 5, 3)
        result, = module.repeatImage().repeat(image, 3)
        original, = RepeatImageBatch.execute(image, 3).result

        self.assertEqual(result.shape, (6, 3, 5, 3))
        self.assertTrue(torch.equal(result, original))
        self.assertEqual(result.dtype, image.dtype)
        self.assertEqual(result.device, image.device)

    def test_single_image_and_amount_one(self):
        image = torch.full((1, 3, 5, 3), 0.25)
        result, = module.repeatImage().repeat(image, 4)
        self.assertEqual(result.shape, (4, 3, 5, 3))
        self.assertTrue(torch.all(result == 0.25))

        unchanged, = module.repeatImage().repeat(image)
        self.assertTrue(torch.equal(unchanged, image))
        unchanged[0, 0, 0, 0] = 1.0
        self.assertEqual(image[0, 0, 0, 0].item(), 0.25)

    def test_mask_batch_repeats_all_input_masks(self):
        mask = torch.stack((torch.zeros((3, 5)), torch.ones((3, 5)))).half()
        result, = module.repeatMask().repeat(mask, 3)

        self.assertEqual(result.shape, (6, 3, 5))
        for index in range(6):
            self.assertTrue(torch.equal(result[index], mask[index % 2]))
        self.assertEqual(result.dtype, mask.dtype)
        self.assertEqual(result.device, mask.device)
        result[0, 0, 0] = 0.5
        self.assertEqual(result[2, 0, 0].item(), 0.0)
        self.assertEqual(mask[0, 0, 0].item(), 0.0)

    def test_single_2d_mask_keeps_batch_dimension(self):
        result, = module.repeatMask().repeat(torch.full((3, 5), 0.75), 4)
        self.assertEqual(result.shape, (4, 3, 5))
        self.assertTrue(torch.all(result == 0.75))

    def test_mask_amount_one_and_core_image_conversion(self):
        mask = torch.full((2, 3, 5), 0.5)
        result, = module.repeatMask().repeat(mask)
        images, = MaskToImage.execute(result).result

        self.assertTrue(torch.equal(result, mask))
        self.assertEqual(images.shape, (2, 3, 5, 3))
        self.assertTrue(torch.all(images == 0.5))

    def test_empty_batches_remain_empty(self):
        image, = module.repeatImage().repeat(torch.empty((0, 3, 5, 3)), 3)
        mask, = module.repeatMask().repeat(torch.empty((0, 3, 5)), 3)
        self.assertEqual(image.shape, (0, 3, 5, 3))
        self.assertEqual(mask.shape, (0, 3, 5))

    def test_registers_both_nodes(self):
        for node_id, cls, name, output in [
            ("logic repeatImage", module.repeatImage, "Repeat Image", "IMAGE"),
            ("logic repeatMask", module.repeatMask, "Repeat Mask", "MASK"),
        ]:
            with self.subTest(node_id=node_id):
                self.assertIs(module.NODE_CLASS_MAPPINGS[node_id], cls)
                self.assertEqual(module.NODE_DISPLAY_NAME_MAPPINGS[node_id], name)
                self.assertEqual(cls.RETURN_TYPES, (output,))
                self.assertEqual(cls.INPUT_TYPES()["required"]["amount"][1]["default"], 1)
