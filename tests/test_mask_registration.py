"""Check package registration without a running ComfyUI or tensor dependencies."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MASK_NODES = {
    "Mask Selection By Index (endman100)": "MaskSelectionByIndex",
    "Mask Selection Max Size (endman100)": "MaskSelectionMaxSize",
    "Mask Selection Min Size (endman100)": "MaskSelectionMinSize",
    "Mask Selection Max Countours (endman100)": "MaskSelectionMaxCountours",
    "Mask Or Mask (endman100)": "MaskOrMask",
    "Mask And Mask (endman100)": "MaskAndMask",
    "Mask Sub Mask (endman100)": "MaskSubMask",
    "Mask Concat (endman100)": "MaskConcatLongImage",
    "Mask Invert (endman100)": "MaskInvert",
    "Mask Repeat (endman100)": "MaskRepeat",
    "Mask Morphology (endman100)": "MaskMorphology",
    "Mask Border (endman100)": "MaskFrameBorder",
    "Mask Temporal Smooth (endman100)": "MaskTemporalSmooth",
    "Fill Mask Area (endman100)": "FillMaskArea",
    "Blend Background (endman100)": "ImageMaskBlendBackground",
    "Add Mask (endman100)": "AddMask",
    "Mask To Region (endman100)": "MaskToRegion",
}


def load_package():
    # Stub only external runtime imports; load every real logic_lite node module.
    stubs = {name: ModuleType(name) for name in (
        "nodes", "folder_paths", "comfy", "comfy.utils",
        "comfy.model_management", "comfy.sdxl_clip", "cv2",
        "torch", "torch.nn", "torch.nn.functional",
    )}
    stubs["nodes"].NODE_CLASS_MAPPINGS = {}
    stubs["nodes"].MAX_RESOLUTION = 16384
    stubs["nodes"].LoraLoader = type("LoraLoader", (), {})
    for name in ("SDXLClipModel", "SDXLRefinerClipModel", "SDXLClipG"):
        setattr(stubs["comfy.sdxl_clip"], name, type(name, (), {}))
    for name, module in stubs.items():
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(stubs[parent], attr, module)

    name = "logic_lite_registration_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py")
    package = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**stubs, name: package}):
        spec.loader.exec_module(package)
    return package


class MaskRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package = load_package()

    def test_preserves_original_and_mask_node_registrations(self):
        package = self.package
        original = {
            **package._LOGIC_CLASS,
            **package._IMAGE_CLASS,
            **package._LORA_UTIL_CLASS,
        }
        self.assertEqual(len(original), 63)
        self.assertFalse(original.keys() & MASK_NODES.keys())
        self.assertEqual(
            {key: cls.__name__ for key, cls in package._MASK_CLASS.items()},
            MASK_NODES,
        )
        self.assertEqual(len(package.NODE_CLASS_MAPPINGS), 63 + 17)
        self.assertEqual(package.NODE_CLASS_MAPPINGS, {**original, **package._MASK_CLASS})
        for key in MASK_NODES:
            node = package.NODE_CLASS_MAPPINGS[key]
            self.assertIsInstance(node.INPUT_TYPES(), dict)
            self.assertTrue(callable(getattr(node, node.FUNCTION)))

    def test_preserves_display_names(self):
        package = self.package
        self.assertEqual(package._MASK_DISPLAY, {key: key for key in MASK_NODES})
        self.assertEqual(package.NODE_DISPLAY_NAME_MAPPINGS, {
            **package._LOGIC_DISPLAY,
            **package._IMAGE_DISPLAY,
            **package._LORA_UTIL_DISPLAY,
            **package._MASK_DISPLAY,
        })
        self.assertEqual(
            package.NODE_DISPLAY_NAME_MAPPINGS.keys(), package.NODE_CLASS_MAPPINGS.keys(),
        )

    def test_serves_mask_extension_alongside_existing_extensions(self):
        web = Path(self.package.WEB_DIRECTORY)
        self.assertEqual(web, ROOT / "web_version" / "v2")
        for name in (
            "mask_and_mask_dynamic_inputs.js", "logiclite.js", "loraUtil.js",
            "loadImageBatch.js", "showAny.js",
        ):
            self.assertTrue((web / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
