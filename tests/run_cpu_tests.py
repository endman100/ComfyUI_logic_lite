"""Run all Python tests without starting ComfyUI or importing GPU initialization.

Usage: python -B tests/run_cpu_tests.py --comfy-root /path/to/ComfyUI

Requires CPU PyTorch, NumPy, Pillow and PyYAML. The legacy image tests compare
against three actual ComfyUI classes, AST-loaded unchanged from the supplied
checkout. A small IO schema/output adapter and CPU intermediate_device replace
the surrounding application only; all tensor calculations use real PyTorch.
OpenCV is not needed by the four mask operations under test.
"""

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch


class CoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {item.name: (item.kind, item.options) for item in cls.define_schema().inputs}}


class NodeOutput:
    def __init__(self, *values):
        self.result = values


def io_type(kind):
    return SimpleNamespace(
        Input=lambda name, **options: SimpleNamespace(name=name, kind=kind, options=options),
        Output=lambda **options: None,
    )


def install_core_references(root):
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise RuntimeError("Use a CPU-only PyTorch build for this test runner")
    torch.set_num_threads(1)
    comfy = ModuleType("comfy")
    management = ModuleType("comfy.model_management")
    management.intermediate_device = lambda: torch.device("cpu")
    comfy.model_management = management
    nodes = ModuleType("nodes")
    node_tree = ast.parse((root / "nodes.py").read_text())
    nodes.MAX_RESOLUTION = next(ast.literal_eval(item.value) for item in node_tree.body
        if isinstance(item, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "MAX_RESOLUTION"
                                               for target in item.targets))
    io = SimpleNamespace(ComfyNode=CoreNode, NodeOutput=NodeOutput,
        Schema=lambda **options: SimpleNamespace(**options),
        Float=io_type("FLOAT"), Int=io_type("INT"), Mask=io_type("MASK"), Image=io_type("IMAGE"))
    extras = ModuleType("comfy_extras")
    sys.modules.update({"comfy": comfy, "comfy.model_management": management,
                        "nodes": nodes, "comfy_extras": extras})
    for filename, selected in (("nodes_mask", {"SolidMask", "MaskToImage"}),
                               ("nodes_images", {"RepeatImageBatch"})):
        source = root / "comfy_extras" / f"{filename}.py"
        tree = ast.parse(source.read_text())
        classes = [item for item in tree.body if isinstance(item, ast.ClassDef) and item.name in selected]
        if {item.name for item in classes} != selected:
            raise RuntimeError(f"Missing CPU reference classes in {source}")
        module = ModuleType(f"comfy_extras.{filename}")
        module.__dict__.update(torch=torch, comfy=comfy, nodes=nodes, IO=io)
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), module.__dict__)
        sys.modules[module.__name__] = module
        setattr(extras, filename, module)
        print(f"CPU reference: {source}: {', '.join(sorted(selected))}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", type=Path, required=True)
    options = parser.parse_args()
    install_core_references(options.comfy_root)
    print(f"PyTorch {torch.__version__}; CUDA build: {torch.version.cuda}; CPU only", flush=True)
    # 先載入原生 OpenCV，避免其他測試的 sys.modules 隔離導致反覆初始化原生子模組。
    if importlib.util.find_spec("cv2") is not None:
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
        import cv2
        print(f"OpenCV {cv2.__version__}: {cv2.__file__}", flush=True)
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
