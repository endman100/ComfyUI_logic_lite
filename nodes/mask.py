import os
import cv2
import torch
import torch.nn.functional as F
import shutil
from datetime import datetime
import numpy as np
import time
from collections.abc import Iterable


def _invert_mask_output(mask, mask_inverse):
    if not mask_inverse:
        return mask
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            return ~mask
        return 1.0 - mask
    if isinstance(mask, list):
        return [_invert_mask_output(_mask, mask_inverse) for _mask in mask]
    raise ValueError("mask is not list or tensor", getattr(mask, "shape", None), type(mask))


class MaskSelectionByIndex:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def selection(self, mask, index=0, mask_inverse=False):
        # print("selection", mask, isinstance(mask, Iterable), mask.shape)
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        if index >= len(mask):
            raise ValueError("index out of range", index, len(mask))


        if isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            result = mask[index].unsqueeze(0)
        elif isinstance(mask, list): # 判断 mask 是否是数组
            result = [mask[index]]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (_invert_mask_output(result, mask_inverse),) # 返回结果 shape 为 (1, h, w)

class MaskSelectionMaxSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def selection(self, mask, mask_inverse=False):
        # print("selection", mask, isinstance(mask, Iterable), mask.shape)
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        if isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            min_size = 0
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size > min_size:
                    min_size = _mask.size
                    min_index = index
            result = mask[min_index].unsqueeze(0)
        elif isinstance(mask, list): # 判断 mask 是否是数组
            min_size = 0
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size > min_size:
                    min_size = _mask.size
                    min_index = index
            result = [mask[min_index]]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (_invert_mask_output(result, mask_inverse),)

class MaskSelectionMinSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def selection(self, mask, mask_inverse=False):
        # print("selection", mask, isinstance(mask, Iterable), mask.shape)
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        if isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            min_size = np.inf
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size < min_size:
                    min_size = _mask.size
                    min_index = index
            result = mask[min_index].unsqueeze(0)
        elif isinstance(mask, list): # 判断 mask 是否是数组
            min_size = np.inf
            min_index = 0
            for index, _mask in enumerate(mask):
                _mask.size = torch.sum(_mask)
                if _mask.size < min_size:
                    min_size = _mask.size
                    min_index = index
            result = [mask[min_index]]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (_invert_mask_output(result, mask_inverse),)
class MaskSelectionMaxCountours:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "merge_all": ("BOOLEAN", {"default": True}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "selection"
    CATEGORY = "mask"

    def _largest_contour_mask(self, mask):
        source_device = mask.device
        source_dtype = mask.dtype
        mask_np = (mask.detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
        countours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_countour = None
        max_countour_area = 0
        for countour in countours:
            area = cv2.contourArea(countour)
            if area > max_countour_area:
                max_countour_area = area
                max_countour = countour

        if max_countour is None:
            print("can't find any countour is None")
            return mask

        h, w = mask.shape[-2], mask.shape[-1]
        result = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(result, [max_countour], -1, 1, -1)
        return torch.tensor(result, device=source_device).to(source_dtype)

    def selection(self, mask, merge_all=True, mask_inverse=False):
        if(not isinstance(mask, Iterable)):
            raise ValueError("mask is not iterable", mask.shape, type(mask))
        elif(len(mask) == 0):
            return (_invert_mask_output(mask, mask_inverse),)
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            result = self._largest_contour_mask(mask).unsqueeze(0)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 3: #batch, h, w
            if merge_all:
                merged_mask = torch.any(mask > 0.5, dim=0)
                result = self._largest_contour_mask(merged_mask.to(mask.dtype)).unsqueeze(0)
            else:
                result = torch.stack([self._largest_contour_mask(_mask) for _mask in mask])
        elif isinstance(mask, list): # 判断 mask 是否是数组
            if merge_all:
                original_dtype = mask[0].dtype
                original_device = mask[0].device
                merged_mask = torch.stack([_mask > 0.5 for _mask in mask]).any(dim=0)
                result = self._largest_contour_mask(merged_mask.to(original_dtype).to(original_device)).unsqueeze(0)
            else:
                result = [self._largest_contour_mask(_mask) for _mask in mask]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (_invert_mask_output(result, mask_inverse),)

def _collect_numbered_masks(mask1, mask2, kwargs):
    if mask1 is None:
        raise ValueError("mask1 is required; connect at least one mask")
    masks = [("mask1", mask1)]
    if mask2 is not None:
        masks.append(("mask2", mask2))
    extra_mask_names = [
        name for name in kwargs
        if name.startswith("mask") and name[4:].isdigit() and int(name[4:]) > 2
    ]
    for name in sorted(extra_mask_names, key=lambda item: int(item[4:])):
        masks.append((name, kwargs[name]))
    return [_normalize_mask_sequence(mask, name) for name, mask in masks]


def _normalize_mask_tensor(mask, name="mask"):
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a tensor, got {type(mask).__name__}")
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    elif mask.dim() == 4 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.dim() != 3:
        raise ValueError(f"{name} must have shape [H,W], [B,H,W], or [B,H,W,1]; got {tuple(mask.shape)}")
    if any(size == 0 for size in mask.shape):
        raise ValueError(f"{name} must have a non-empty batch and positive height/width; got {tuple(mask.shape)}")
    if mask.is_complex():
        raise TypeError(f"{name} must contain real mask values")
    return mask


def _combine_mask_tensors(masks, operation, merge_all, threshold):
    first = masks[0]
    if any(mask.shape[1:] != first.shape[1:] for mask in masks):
        raise ValueError("all mask inputs must have the same height and width", [tuple(mask.shape) for mask in masks])
    if not merge_all and any(mask.shape[0] != first.shape[0] for mask in masks):
        raise ValueError("all mask inputs must have the same batch size when merge_all=False; broadcasting is not allowed",
                         [mask.shape[0] for mask in masks])

    result = first > threshold
    if merge_all:
        if operation == "or":
            result = result.any(dim=0, keepdim=True)
        elif operation == "and":
            result = result.all(dim=0, keepdim=True)
        elif operation == "sub":
            # Flat order is mask1's entire batch, then mask2's batch, etc.
            result = result[:1] & ~result[1:].any(dim=0, keepdim=True)

    for mask in masks[1:]:
        mask_bool = mask.to(device=first.device) > threshold
        if merge_all:
            mask_bool = (mask_bool.all(dim=0, keepdim=True) if operation == "and"
                         else mask_bool.any(dim=0, keepdim=True))
        if operation == "or":
            result = torch.logical_or(result, mask_bool)
        elif operation == "and":
            result = torch.logical_and(result, mask_bool)
        elif operation == "sub":
            result = torch.logical_and(result, ~mask_bool)
        else:
            raise ValueError("unknown mask operation", operation)

    return result.to(first.dtype)


def _combine_numbered_masks(mask1, mask2, merge_all, operation, mask_inverse, threshold, kwargs):
    masks = _collect_numbered_masks(mask1, mask2, kwargs)

    result = _combine_mask_tensors(masks, operation, merge_all, threshold)
    return (_invert_mask_output(result, mask_inverse),)


def _normalize_mask_sequence(mask, name="mask"):
    """Normalize tensor/list batches without truncation or spatial broadcasting."""
    if not isinstance(mask, list):
        return _normalize_mask_tensor(mask, name)
    if not mask:
        raise ValueError(f"{name} list must not be empty")
    items = [_normalize_mask_tensor(item, f"{name}[{index}]") for index, item in enumerate(mask)]
    first = items[0]
    if any(item.shape[1:] != first.shape[1:] for item in items):
        raise ValueError(f"{name} list items must have the same height and width", [tuple(item.shape) for item in items])
    return torch.cat([item.to(device=first.device) for item in items], dim=0)


class MaskConcatLongImage:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "downscale_resize": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
            },
            "optional": {"mask2": ("MASK",)},
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "method"
    CATEGORY = "mask"

    def _resize_mask_sequence(self, mask, downscale_resize):
        if downscale_resize == 1.0:
            return mask

        h, w = mask.shape[-2], mask.shape[-1]
        target_h = max(1, int(round(h * downscale_resize)))
        target_w = max(1, int(round(w * downscale_resize)))
        mask = mask.unsqueeze(1).float()
        mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        return mask.squeeze(1)

    def method(self, mask1, mask2=None, downscale_resize=1.0, **kwargs):
        downscale_resize = float(downscale_resize)
        if not 0.0 < downscale_resize <= 1.0:
            raise ValueError("downscale_resize must be greater than 0.0 and less than or equal to 1.0", downscale_resize)

        masks = _collect_numbered_masks(mask1, mask2, kwargs)
        first_length = masks[0].shape[0]
        lengths = [mask.shape[0] for mask in masks]
        if any(length != first_length for length in lengths):
            raise ValueError("all mask inputs must have the same batch size (temporal length)", lengths)

        device = masks[0].device
        resized_masks = [
            self._resize_mask_sequence(mask.to(device=device), downscale_resize).clamp(0.0, 1.0)
            for mask in masks
        ]

        heights = [mask.shape[-2] for mask in resized_masks]
        if any(height != heights[0] for height in heights):
            raise ValueError("all mask inputs must have the same height after resize", heights)

        long_mask = torch.cat(resized_masks, dim=2)
        image = long_mask.unsqueeze(-1).repeat(1, 1, 1, 3).float()
        return (long_mask, image)


class MaskOrMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "merge_all": ("BOOLEAN", {"default": False}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"mask2": ("MASK",)},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask1, mask2=None, merge_all=False, mask_inverse=False, threshold=0.5, **kwargs):
        return _combine_numbered_masks(mask1, mask2, merge_all, "or", mask_inverse, threshold, kwargs)

class MaskAndMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "merge_all": ("BOOLEAN", {"default": False}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"mask2": ("MASK",)},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask1, mask2=None, merge_all=False, mask_inverse=False, threshold=0.5, **kwargs):
        return _combine_numbered_masks(mask1, mask2, merge_all, "and", mask_inverse, threshold, kwargs)


class MaskSubMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "merge_all": ("BOOLEAN", {"default": False}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"mask2": ("MASK",)},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask1, mask2=None, merge_all=False, mask_inverse=False, threshold=0.5, **kwargs):
        return _combine_numbered_masks(mask1, mask2, merge_all, "sub", mask_inverse, threshold, kwargs)

class MaskInvert:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "mask_inverse": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask, mask_inverse=False, threshold=0.5):
        if isinstance(mask, torch.Tensor) and mask.dim() == 3:
            original_dtype = mask.dtype
            mask = mask > threshold
            result = ~mask
            result = result.to(original_dtype)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            original_dtype = mask.dtype
            mask = mask > threshold
            result = ~mask
            result = result.unsqueeze(0).to(original_dtype)
        elif isinstance(mask, list):
            result = []
            for _mask in mask:
                original_dtype = _mask.dtype
                _mask = _mask > threshold
                _result = ~_mask
                result.append(_result)
            result = [ _result.to(original_dtype) for _result in result]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (_invert_mask_output(result, mask_inverse),)


class MaskRepeat:
    MODES = [
        "repeat_all",
        "repeat_each",
    ]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "repeat_count": ("INT", {"default": 1, "min": 1, "max": 999, "step": 1}),
                "mode": (cls.MODES, {"default": "repeat_all"}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def _repeat_tensor(self, mask, repeat_count, mode):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)

        if mask.dim() != 3:
            raise ValueError("mask tensor must have shape [H,W], [B,H,W], or [B,H,W,1]", mask.shape, type(mask))

        if mode == "repeat_all":
            return mask.repeat((repeat_count, 1, 1))
        if mode == "repeat_each":
            return mask.repeat_interleave(repeat_count, dim=0)
        raise ValueError("unsupported repeat mode", mode)

    def _repeat_list(self, mask, repeat_count, mode):
        if mode == "repeat_all":
            return [_mask.clone() for _ in range(repeat_count) for _mask in mask]
        if mode == "repeat_each":
            return [_mask.clone() for _mask in mask for _ in range(repeat_count)]
        raise ValueError("unsupported repeat mode", mode)

    def method(self, mask, repeat_count=1, mode="repeat_all", mask_inverse=False):
        repeat_count = int(repeat_count)
        if repeat_count < 1:
            raise ValueError("repeat_count must be at least 1", repeat_count)

        if isinstance(mask, torch.Tensor):
            result = self._repeat_tensor(mask, repeat_count, mode)
        elif isinstance(mask, list):
            result = self._repeat_list(mask, repeat_count, mode)
        else:
            raise ValueError("mask is not list or tensor", getattr(mask, "shape", None), type(mask))
        return (_invert_mask_output(result, mask_inverse),)


class MaskMorphology:
    METHODS = [
        "erosion",
        "dilation",
        "open",
        "close",
    ]
    KERNEL_SHAPES = ["ellipse", "rect", "cross"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "method": (cls.METHODS, {"default": "erosion"}),
                "kernel_shape": (cls.KERNEL_SHAPES, {"default": "ellipse"}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 255, "step": 2}),
                "iterations": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def _kernel(self, kernel_shape, kernel_size):
        kernel_size = max(1, int(kernel_size))
        shape_map = {
            "ellipse": cv2.MORPH_ELLIPSE,
            "rect": cv2.MORPH_RECT,
            "cross": cv2.MORPH_CROSS,
        }
        return cv2.getStructuringElement(shape_map[kernel_shape], (kernel_size, kernel_size))

    def _morphology_mask(self, mask, method, kernel_shape, kernel_size, iterations, threshold):
        source_device = mask.device
        source_dtype = mask.dtype
        kernel = self._kernel(kernel_shape, kernel_size)
        mask_np = (mask.detach().cpu().numpy() > threshold).astype(np.uint8) * 255

        if method == "erosion":
            result = cv2.erode(mask_np, kernel, iterations=iterations)
        elif method == "dilation":
            result = cv2.dilate(mask_np, kernel, iterations=iterations)
        elif method == "open":
            result = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=iterations)
        elif method == "close":
            result = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        else:
            raise ValueError("unsupported morphology method", method)

        result = (result.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return torch.tensor(result, device=source_device).to(source_dtype)

    def method(self, mask, method="erosion", kernel_shape="ellipse", kernel_size=3, iterations=1, threshold=0.5, mask_inverse=False):
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            result = self._morphology_mask(mask, method, kernel_shape, kernel_size, iterations, threshold).unsqueeze(0)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 3:
            result = torch.stack([
                self._morphology_mask(_mask, method, kernel_shape, kernel_size, iterations, threshold)
                for _mask in mask
            ])
        elif isinstance(mask, list):
            result = [
                self._morphology_mask(_mask, method, kernel_shape, kernel_size, iterations, threshold)
                for _mask in mask
            ]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (_invert_mask_output(result, mask_inverse),)


class MaskFrameBorder:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "dilate_size": ("INT", {"default": 3, "min": 0, "max": 128, "step": 1}),
                "erode_size": ("INT", {"default": 3, "min": 0, "max": 128, "step": 1}),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def _kernel(self, size):
        size = max(0, int(size))
        kernel_size = size * 2 + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    def _border_mask(self, mask, dilate_size, erode_size, threshold):
        source_device = mask.device
        source_dtype = mask.dtype
        dilate_kernel = self._kernel(dilate_size)
        erode_kernel = self._kernel(erode_size)
        mask_np = (mask.detach().cpu().numpy() > threshold).astype(np.uint8) * 255

        dilated = cv2.dilate(mask_np, dilate_kernel, iterations=1)
        eroded = cv2.erode(mask_np, erode_kernel, iterations=1)
        result = cv2.subtract(dilated, eroded)

        result = (result.astype(np.float32) / 255.0).clip(0.0, 1.0)
        return torch.tensor(result, device=source_device).to(source_dtype)

    def method(self, mask, dilate_size=3, erode_size=3, threshold=0.5, mask_inverse=False):
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            result = self._border_mask(mask, dilate_size, erode_size, threshold).unsqueeze(0)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 3:
            result = torch.stack([
                self._border_mask(_mask, dilate_size, erode_size, threshold)
                for _mask in mask
            ])
        elif isinstance(mask, list):
            result = [
                self._border_mask(_mask, dilate_size, erode_size, threshold)
                for _mask in mask
            ]
        else:
            raise ValueError("mask is not list or tensor", getattr(mask, "shape", None), type(mask))
        return (_invert_mask_output(result, mask_inverse),)


class MaskTemporalSmooth:
    METHODS = [
        "vote",
        "or",
        "and",
    ]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "method": (cls.METHODS, {"default": "vote"}),
                "window_size": ("INT", {"default": 5, "min": 1, "max": 99, "step": 2}),
                "kernel_size": ("INT", {"default": 1, "min": 1, "max": 255, "step": 2}),
                "mask_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "use_gpu": ("BOOLEAN", {"default": False}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def _window_counts(self, mask_bool, window_size, kernel_size):
        kernel = torch.ones(
            (1, 1, window_size, kernel_size, kernel_size),
            device=mask_bool.device,
            dtype=torch.float32,
        )
        padding = (window_size // 2, kernel_size // 2, kernel_size // 2)
        mask_5d = mask_bool.float().unsqueeze(0).unsqueeze(0)
        ones_count = F.conv3d(mask_5d, kernel, padding=padding).squeeze(0).squeeze(0)
        total_count = F.conv3d(torch.ones_like(mask_5d), kernel, padding=padding).squeeze(0).squeeze(0)
        return ones_count, total_count

    def _stabilize_tensor(self, mask, method, window_size, kernel_size, mask_threshold, use_gpu):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError("mask tensor must be 2D or 3D", mask.shape, type(mask))

        original_dtype = mask.dtype
        original_device = mask.device
        if use_gpu:
            if not torch.cuda.is_available():
                raise RuntimeError("use_gpu is enabled, but CUDA is not available")
            mask = mask.to("cuda")

        frame_count = mask.shape[0]
        if frame_count == 0:
            return mask.to(device=original_device, dtype=original_dtype)

        window_size = max(1, int(window_size))
        if window_size % 2 == 0:
            window_size += 1
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1

        mask_bool = mask > mask_threshold
        ones_count, total_count = self._window_counts(mask_bool, window_size, kernel_size)

        if method == "vote":
            zeros_count = total_count - ones_count
            result = ones_count >= zeros_count
        elif method == "or":
            result = ones_count > 0
        elif method == "and":
            result = ones_count >= total_count
        else:
            raise ValueError("unsupported temporal stabilize method", method)

        return result.to(device=original_device, dtype=original_dtype)

    def method(self, mask, method="vote", window_size=5, kernel_size=1, mask_threshold=0.5, use_gpu=False, mask_inverse=False):
        if isinstance(mask, torch.Tensor):
            result = self._stabilize_tensor(mask, method, window_size, kernel_size, mask_threshold, use_gpu)
        elif isinstance(mask, list):
            if len(mask) == 0:
                return (_invert_mask_output(mask, mask_inverse),)
            original_type_result = self._stabilize_tensor(
                torch.stack(mask),
                method,
                window_size,
                kernel_size,
                mask_threshold,
                use_gpu,
            )
            result = [stabilized_mask for stabilized_mask in original_type_result]
        else:
            raise ValueError("mask is not list or tensor", mask.shape, type(mask))
        return (_invert_mask_output(result, mask_inverse),)

class FillMaskArea:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "r": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "g": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "b": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, image, mask, r=0, g=0, b=0):
        print(f"FillMaskArea image.shape:{image.shape}, mask.shape:{mask.shape}, r:{r}, g:{g}, b:{b}")
        # print(f"FillMaskArea image:{image}, max:{torch.max(image)}, min:{torch.min(image)}")
        # print(f"FillMaskArea mask:{mask}, max:{torch.max(mask)}, min:{torch.min(mask)}")
        if(isinstance(image, torch.Tensor)):
            image = image.clone()
        elif(isinstance(image, list)):
            image = [img.clone() for img in image]

        if(isinstance(mask, torch.Tensor)):
            mask = mask.clone()
        elif(isinstance(mask, list)):
            mask = [m.clone() for m in mask]


        if isinstance(mask, torch.Tensor) and mask.dim() == 3:
            if(mask.shape[-1] != 1):
                mask = mask.unsqueeze(-1)
            else:
                mask = mask.unsqueeze(0)
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(-1).unsqueeze(0)
        if isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0)

        for index, _mask in enumerate(mask):
            h, w = _mask.shape[0], _mask.shape[1]
            base_color = torch.zeros((h, w, 3), dtype=torch.float32)
            base_color[:,:,0] = r / 255.0
            base_color[:,:,1] = g / 255.0
            base_color[:,:,2] = b / 255.0

            _mask = _mask.repeat(1, 1, 3)
            image[index] = image[index] * _mask  + base_color * (1 - _mask)
        return (image,)

class ImageMaskBlendBackground:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "background_images": ("IMAGE",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def _normalize_images(self, images, name):
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"{name} must be IMAGE tensor", type(images))
        images = images.clone()
        if images.dim() == 3:
            images = images.unsqueeze(0)
        if images.dim() != 4:
            raise ValueError(f"{name} must have shape [B,H,W,C] or [H,W,C]", images.shape)
        return images

    def _normalize_masks(self, masks):
        if isinstance(masks, torch.Tensor):
            masks = masks.clone()
        elif isinstance(masks, list):
            masks = torch.stack([m.clone() for m in masks])
        else:
            raise ValueError("masks must be MASK tensor or list", type(masks))

        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        if masks.dim() == 3:
            if masks.shape[-1] == 1:
                masks = masks.unsqueeze(0)
            else:
                masks = masks.unsqueeze(-1)
        if masks.dim() != 4 or masks.shape[-1] != 1:
            raise ValueError("masks must have shape [B,H,W], [H,W], or [B,H,W,1]", masks.shape)
        return masks

    def _cover_resize_crop(self, background_images, target_h, target_w):
        bg_h, bg_w = background_images.shape[1], background_images.shape[2]
        if bg_h == target_h and bg_w == target_w:
            return background_images

        scale = max(target_h / bg_h, target_w / bg_w)
        resized_h = max(target_h, int(np.ceil(bg_h * scale)))
        resized_w = max(target_w, int(np.ceil(bg_w * scale)))

        background_images = background_images.permute(0, 3, 1, 2)
        background_images = F.interpolate(
            background_images,
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )

        crop_y = max(0, (resized_h - target_h) // 2)
        crop_x = max(0, (resized_w - target_w) // 2)
        background_images = background_images[:, :, crop_y:crop_y + target_h, crop_x:crop_x + target_w]
        return background_images.permute(0, 2, 3, 1)

    def method(self, images, masks, background_images, invert_mask=False):
        images = self._normalize_images(images, "images")
        background_images = self._normalize_images(background_images, "background_images")
        masks = self._normalize_masks(masks).to(device=images.device, dtype=images.dtype)
        background_images = background_images.to(device=images.device, dtype=images.dtype)

        image_count = images.shape[0]
        mask_count = masks.shape[0]
        background_count = background_images.shape[0]

        if image_count != mask_count:
            raise ValueError("images and masks length must match", image_count, mask_count)
        if background_count not in (1, image_count):
            raise ValueError("background_images length must be 1 or match images length", background_count, image_count)
        if images.shape[1:3] != masks.shape[1:3]:
            raise ValueError("images and masks size must match", images.shape, masks.shape)
        if images.shape[-1] != background_images.shape[-1]:
            raise ValueError("images and background_images channel count must match", images.shape, background_images.shape)

        if background_count == 1 and image_count > 1:
            background_images = background_images.repeat(image_count, 1, 1, 1)
        background_images = self._cover_resize_crop(background_images, images.shape[1], images.shape[2])

        masks = masks.clamp(0.0, 1.0)
        if invert_mask:
            masks = 1.0 - masks

        result = images * (1.0 - masks) + background_images * masks
        return (result,)

class AddMask():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, image, mask):
        print(f"FillMaskArea image.shape:{image.shape}, mask.shape:{mask.shape})")

        if(isinstance(image, torch.Tensor)):
            image = image.clone()
        elif(isinstance(image, list)):
            image = [img.clone() for img in image]
        if(isinstance(mask, torch.Tensor)):
            mask = mask.clone()
        elif(isinstance(mask, list)):
            mask = [m.clone() for m in mask]

        if isinstance(mask, torch.Tensor) and mask.dim() == 3:
            if(mask.shape[-1] != 1):
                mask = mask.unsqueeze(-1)
            else:
                mask = mask.unsqueeze(0)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(-1).unsqueeze(0)

        if isinstance(image, torch.Tensor) and image.dim() == 3:
            image = image.unsqueeze(0)

        #cover bgr to bgra
        if image.shape[-1] == 3:
            image = torch.cat([image, torch.ones_like(image[...,0:1])], dim=-1)
        for index, _mask in enumerate(mask):
            image[index][:,:,3] = _mask.squeeze(-1)
        return (image,)

class MaskToRegion():
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",)
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "IMAGE")
    RETURN_NAMES = ("width", "height", "x", "y", "preview region image")
    FUNCTION = "method"
    CATEGORY = "mask"

    def method(self, mask): #找到最大輪廓並回傳其外接矩形 find the largest contour and return its bounding box
        #複製mask避免改變原始資料 copy mask to avoid changing the original data
        if(isinstance(mask, torch.Tensor)):
            mask = mask.clone()
        elif(isinstance(mask, list)):
            mask = [m.clone() for m in mask]

        #校正mask維度 correct mask dimension
        if isinstance(mask, torch.Tensor) and mask.dim() == 3:
            if(mask.shape[-1] != 1):
                mask = mask.unsqueeze(-1)
            else:
                mask = mask.unsqueeze(0)
        elif isinstance(mask, torch.Tensor) and mask.dim() == 2:
            mask = mask.unsqueeze(-1).unsqueeze(0)

        #找到最大輪廓 find the largest contour
        max_area, max_contour = 0, None
        for mask_index, _mask in enumerate(mask):
            _mask = _mask.cpu().numpy().astype(np.uint8)
            countours, _ = cv2.findContours(_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for countour in countours:
                area = cv2.contourArea(countour)
                if area > max_area:
                    max_area = area
                    max_contour = countour

        #找不到輪廓時回傳(0, 0, 0, 0) return (0, 0, 0, 0) when no contour is found
        if max_contour is None:
            return (0, 0, 0, 0)

        #找到外接矩形 find the bounding box
        x, y, w, h = cv2.boundingRect(max_contour)
        print(f"MaskToRegion x:{x}, y:{y}, w:{w}, h:{h}")

        #繪製外接矩形 draw the bounding box
        _h, _w = mask.shape[1], mask.shape[2]
        image = np.zeros((_h, _w, 3), dtype=np.uint8)
        cv2.drawContours(image, [max_contour], -1, (255, 255, 255), -1)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        image = torch.tensor(image).unsqueeze(0)

        return (w, h, x, y, image)


def _video_int(value, name, minimum=0, odd=False):
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer))
            or value < minimum or (odd and value % 2 == 0)):
        rule = "正奇數" if odd else f"大於或等於 {minimum} 的整數"
        raise ValueError(f"{name} 必須是{rule}；收到 {value!r}")
    return int(value)


def _video_threshold(value, name):
    if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} 必須是 0 到 1 之間的有限數值；收到 {value!r}")
    return value


def _normalize_video_mask(mask, name="mask"):
    """將張量或清單轉成 [T,H,W]，不改變裝置或影像尺寸。"""
    if isinstance(mask, list):
        if not mask:
            raise ValueError(f"{name} 清單不可為空")
        if isinstance(mask[0], torch.Tensor):
            # 清單元素可能各自包含多個影格；只逐項正規化，不逐影格運算。
            items = [_normalize_video_mask(item, f"{name}[{i}]") for i, item in enumerate(mask)]
            if any(item.shape[1:] != items[0].shape[1:] for item in items):
                raise ValueError(f"{name} 清單元素必須具有相同 H,W")
            if any(item.device != items[0].device for item in items):
                raise ValueError(f"{name} 清單元素必須位於相同裝置")
            return torch.cat(items, dim=0)
        try:
            mask = torch.as_tensor(mask)
        except (TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{name} 必須是規則的數值清單或遮罩張量清單") from error
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} 必須是張量或清單；收到 {type(mask).__name__}")
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 3 or any(size == 0 for size in mask.shape):
        raise ValueError(f"{name} 必須是非空的 [H,W]、[T,H,W] 或 [T,H,W,1]；收到 {tuple(mask.shape)}")
    if mask.is_complex():
        raise TypeError(f"{name} 必須包含實數遮罩值")
    return mask


def _stack_video_masks(mask1, mask2, kwargs):
    if mask1 is None:
        raise ValueError("mask1 為必填，請至少連接一個遮罩")
    # 動態輸入需要逐接頭驗證；像素運算統一在 stack 後執行。
    for name in kwargs:
        suffix = name[4:] if name.startswith("mask") else ""
        if not suffix.isascii() or not suffix.isdigit() or int(suffix) < 3 or name != f"mask{int(suffix)}":
            raise ValueError(f"不支援的參數 {name!r}；額外輸入必須命名為 mask3、mask4 等")
    named = {"mask1": mask1}
    if mask2 is not None:
        named["mask2"] = mask2
    named.update(kwargs)
    masks = [_normalize_video_mask(value, name) for name, value in named.items()]
    first = masks[0]
    for name, value in zip(named, masks):
        if value.shape != first.shape:
            raise ValueError(f"所有輸入必須具有相同 T,H,W；mask1={tuple(first.shape)}，{name}={tuple(value.shape)}")
        if value.device != first.device:
            raise ValueError(f"所有輸入必須位於相同裝置；mask1={first.device}，{name}={value.device}")
    return torch.stack(masks, dim=0)


def _video_temporal_vote(binary, radius, vote_threshold):
    radius = _video_int(radius, "radius")
    threshold = _video_threshold(vote_threshold, "vote_threshold")
    result = binary.to(torch.float32)
    if radius == 0:
        return result
    length = result.shape[0]
    radius = min(radius, length - 1)
    indices = torch.arange(length, device=result.device)
    starts = (indices - radius).clamp(min=0)
    ends = (indices + radius + 1).clamp(max=length)
    # 前綴和只沿 T 配置線性空間，邊界除以實際可用影格數。
    prefix = F.pad(result.cumsum(dim=0), (0, 0, 0, 0, 1, 0))
    averages = (prefix[ends] - prefix[starts]) / (ends - starts).view(-1, 1, 1)
    return (averages > threshold).to(torch.float32)


class TemporalMaskOr:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask1": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("INT", {"default": 0, "min": 0}),
                "vote_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mask_inverse": ("BOOLEAN", {"default": False}),
            },
            "optional": {"mask2": ("MASK",)},
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "combine"
    CATEGORY = "mask/video"
    DESCRIPTION = "逐影格二值化並取聯集，再對時間鄰域平均值取嚴格大於門檻的結果；最後可反相。"

    def _reduce(self, binary):
        return binary.any(dim=0)

    def combine(self, mask1, mask2=None, threshold=0.5, radius=0,
                vote_threshold=0.5, mask_inverse=False, **kwargs):
        threshold = _video_threshold(threshold, "threshold")
        binary = self._reduce(_stack_video_masks(mask1, mask2, kwargs) > threshold)
        result = _video_temporal_vote(binary, radius, vote_threshold)
        return (_invert_mask_output(result, mask_inverse),)


class TemporalMaskAnd(TemporalMaskOr):
    DESCRIPTION = "逐影格二值化並取交集，再對時間鄰域平均值取嚴格大於門檻的結果；最後可反相。"

    def _reduce(self, binary):
        return binary.all(dim=0)


class TemporalMaskMorphology:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mask": ("MASK",),
            "method": (["erosion", "dilation", "open", "close"], {"default": "dilation"}),
            "kernel_size": ("INT", {"default": 3, "min": 1, "step": 2}),
            "frame_kernel_size": ("INT", {"default": 1, "min": 1, "step": 2}),
            "kernel_shape": (["rect"], {"default": "rect"}),
            "iterations": ("INT", {"default": 1, "min": 1}),
            "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "mask_inverse": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "morphology"
    CATEGORY = "mask/video"
    DESCRIPTION = "單一影片遮罩的真正 3D 矩形核心形態學；時間核心 1 表示僅處理空間，邊界複製填補，最後可反相。"

    @staticmethod
    def _dilation(volume, kernel, padding):
        return F.max_pool3d(F.pad(volume, padding, mode="replicate"), kernel, stride=1)

    @classmethod
    def _erosion(cls, volume, kernel, padding):
        return -cls._dilation(-volume, kernel, padding)

    def morphology(self, mask, method="dilation", kernel_size=3, frame_kernel_size=1,
                   kernel_shape="rect", iterations=1, threshold=0.5, mask_inverse=False):
        spatial = _video_int(kernel_size, "kernel_size", minimum=1, odd=True)
        temporal = _video_int(frame_kernel_size, "frame_kernel_size", minimum=1, odd=True)
        iterations = _video_int(iterations, "iterations", minimum=1)
        threshold = _video_threshold(threshold, "threshold")
        if method not in ("erosion", "dilation", "open", "close"):
            raise ValueError(f"不支援的 method：{method!r}")
        if kernel_shape != "rect":
            raise ValueError(f"kernel_shape 僅支援 rect；收到 {kernel_shape!r}")
        volume = (_normalize_video_mask(mask) > threshold).to(torch.float32)[None, None]
        kernel = (temporal, spatial, spatial)
        padding = (spatial // 2, spatial // 2, spatial // 2, spatial // 2, temporal // 2, temporal // 2)
        # open/close 的 iterations 分別套用於各階段，與慣用形態學定義一致。
        if method in ("erosion", "open"):
            for _ in range(iterations):
                volume = self._erosion(volume, kernel, padding)
        if method in ("dilation", "open", "close"):
            for _ in range(iterations):
                volume = self._dilation(volume, kernel, padding)
        if method == "close":
            for _ in range(iterations):
                volume = self._erosion(volume, kernel, padding)
        return (_invert_mask_output(volume[0, 0], mask_inverse),)


NODE_CLASS_MAPPINGS = {
    "Temporal Mask Or (endman100)": TemporalMaskOr,
    "Temporal Mask And (endman100)": TemporalMaskAnd,
    "Temporal Mask Morphology (endman100)": TemporalMaskMorphology,
    "Mask Selection By Index (endman100)": MaskSelectionByIndex,
    "Mask Selection Max Size (endman100)": MaskSelectionMaxSize,
    "Mask Selection Min Size (endman100)": MaskSelectionMinSize,
    "Mask Selection Max Countours (endman100)": MaskSelectionMaxCountours,
    "Mask Or Mask (endman100)": MaskOrMask,
    "Mask And Mask (endman100)": MaskAndMask,
    "Mask Sub Mask (endman100)": MaskSubMask,
    "Mask Concat (endman100)": MaskConcatLongImage,
    "Mask Invert (endman100)": MaskInvert,
    "Mask Repeat (endman100)": MaskRepeat,
    "Mask Morphology (endman100)": MaskMorphology,
    "Mask Border (endman100)": MaskFrameBorder,
    "Mask Temporal Smooth (endman100)": MaskTemporalSmooth,
    "Fill Mask Area (endman100)": FillMaskArea,
    "Blend Background (endman100)": ImageMaskBlendBackground,
    "Add Mask (endman100)": AddMask,
    "Mask To Region (endman100)": MaskToRegion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Temporal Mask Or (endman100)": "Temporal Mask Or (endman100)",
    "Temporal Mask And (endman100)": "Temporal Mask And (endman100)",
    "Temporal Mask Morphology (endman100)": "Temporal Mask Morphology (endman100)",
    "Mask Selection By Index (endman100)": "Mask Selection By Index (endman100)",
    "Mask Selection Max Size (endman100)": "Mask Selection Max Size (endman100)",
    "Mask Selection Min Size (endman100)": "Mask Selection Min Size (endman100)",
    "Mask Selection Max Countours (endman100)": "Mask Selection Max Countours (endman100)",
    "Mask Or Mask (endman100)": "Mask Or Mask (endman100)",
    "Mask And Mask (endman100)": "Mask And Mask (endman100)",
    "Mask Sub Mask (endman100)": "Mask Sub Mask (endman100)",
    "Mask Concat (endman100)": "Mask Concat (endman100)",
    "Mask Invert (endman100)": "Mask Invert (endman100)",
    "Mask Repeat (endman100)": "Mask Repeat (endman100)",
    "Mask Morphology (endman100)": "Mask Morphology (endman100)",
    "Mask Border (endman100)": "Mask Border (endman100)",
    "Mask Temporal Smooth (endman100)": "Mask Temporal Smooth (endman100)",
    "Fill Mask Area (endman100)": "Fill Mask Area (endman100)",
    "Blend Background (endman100)": "Blend Background (endman100)",
    "Add Mask (endman100)": "Add Mask (endman100)",
    "Mask To Region (endman100)": "Mask To Region (endman100)"
}
