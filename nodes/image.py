class loadImageBatch:
    def __init__(self):
        import folder_paths

        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_paths": ("LIST", {}),
            }
        }

    INPUT_IS_LIST = True
    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image_list", "mask_list", "count")
    OUTPUT_IS_LIST = (True, True, False)
    OUTPUT_NODE = True
    FUNCTION = "load_images"

    @staticmethod
    def _normalize_paths(image_paths):
        if image_paths is None:
            return []

        # ComfyUI list mode may pass a scalar string, a list of strings,
        # or nested list wrappers from upstream list outputs.
        queue = [image_paths]
        tokens = []
        while queue:
            item = queue.pop(0)
            if isinstance(item, (list, tuple)):
                queue.extend(item)
            else:
                tokens.extend(str(item).replace("\r", "\n").replace(",", "\n").split("\n"))

        names = []
        for p in tokens:
            v = str(p).strip()
            if v:
                names.append(v)
        return names

    def _load_single(self, image_name):
        import os
        import numpy as np
        import torch
        import folder_paths
        import node_helpers
        from PIL import Image, ImageOps, ImageSequence

        # 若是絕對路徑則直接使用，否則透過 ComfyUI folder_paths 解析
        if os.path.isabs(image_name):
            image_path = image_name
        else:
            image_path = folder_paths.get_annotated_filepath(image_name)
        img = node_helpers.pillow(Image.open, image_path)
        output_images = []
        output_masks = []
        w, h = None, None
        for frame in ImageSequence.Iterator(img):
            frame = node_helpers.pillow(ImageOps.exif_transpose, frame)
            if frame.mode == 'I':
                frame = frame.point(lambda i: i * (1 / 255))
            rgb = frame.convert("RGB")
            if w is None:
                w, h = rgb.size
            if rgb.size[0] != w or rgb.size[1] != h:
                continue
            image_array = np.array(rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_array)[None,]
            if 'A' in frame.getbands():
                mask = np.array(frame.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif frame.mode == 'P' and 'transparency' in frame.info:
                mask = np.array(frame.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image_tensor)
            output_masks.append(mask.unsqueeze(0))
            if img.format == "MPO":
                break
        if len(output_images) > 1:
            return torch.cat(output_images, dim=0), torch.cat(output_masks, dim=0)
        return output_images[0], output_masks[0]

    def load_images(self, **kwargs):
        import os
        import uuid
        import numpy as np
        from PIL import Image

        all_images = []
        all_masks = []
        preview_results = []

        names = self._normalize_paths(kwargs.get("image_paths", None))

        for name in names:
            try:
                img, mask = self._load_single(name)
            except Exception as e:
                print(f"[loadImageBatch] Skipping ({name}): {e}")
                continue
            all_images.append(img)
            all_masks.append(mask)
            # 儲存每張圖至暫存資料夾供 UI 預覽
            for frame_idx in range(img.shape[0]):
                pil_img = Image.fromarray(
                    np.clip(255. * img[frame_idx].cpu().numpy(), 0, 255).astype(np.uint8)
                )
                filename = f"loadbatch_{uuid.uuid4().hex[:12]}.png"
                out_path = os.path.join(self.output_dir, filename)
                pil_img.save(out_path, compress_level=self.compress_level)
                preview_results.append({"filename": filename, "subfolder": "", "type": self.type})
        if not all_images:
            return {"result": ([], [], 0)}
        return {"result": (all_images, all_masks, len(all_images))}

    @classmethod
    def IS_CHANGED(s, **kwargs):
        import os
        import hashlib
        import folder_paths

        m = hashlib.sha256()
        for name in s._normalize_paths(kwargs.get("image_paths", None)):
            try:
                path = name if os.path.isabs(name) else folder_paths.get_annotated_filepath(name)
                with open(path, 'rb') as f:
                    m.update(f.read())
            except Exception:
                # Ignore unreadable paths here; execution path will skip them too.
                m.update(name.encode("utf-8", errors="ignore"))
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True


class imageListToImages:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_list": ("IMAGE",),
            },
            "optional": {
                "mask_list": ("MASK",),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    INPUT_IS_LIST = True
    FUNCTION = "to_batch"

    def to_batch(self, image_list, mask_list=None):
        import torch
        import comfy.utils

        # 以第一張為基準尺寸，自動 resize 其他張
        target_h, target_w = image_list[0].shape[1], image_list[0].shape[2]
        aligned_images = [image_list[0]]
        for img in image_list[1:]:
            if img.shape[1] != target_h or img.shape[2] != target_w:
                img = comfy.utils.common_upscale(
                    img.movedim(-1, 1), target_w, target_h, "bilinear", "center"
                ).movedim(1, -1)
            aligned_images.append(img)
        out_image = torch.cat(aligned_images, dim=0)

        if mask_list:
            aligned_masks = [mask_list[0]]
            for msk in mask_list[1:]:
                if msk.shape[-2] != target_h or msk.shape[-1] != target_w:
                    msk = comfy.utils.common_upscale(
                        msk.unsqueeze(1).float(), target_w, target_h, "bilinear", "center"
                    ).squeeze(1)
                aligned_masks.append(msk)
            out_mask = torch.cat(aligned_masks, dim=0)
        else:
            out_mask = torch.zeros((out_image.shape[0], target_h, target_w), dtype=torch.float32)

        return (out_image, out_mask)


class resizeImage:
    RESIZE_METHODS = ["nearest", "linear", "cubic", "area", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        from nodes import MAX_RESOLUTION

        return {
            "required": {
                "image": ("IMAGE",),
                "resize_mode": (["width_height", "total_pixels"], {"default": "width_height"}),
                "total_megapixels": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 9999.0, "step": 0.01}),
                "divisiblke_by": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "method": (s.RESIZE_METHODS, {"default": "linear"}),
                "resize_batch": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 33, "min": 0, "max": 9999, "step": 1}),
            },
            "optional": {
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "height", "width", "divisiblke_by")
    FUNCTION = "resize"

    def _interpolation(self, method):
        import cv2

        resize_methods = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        return resize_methods[method]

    def _resize_one(self, image, height, width, method, resize_batch):
        import cv2
        import numpy as np
        import torch

        interpolation = self._interpolation(method)
        source = image.detach()
        device = source.device
        dtype = source.dtype
        np_source = source.cpu().numpy()

        if np_source.ndim == 3:
            if np_source.shape[-1] not in (1, 3, 4):
                resized = []
                for item in np_source:
                    resized_item = cv2.resize(
                        np.ascontiguousarray(item), (width, height), interpolation=interpolation
                    )
                    resized.append(resized_item)
                return torch.from_numpy(np.stack(resized, axis=0)).to(device=device, dtype=dtype)

            resized = cv2.resize(
                np.ascontiguousarray(np_source), (width, height), interpolation=interpolation
            )
            return torch.from_numpy(resized).to(device=device, dtype=dtype)

        if np_source.ndim == 4:
            resized = []
            for item in np_source:
                resized_item = cv2.resize(
                    np.ascontiguousarray(item), (width, height), interpolation=interpolation
                )
                resized.append(resized_item)
            return torch.from_numpy(np.stack(resized, axis=0)).to(device=device, dtype=dtype)

        if np_source.ndim == 2:
            resized = cv2.resize(
                np.ascontiguousarray(np_source), (width, height), interpolation=interpolation
            )
            return torch.from_numpy(resized).to(device=device, dtype=dtype)

        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    def _resize_frames(self, image, batch_size, method):
        import cv2
        import numpy as np
        import torch

        if batch_size <= 0:
            return image

        source = image.detach()
        device = source.device
        dtype = source.dtype
        np_source = source.cpu().numpy()
        source_count = np_source.shape[0]

        if source_count == batch_size:
            return image
        if source_count == 0:
            raise ValueError("Cannot resize batch size from an empty image batch")
        if source_count == 1:
            np_result = np.repeat(np_source, batch_size, axis=0)
            return torch.from_numpy(np_result).to(device=device, dtype=dtype)

        positions = np.linspace(0, source_count - 1, batch_size)
        resized = []
        for pos in positions:
            if method == "nearest":
                resized.append(np_source[int(round(pos))])
                continue

            low = int(np.floor(pos))
            high = int(np.ceil(pos))
            if low == high:
                resized.append(np_source[low])
                continue

            alpha = float(pos - low)
            resized.append(cv2.addWeighted(np_source[low], 1.0 - alpha, np_source[high], alpha, 0.0))

        return torch.from_numpy(np.stack(resized, axis=0)).to(device=device, dtype=dtype)

    def _images_to_batch(self, images):
        import torch

        if not images:
            raise ValueError("Cannot resize batch size from an empty image list")
        if not all(isinstance(item, torch.Tensor) for item in images):
            raise ValueError("Image list items must be tensors")

        first = images[0]
        if first.ndim == 4 and first.shape[0] == 1:
            return torch.cat(images, dim=0)
        if first.ndim == 3:
            return torch.stack(images, dim=0)
        raise ValueError(f"Unsupported image list item shape: {tuple(first.shape)}")

    def _nearest_multiple(self, value, divisiblke_by):
        lower = (value // divisiblke_by) * divisiblke_by
        upper = lower + divisiblke_by
        if lower <= 0:
            return upper
        if value - lower <= upper - value:
            return lower
        return upper

    def _source_size(self, image, resize_batch):
        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("Cannot get size from an empty image list")
            return self._source_size(image[0], resize_batch)

        if image.ndim == 4:
            return int(image.shape[1]), int(image.shape[2])
        if image.ndim == 3:
            if image.shape[-1] not in (1, 3, 4):
                return int(image.shape[1]), int(image.shape[2])
            return int(image.shape[0]), int(image.shape[1])
        if image.ndim == 2:
            return int(image.shape[0]), int(image.shape[1])
        raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

    def _size_from_total_pixels(self, source_height, source_width, total_megapixels):
        total_pixels = max(1, int(round(float(total_megapixels) * 1000000)))
        aspect = source_width / source_height
        width = max(1, int(round((total_pixels * aspect) ** 0.5)))
        height = max(1, int(round(width / aspect)))
        return height, width

    def resize(
        self,
        image,
        resize_mode="width_height",
        total_megapixels=1.0,
        method="linear",
        resize_batch=False,
        batch_size=33,
        divisiblke_by=0,
        height=0,
        width=0,
    ):
        if isinstance(height, (list, tuple)):
            height = height[0]
        if isinstance(width, (list, tuple)):
            width = width[0]
        if isinstance(divisiblke_by, (list, tuple)):
            divisiblke_by = divisiblke_by[0]
        if isinstance(resize_batch, (list, tuple)):
            resize_batch = resize_batch[0]
        if isinstance(batch_size, (list, tuple)):
            batch_size = batch_size[0]
        if isinstance(total_megapixels, (list, tuple)):
            total_megapixels = total_megapixels[0]
        if isinstance(resize_mode, (list, tuple)):
            resize_mode = resize_mode[0]

        height = int(height)
        width = int(width)
        divisiblke_by = int(divisiblke_by)
        batch_size = int(batch_size)
        source_height, source_width = self._source_size(image, resize_batch)
        if resize_mode == "total_pixels":
            height, width = self._size_from_total_pixels(source_height, source_width, total_megapixels)
        else:
            if height <= 0:
                height = source_height
            if width <= 0:
                width = source_width
        if divisiblke_by > 0:
            height = self._nearest_multiple(height, divisiblke_by)
            width = self._nearest_multiple(width, divisiblke_by)

        if isinstance(image, (list, tuple)):
            resized = [self._resize_one(item, height, width, method, resize_batch) for item in image]
            if resize_batch and batch_size > 0:
                resized = self._resize_frames(self._images_to_batch(resized), batch_size, method)
        else:
            resized = self._resize_one(image, height, width, method, resize_batch)
            if resize_batch and resized.ndim == 4:
                resized = self._resize_frames(resized, batch_size, method)
        return (resized, height, width, divisiblke_by)


class resizeMask:
    RESIZE_METHODS = ["nearest", "linear", "cubic", "area", "lanczos"]

    @classmethod
    def INPUT_TYPES(s):
        from nodes import MAX_RESOLUTION

        return {
            "required": {
                "mask": ("MASK",),
                "resize_mode": (["width_height", "total_pixels"], {"default": "width_height"}),
                "total_megapixels": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 9999.0, "step": 0.01}),
                "divisiblke_by": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "method": (s.RESIZE_METHODS, {"default": "nearest"}),
                "resize_batch": ("BOOLEAN", {"default": False}),
                "batch_size": ("INT", {"default": 33, "min": 0, "max": 9999, "step": 1}),
            },
            "optional": {
                "height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("MASK", "INT", "INT", "INT")
    RETURN_NAMES = ("mask", "height", "width", "divisiblke_by")
    FUNCTION = "resize"

    def _interpolation(self, method):
        import cv2

        resize_methods = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        return resize_methods[method]

    def _resize_one(self, mask, height, width, method, resize_batch):
        import cv2
        import numpy as np
        import torch

        interpolation = self._interpolation(method)
        source = mask.detach()
        device = source.device
        dtype = source.dtype
        np_source = source.cpu().numpy()

        if np_source.ndim == 2:
            resized = cv2.resize(
                np.ascontiguousarray(np_source), (width, height), interpolation=interpolation
            )
            return torch.from_numpy(resized).to(device=device, dtype=dtype)

        if np_source.ndim == 3:
            resized = []
            for item in np_source:
                resized_item = cv2.resize(
                    np.ascontiguousarray(item), (width, height), interpolation=interpolation
                )
                resized.append(resized_item)
            return torch.from_numpy(np.stack(resized, axis=0)).to(device=device, dtype=dtype)

        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    def _resize_frames(self, mask, batch_size, method):
        import cv2
        import numpy as np
        import torch

        if batch_size <= 0:
            return mask

        source = mask.detach()
        device = source.device
        dtype = source.dtype
        np_source = source.cpu().numpy()
        source_count = np_source.shape[0]

        if source_count == batch_size:
            return mask
        if source_count == 0:
            raise ValueError("Cannot resize batch size from an empty mask batch")
        if source_count == 1:
            np_result = np.repeat(np_source, batch_size, axis=0)
            return torch.from_numpy(np_result).to(device=device, dtype=dtype)

        positions = np.linspace(0, source_count - 1, batch_size)
        resized = []
        for pos in positions:
            if method == "nearest":
                resized.append(np_source[int(round(pos))])
                continue

            low = int(np.floor(pos))
            high = int(np.ceil(pos))
            if low == high:
                resized.append(np_source[low])
                continue

            alpha = float(pos - low)
            resized.append(cv2.addWeighted(np_source[low], 1.0 - alpha, np_source[high], alpha, 0.0))

        return torch.from_numpy(np.stack(resized, axis=0)).to(device=device, dtype=dtype)

    def _masks_to_batch(self, masks):
        import torch

        if not masks:
            raise ValueError("Cannot resize batch size from an empty mask list")
        if not all(isinstance(item, torch.Tensor) for item in masks):
            raise ValueError("Mask list items must be tensors")

        first = masks[0]
        if first.ndim == 3 and first.shape[0] == 1:
            return torch.cat(masks, dim=0)
        if first.ndim == 2:
            return torch.stack(masks, dim=0)
        raise ValueError(f"Unsupported mask list item shape: {tuple(first.shape)}")

    def _nearest_multiple(self, value, divisiblke_by):
        lower = (value // divisiblke_by) * divisiblke_by
        upper = lower + divisiblke_by
        if lower <= 0:
            return upper
        if value - lower <= upper - value:
            return lower
        return upper

    def _source_size(self, mask, resize_batch):
        if isinstance(mask, (list, tuple)):
            if not mask:
                raise ValueError("Cannot get size from an empty mask list")
            return self._source_size(mask[0], resize_batch)

        if mask.ndim == 3:
            return int(mask.shape[1]), int(mask.shape[2])
        if mask.ndim == 2:
            return int(mask.shape[0]), int(mask.shape[1])
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    def _size_from_total_pixels(self, source_height, source_width, total_megapixels):
        total_pixels = max(1, int(round(float(total_megapixels) * 1000000)))
        aspect = source_width / source_height
        width = max(1, int(round((total_pixels * aspect) ** 0.5)))
        height = max(1, int(round(width / aspect)))
        return height, width

    def resize(
        self,
        mask,
        resize_mode="width_height",
        total_megapixels=1.0,
        method="nearest",
        resize_batch=False,
        batch_size=33,
        divisiblke_by=0,
        height=0,
        width=0,
    ):
        if isinstance(height, (list, tuple)):
            height = height[0]
        if isinstance(width, (list, tuple)):
            width = width[0]
        if isinstance(divisiblke_by, (list, tuple)):
            divisiblke_by = divisiblke_by[0]
        if isinstance(resize_batch, (list, tuple)):
            resize_batch = resize_batch[0]
        if isinstance(batch_size, (list, tuple)):
            batch_size = batch_size[0]
        if isinstance(total_megapixels, (list, tuple)):
            total_megapixels = total_megapixels[0]
        if isinstance(resize_mode, (list, tuple)):
            resize_mode = resize_mode[0]

        height = int(height)
        width = int(width)
        divisiblke_by = int(divisiblke_by)
        batch_size = int(batch_size)
        source_height, source_width = self._source_size(mask, resize_batch)
        if resize_mode == "total_pixels":
            height, width = self._size_from_total_pixels(source_height, source_width, total_megapixels)
        else:
            if height <= 0:
                height = source_height
            if width <= 0:
                width = source_width
        if divisiblke_by > 0:
            height = self._nearest_multiple(height, divisiblke_by)
            width = self._nearest_multiple(width, divisiblke_by)

        if isinstance(mask, (list, tuple)):
            resized = [self._resize_one(item, height, width, method, resize_batch) for item in mask]
            if resize_batch and batch_size > 0:
                resized = self._resize_frames(self._masks_to_batch(resized), batch_size, method)
        else:
            resized = self._resize_one(mask, height, width, method, resize_batch)
            if resize_batch and resized.ndim == 3:
                resized = self._resize_frames(resized, batch_size, method)
        return (resized, height, width, divisiblke_by)


class gaussianBlurMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "kernel_size": ("INT", {"default": 15, "min": 1, "max": 255, "step": 2}),
                "sigma": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "blur"

    def _normalize_kernel_size(self, kernel_size):
        kernel_size = max(1, int(kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    def _blur_tensor(self, mask, kernel_size, sigma):
        import torch
        import torch.nn.functional as F

        source = mask.detach()
        device = source.device
        dtype = source.dtype

        if kernel_size <= 1:
            return source.clone().to(device=device, dtype=dtype)

        if sigma <= 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

        if source.ndim == 2:
            batch = source.unsqueeze(0)
            squeeze_batch = True
        elif source.ndim == 3:
            batch = source
            squeeze_batch = False
        else:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

        x = batch.to(device=device, dtype=torch.float32).unsqueeze(1)
        coords = torch.arange(kernel_size, device=device, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()

        padding = kernel_size // 2
        x = F.pad(x, (padding, padding, 0, 0), mode="replicate")
        x = F.conv2d(x, kernel_1d.view(1, 1, 1, kernel_size))
        x = F.pad(x, (0, 0, padding, padding), mode="replicate")
        x = F.conv2d(x, kernel_1d.view(1, 1, kernel_size, 1))

        result = x.squeeze(1).clamp(0.0, 1.0)
        if squeeze_batch:
            result = result.squeeze(0)
        return result.to(device=device, dtype=dtype)

    def blur(self, mask, kernel_size=15, sigma=3.0):
        if isinstance(kernel_size, (list, tuple)):
            kernel_size = kernel_size[0]
        if isinstance(sigma, (list, tuple)):
            sigma = sigma[0]

        kernel_size = self._normalize_kernel_size(kernel_size)
        sigma = float(sigma)

        if isinstance(mask, (list, tuple)):
            return ([self._blur_tensor(item, kernel_size, sigma) for item in mask],)
        return (self._blur_tensor(mask, kernel_size, sigma),)


class imageColorToMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "red": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "green": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "blue": ("INT", {"default": 255, "min": 0, "max": 255, "step": 1}),
                "threshold": ("FLOAT", {"default": 30.0, "min": 0.0, "max": 441.6729, "step": 0.1}),
                "invert": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "color_to_mask"

    def _first_value(self, value):
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def _color_to_mask_tensor(self, image, red, green, blue, threshold, invert):
        import torch

        source = image.detach()
        device = source.device

        if source.ndim == 3:
            source = source.unsqueeze(0)
            squeeze_batch = True
        elif source.ndim == 4:
            squeeze_batch = False
        else:
            raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

        if source.shape[-1] < 3:
            rgb = source[..., :1].repeat_interleave(3, dim=-1)
        else:
            rgb = source[..., :3]

        target = torch.tensor(
            [float(red), float(green), float(blue)],
            dtype=torch.float32,
            device=device,
        ).view(1, 1, 1, 3) / 255.0
        threshold = float(threshold) / 255.0

        distance = torch.linalg.vector_norm(rgb.to(dtype=torch.float32) - target, dim=-1)
        mask = (distance <= threshold).to(dtype=torch.float32)
        if invert:
            mask = 1.0 - mask
        if squeeze_batch:
            mask = mask.squeeze(0)
        return mask

    def color_to_mask(self, image, red=255, green=255, blue=255, threshold=30.0, invert=False):
        red = int(self._first_value(red))
        green = int(self._first_value(green))
        blue = int(self._first_value(blue))
        threshold = float(self._first_value(threshold))
        invert = bool(self._first_value(invert))

        if isinstance(image, (list, tuple)):
            return ([self._color_to_mask_tensor(item, red, green, blue, threshold, invert) for item in image],)
        return (self._color_to_mask_tensor(image, red, green, blue, threshold, invert),)


class dominantImageColor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("red", "green", "blue")
    FUNCTION = "dominant_color"

    def _images_to_batch(self, images):
        import torch

        if not images:
            raise ValueError("Cannot get dominant color from an empty image list")
        if not all(isinstance(item, torch.Tensor) for item in images):
            raise ValueError("Image list items must be tensors")

        first = images[0]
        if first.ndim == 4 and first.shape[0] == 1:
            return torch.cat(images, dim=0)
        if first.ndim == 3:
            return torch.stack(images, dim=0)
        if first.ndim == 4:
            return torch.cat(images, dim=0)
        raise ValueError(f"Unsupported image list item shape: {tuple(first.shape)}")

    def _masks_to_batch(self, masks):
        import torch

        if not masks:
            return None
        if not all(isinstance(item, torch.Tensor) for item in masks):
            raise ValueError("Mask list items must be tensors")

        normalized = []
        for item in masks:
            if item.ndim == 2:
                normalized.append(item.unsqueeze(0))
            elif item.ndim == 3:
                normalized.append(item)
            elif item.ndim == 4 and item.shape[-1] == 1:
                normalized.append(item.squeeze(-1))
            else:
                raise ValueError(f"Unsupported mask list item shape: {tuple(item.shape)}")
        return torch.cat(normalized, dim=0)

    def _normalize_image(self, image):
        if isinstance(image, (list, tuple)):
            image = self._images_to_batch(image)

        if image.ndim == 3:
            image = image.unsqueeze(0)
        elif image.ndim != 4:
            raise ValueError(f"Unsupported image shape: {tuple(image.shape)}")

        if image.shape[-1] < 3:
            return image[..., :1].repeat_interleave(3, dim=-1)
        return image[..., :3]

    def _normalize_mask(self, mask, batch_size, height, width, device):
        import torch
        import torch.nn.functional as F

        if mask is None:
            return None
        if isinstance(mask, (list, tuple)):
            mask = self._masks_to_batch(mask)
            if mask is None:
                return None

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.ndim != 3:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

        mask = mask.to(device=device, dtype=torch.float32)
        if mask.shape[-2] != height or mask.shape[-1] != width:
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(height, width),
                mode="nearest",
            ).squeeze(1)

        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.repeat(batch_size, 1, 1)
        elif mask.shape[0] != batch_size:
            raise ValueError(
                f"Mask batch size {mask.shape[0]} does not match image batch size {batch_size}"
            )

        return mask > 0.5

    def dominant_color(self, image, mask=None):
        import torch

        rgb = self._normalize_image(image).detach()
        device = rgb.device
        batch_size, height, width = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
        mask_bool = self._normalize_mask(mask, batch_size, height, width, device)

        rgb_255 = (rgb.to(dtype=torch.float32).clamp(0.0, 1.0) * 255.0).round().to(dtype=torch.int64)
        pixels = rgb_255.reshape(-1, 3)
        if mask_bool is not None:
            selected = mask_bool.reshape(-1)
            if bool(selected.any()):
                pixels = pixels[selected]

        if pixels.numel() == 0:
            return (0, 0, 0)

        packed = (pixels[:, 0] << 16) + (pixels[:, 1] << 8) + pixels[:, 2]
        values, counts = torch.unique(packed, return_counts=True)
        color = int(values[torch.argmax(counts)].item())
        red = (color >> 16) & 255
        green = (color >> 8) & 255
        blue = color & 255
        return (red, green, blue)


class maskFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        from nodes import MAX_RESOLUTION

        return {
            "required": {
                "mask": ("MASK",),
                "batch_index": ("INT", {"default": 0, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION}),
                "length": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "frombatch"
    SEARCH_ALIASES = ["select mask", "pick from batch", "extract mask"]

    def _first_value(self, value):
        if isinstance(value, (list, tuple)):
            return value[0]
        return value

    def _from_tensor_batch(self, mask, batch_index, length):
        source = mask.detach()
        if source.ndim == 2:
            source = source.unsqueeze(0)
        elif source.ndim != 3:
            raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

        if batch_index < 0:
            batch_index += source.shape[0]
        batch_index = max(0, min(source.shape[0] - 1, batch_index))
        length = min(source.shape[0] - batch_index, length)
        return source[batch_index:batch_index + length].clone()

    def frombatch(self, mask, batch_index=0, length=1):
        import torch

        batch_index = int(self._first_value(batch_index))
        length = int(self._first_value(length))

        if isinstance(mask, (list, tuple)):
            if not mask:
                raise ValueError("Cannot get mask from an empty batch")
            if batch_index < 0:
                batch_index += len(mask)
            batch_index = max(0, min(len(mask) - 1, batch_index))
            length = min(len(mask) - batch_index, length)
            selected = []
            for item in mask[batch_index:batch_index + length]:
                if item.ndim == 2:
                    selected.append(item.unsqueeze(0))
                elif item.ndim == 3:
                    selected.append(item)
                else:
                    raise ValueError(f"Unsupported mask shape: {tuple(item.shape)}")
            return (torch.cat(selected, dim=0),)
        return (self._from_tensor_batch(mask, batch_index, length),)


class emptyMask:
    @classmethod
    def INPUT_TYPES(s):
        from nodes import MAX_RESOLUTION

        return {
            "required": {
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "solid"

    def solid(self, value, width, height, batch_size=1):
        import torch
        import comfy.model_management

        mask = torch.full(
            (batch_size, height, width), value,
            dtype=torch.float32, device=comfy.model_management.intermediate_device(),
        )
        return (mask,)


class repeatImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "repeat"

    def repeat(self, image, amount=1):
        return (image.repeat((amount, 1, 1, 1)),)


class repeatMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    CATEGORY = "LogicLite/Logic/Image"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "repeat"

    def repeat(self, mask, amount=1):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return (mask.repeat((amount, 1, 1)),)


NODE_CLASS_MAPPINGS = {
    "logic repeatImage": repeatImage,
    "logic repeatMask": repeatMask,
    "logic emptyMask": emptyMask,
    "logic loadImageBatch": loadImageBatch,
    "logic imageListToImages": imageListToImages,
    "logic resizeImage": resizeImage,
    "logic resizeMask": resizeMask,
    "logic gaussianBlurMask": gaussianBlurMask,
    "logic imageColorToMask": imageColorToMask,
    "logic dominantImageColor": dominantImageColor,
    "logic maskFromBatch": maskFromBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "logic repeatImage": "Repeat Image",
    "logic repeatMask": "Repeat Mask",
    "logic emptyMask": "Empty Mask",
    "logic loadImageBatch": "Load Image Batch",
    "logic imageListToImages": "Image List to Images",
    "logic resizeImage": "Resize Image",
    "logic resizeMask": "Resize Mask",
    "logic gaussianBlurMask": "Gaussian Blur Mask",
    "logic imageColorToMask": "Image Color to Mask",
    "logic dominantImageColor": "Dominant Image Color",
    "logic maskFromBatch": "Get Mask from Batch",
}
