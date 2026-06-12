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


NODE_CLASS_MAPPINGS = {
    "logic loadImageBatch": loadImageBatch,
    "logic imageListToImages": imageListToImages,
    "logic resizeImage": resizeImage,
    "logic resizeMask": resizeMask,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "logic loadImageBatch": "Load Image Batch",
    "logic imageListToImages": "Image List to Images",
    "logic resizeImage": "Resize Image",
    "logic resizeMask": "Resize Mask",
}
