import os
import numpy as np
import torch
import hashlib
import uuid
from PIL import Image
import folder_paths
import comfy.utils


class loadImageBatch:
    def __init__(self):
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
        import node_helpers
        from PIL import ImageOps, ImageSequence
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


NODE_CLASS_MAPPINGS = {
    "logic loadImageBatch": loadImageBatch,
    "logic imageListToImages": imageListToImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "logic loadImageBatch": "Load Image Batch",
    "logic imageListToImages": "Image List to Images",
}
