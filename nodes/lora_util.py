from nodes import LoraLoader


class loraUtilLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_paths": ("LIST", {}),
                "model_weights": ("LIST", {}),
                "clip_weights": ("LIST", {}),
            }
        }

    CATEGORY = "LogicLite/Logic/Lora"
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"

    @staticmethod
    def _flatten(value):
        if value is None:
            return []
        queue = [value]
        out = []
        while queue:
            item = queue.pop(0)
            if isinstance(item, (list, tuple)):
                queue.extend(item)
            else:
                out.append(item)
        return out

    @classmethod
    def _normalize_paths(cls, value):
        tokens = []
        for item in cls._flatten(value):
            tokens.extend(str(item).replace("\r", "\n").replace(",", "\n").split("\n"))
        return [token.strip() for token in tokens if token and token.strip()]

    @classmethod
    def _normalize_weights(cls, value):
        tokens = []
        for item in cls._flatten(value):
            tokens.extend(str(item).replace("\r", "\n").replace(",", "\n").split("\n"))

        weights = []
        for token in tokens:
            token = str(token).strip()
            if not token:
                continue
            try:
                weights.append(float(token))
            except ValueError:
                weights.append(1.0)
        return weights

    @staticmethod
    def _weight_at(weights, index, default):
        if index < len(weights):
            return weights[index]
        if weights:
            return weights[-1]
        return default

    def load_loras(self, model, clip, lora_paths, model_weights, clip_weights):
        paths = self._normalize_paths(lora_paths)
        model_strengths = self._normalize_weights(model_weights)
        clip_strengths = self._normalize_weights(clip_weights)
        loader = LoraLoader()

        for index, lora_name in enumerate(paths):
            if not lora_name or lora_name.lower() == "none":
                continue

            strength_model = self._weight_at(model_strengths, index, 1.0)
            strength_clip = self._weight_at(clip_strengths, index, strength_model)
            if strength_model == 0 and strength_clip == 0:
                continue

            model, clip = loader.load_lora(model, clip, lora_name, strength_model, strength_clip)

        return (model, clip)

    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        return True


NODE_CLASS_MAPPINGS = {
    "logic loraUtilLoader": loraUtilLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "logic loraUtilLoader": "Power Lora Loader (List API)",
}
