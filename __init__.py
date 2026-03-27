from .nodes.logic import NODE_CLASS_MAPPINGS as _LOGIC_CLASS, NODE_DISPLAY_NAME_MAPPINGS as _LOGIC_DISPLAY
from .nodes.image import NODE_CLASS_MAPPINGS as _IMAGE_CLASS, NODE_DISPLAY_NAME_MAPPINGS as _IMAGE_DISPLAY

NODE_CLASS_MAPPINGS = {**_LOGIC_CLASS, **_IMAGE_CLASS}
NODE_DISPLAY_NAME_MAPPINGS = {**_LOGIC_DISPLAY, **_IMAGE_DISPLAY}

import os
import yaml
cwd_path = os.path.dirname(os.path.realpath(__file__))


WEB_DIRECTORY =  os.path.join(cwd_path, f"web_version/v2")
print(f"Web directory: {WEB_DIRECTORY}")
print(os.path.exists(WEB_DIRECTORY))

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', "WEB_DIRECTORY"]
