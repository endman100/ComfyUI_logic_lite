"""獨立實作的高精度影片／遮罩 I/O，不依賴 VideoHelperSuite。"""

import asyncio
from collections.abc import Mapping
from contextlib import ExitStack
from fractions import Fraction
import glob
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from uuid import uuid4
from zipfile import ZipFile

from aiohttp import web
from aiohttp.helpers import content_disposition_header
from comfy.utils import ProgressBar
import numpy as np
import torch
import torch.nn.functional as F
import folder_paths


FORMATS = {
    "[mask] ProRes 4444 (.mov)": ("prores", "mov", "12"),
    "[mask] ProRes 4444 XQ (.mov)": ("prores_xq", "mov", "12"),
    "[mask] VP9 alpha (.webm)": ("vp9", "webm", "8"),
    "[mask] PNG16 RGBA sequence": ("png", "png", "16"),
    "[mask] EXR float sequence": ("exr", "exr", "float"),
    "[no mask] H.264 (.mp4)": ("h264", "mp4", "8"),
}
LOAD_ID = "Load Video With Mask (Upload) (endman100)"
SAVE_ID = "Video Combine With Mask (endman100)"
_BUFFER_BYTES = 16 * 1024 * 1024
_MEDIA_ROUTE = "/logiclite/video-mask"
_VIDEO_EXTENSIONS = frozenset(".avi .flv .gif .m4v .mkv .mov .mp4 .mpeg .mpg .mts .m2ts .ts .webm .wmv".split())
_SEQUENCE_EXTENSIONS = frozenset({".png", ".exr"})
_PREVIEW_INPUT_FLAGS = ["-protocol_whitelist", "file", "-format_whitelist",
                        "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpeg,mpegts,flv,asf,gif"]
_load_preview_slots = asyncio.Semaphore(2)
_media = {}


def _media_file(path):
    roots = (folder_paths.get_input_directory(), folder_paths.get_output_directory(), folder_paths.get_temp_directory())
    return (path.is_file() and path.resolve() == path
            and any(path.is_relative_to(Path(root).resolve()) for root in roots))


def _media_ui(preview, raw, archive_root=None, source=None):
    token = uuid4().hex
    if not all(_media_file(path) for path in raw):
        raw = ()
    _media[token] = (preview, tuple(raw), archive_root)
    url = f"{_MEDIA_ROUTE}/{token}"
    item = {"preview": f"{url}/preview", "raw": f"{url}/raw" if raw else None}
    if source is not None:
        item["source"] = source
    return {"mask_video_previews": [item]}


def _zip_sequence(files, root):
    archive = tempfile.TemporaryFile()
    try:
        with ZipFile(archive, "w") as output:
            for path in files:
                if not _media_file(path):
                    raise web.HTTPNotFound()
                output.write(path, path.relative_to(root).as_posix())
        archive.seek(0)
        return archive
    except BaseException:
        archive.close()
        raise


async def _serve_media(request):
    media = _media.get(request.match_info["token"])
    if media is None:
        raise web.HTTPNotFound()
    preview, raw, archive_root = media
    if request.match_info["kind"] == "preview":
        if not _media_file(preview):
            raise web.HTTPNotFound()
        return web.FileResponse(preview, headers={"Content-Type": "video/webm", "X-Content-Type-Options": "nosniff"})
    if not raw:
        raise web.HTTPNotFound()
    filename = f"{archive_root.name or 'sequence'}.zip" if archive_root is not None else raw[0].name
    headers = {"Content-Disposition": content_disposition_header("attachment", filename=filename),
               "X-Content-Type-Options": "nosniff"}
    if archive_root is None:
        if not _media_file(raw[0]):
            raise web.HTTPNotFound()
        return web.FileResponse(raw[0], headers=headers)
    with await asyncio.to_thread(_zip_sequence, raw, archive_root) as archive:
        response = web.StreamResponse(headers={**headers, "Content-Type": "application/zip"})
        response.content_length = os.fstat(archive.fileno()).st_size
        await response.prepare(request)
        if request.method != "HEAD":
            while chunk := archive.read(1024 * 1024):
                await response.write(chunk)
        return response


def _register_media_route(routes):
    routes.post(_MEDIA_ROUTE + "/load-preview")(_serve_load_preview)
    routes.get(_MEDIA_ROUTE + r"/{token:[0-9a-f]{32}}/{kind:preview|raw}")(_serve_media)


def find_ffmpeg():
    """依明確覆寫、imageio、VHS 已有設定、PATH 的順序尋找，不匯入 VHS。"""
    override = os.environ.get("LOGIC_LITE_FFMPEG")
    if override:
        found = shutil.which(override)
        if not found:
            raise RuntimeError("LOGIC_LITE_FFMPEG 指向不存在或不可執行的 ffmpeg")
        return found
    candidates = []
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        candidates.append(get_ffmpeg_exe())
    except (ImportError, RuntimeError, OSError):
        pass
    candidates.append(os.environ.get("VHS_FORCE_FFMPEG_PATH"))
    candidates.extend(getattr(module, "ffmpeg_path", None) for name, module in tuple(sys.modules.items())
                      if name.endswith("videohelpersuite.utils"))
    candidates.append(shutil.which("ffmpeg"))
    for candidate in candidates:
        found = shutil.which(candidate) if candidate else None
        if found:
            return found
    raise RuntimeError("找不到 ffmpeg；請安裝 CPU ffmpeg，或設定 LOGIC_LITE_FFMPEG")


def find_ffprobe(ffmpeg):
    override = os.environ.get("LOGIC_LITE_FFPROBE")
    if override:
        found = shutil.which(override)
        if not found:
            raise RuntimeError("LOGIC_LITE_FFPROBE 指向不存在或不可執行的 ffprobe")
        return found
    companion = Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt" else "ffprobe")
    found = shutil.which(str(companion)) or shutil.which("ffprobe")
    if not found:
        raise RuntimeError("找不到 ffprobe；請安裝 ffmpeg 工具組，或設定 LOGIC_LITE_FFPROBE")
    return found


def _probe(path, ffmpeg, input_flags=()):
    result = subprocess.run([find_ffprobe(ffmpeg), "-v", "error",
                             "-show_streams", "-show_format", "-of", "json", *input_flags, str(path)], capture_output=True)
    if result.returncode:
        raise RuntimeError("無法讀取媒體資訊：" + result.stderr.decode("utf-8", errors="replace"))
    metadata = json.loads(result.stdout)
    streams = metadata.get("streams", [])
    stream = next((item for item in streams if item.get("codec_type") == "video"), None)
    if stream is None:
        raise ValueError(f"檔案沒有影像軌：{path}")
    stream["audio"] = next((item for item in streams if item.get("codec_type") == "audio"), None)
    pixel_format = stream.get("pix_fmt", "")
    tags = {key.lower(): str(value) for key, value in stream.get("tags", {}).items()}
    stream["has_alpha"] = (any(part in pixel_format for part in ("rgba", "bgra", "argb", "abgr", "yuva", "gbrap"))
                           or pixel_format.startswith("ya") or tags.get("alpha_mode") == "1")
    try:
        stream["fps"] = float(Fraction(stream.get("avg_frame_rate", "0/1")))
    except (ValueError, ZeroDivisionError):
        stream["fps"] = 0.0
    stream["duration_seconds"] = 0.0
    for value in (stream.get("duration"), metadata.get("format", {}).get("duration")):
        try:
            duration = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(duration) and duration > 0:
            stream["duration_seconds"] = duration
            break
    return stream


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} 必須是大於或等於 {minimum} 的整數")
    return value


def _rate(value, name, allow_zero=False):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} 必須是{'非負' if allow_zero else '正'}有限數值")
    return float(value)


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def _input_media():
    root = Path(folder_paths.get_input_directory())
    choices = []
    for directory, _, names in os.walk(root):
        relative = Path(directory).relative_to(root)
        sequence = [name for name in names if Path(name).suffix.lower() in _SEQUENCE_EXTENSIONS]
        choices.extend((relative / name).as_posix() for name in names
                       if Path(name).suffix.lower() in _VIDEO_EXTENSIONS)
        if relative.parts and sequence and len({Path(name).suffix.lower() for name in sequence}) == 1:
            choices.append(relative.as_posix())
        else:
            choices.extend((relative / name).as_posix() for name in sequence)
    return sorted(choices, key=_natural_key)


def _sequence_signature(path):
    """只讀取影格標頭，檢查整批序列的尺寸、精度與通道一致。"""
    with path.open("rb") as stream:
        if path.suffix.lower() == ".png":
            header = stream.read(29)
            if len(header) != 29 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
                raise ValueError(f"無效的 PNG 標頭：{path}")
            return struct.unpack(">IIBB", header[16:26])
        if stream.read(4) != struct.pack("<I", 20000630):
            raise ValueError(f"無效的 EXR 標頭：{path}")
        version = stream.read(4)
        if len(version) != 4 or struct.unpack("<I", version)[0] & 0x1A00:
            raise ValueError("僅支援一般單層 scanline EXR，不支援 tiled／deep／multipart EXR")

        def string():
            value = bytearray()
            while True:
                byte = stream.read(1)
                if not byte or len(value) > 255:
                    raise ValueError(f"無效的 EXR 屬性：{path}")
                if byte == b"\0":
                    return bytes(value)
                value.extend(byte)

        attributes = {}
        while True:
            name = string()
            if not name:
                break
            string()
            length = stream.read(4)
            if len(length) != 4:
                raise ValueError(f"不完整的 EXR 標頭：{path}")
            size = struct.unpack("<I", length)[0]
            if size > 16 * 1024 * 1024:
                raise ValueError(f"EXR 屬性過大：{path}")
            value = stream.read(size)
            if len(value) != size:
                raise ValueError(f"不完整的 EXR 屬性：{path}")
            if name in (b"dataWindow", b"channels"):
                attributes[name] = value
        if b"dataWindow" not in attributes or b"channels" not in attributes:
            raise ValueError(f"EXR 缺少 dataWindow 或 channels：{path}")
        return attributes[b"dataWindow"], attributes[b"channels"]


def _resolve_source(video, input_only=False):
    if not isinstance(video, str) or not video.strip():
        raise ValueError("請先選擇或上傳影片／PNG／EXR 序列，或填入來源路徑")
    text = video.strip()
    if text.endswith((" [input]", " [output]", " [temp]")):
        text = folder_paths.get_annotated_filepath(text)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = Path(folder_paths.get_input_directory()) / path
    path = path.resolve()
    root = Path(folder_paths.get_input_directory()).resolve() if input_only else None
    if root is not None and not path.is_relative_to(root):
        raise PermissionError("即時預覽限 ComfyUI input 內的檔案；其他來源請執行工作流預覽")
    if path.is_dir():
        files = [item for item in path.iterdir() if item.suffix.lower() in (".png", ".exr") and item.is_file()]
    elif any(char in path.name for char in ("*", "?", "#", "%")):
        pattern = re.sub(r"%0?\d*d|#+", "*", str(path))
        files = [Path(item) for item in glob.glob(pattern) if Path(item).is_file()]
    elif path.suffix.lower() in _SEQUENCE_EXTENSIONS and path.is_file():
        match = re.fullmatch(r"(.*?)(\d+)", path.stem)
        if match:
            prefix, number = match.groups()
            files = [item for item in path.parent.iterdir()
                     if item.is_file() and item.suffix.lower() == path.suffix.lower()
                     and re.fullmatch(re.escape(prefix) + r"\d+", item.stem)
                     and int(re.search(r"\d+$", item.stem).group()) >= int(number)]
        else:
            files = [path]
    elif path.is_file():
        return path, None
    else:
        raise ValueError(f"找不到 video：{path}")
    files = sorted(files, key=_natural_key)
    if not files or any(item.suffix.lower() not in (".png", ".exr") for item in files):
        raise ValueError("序列必須包含 PNG 或 EXR 影格")
    if len({item.suffix.lower() for item in files}) != 1:
        raise ValueError("同一序列不可混用 PNG 與 EXR；請指定檔名 pattern")
    if root is not None and not all(item.resolve().is_relative_to(root) for item in files):
        raise PermissionError("即時預覽的序列影格必須位於 ComfyUI input 內")
    return files[0], files


def _load_sequence(first, files, sequence_fps, force_rate, skip, nth, cap, missing_alpha, preview_only=False):
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    try:
        import cv2
    except (ImportError, OSError) as error:
        raise RuntimeError("PNG／EXR 序列需要 OpenCV（opencv-python）；不會降精度或改用 ffmpeg 解碼") from error
    signature = _sequence_signature(first)
    if not preview_only:
        for path in files:
            if _sequence_signature(path) != signature:
                raise ValueError(f"序列影格必須具有相同尺寸、位元深度與通道：{path}")

    # 以影格中心的時間做最近鄰重採樣，再套用跳格、抽格與 cap；不插值原始像素。
    rate = force_rate or sequence_fps
    count = math.floor(len(files) * rate / sequence_fps + 0.5)
    stop = min(count, skip + cap * nth) if cap else count
    positions = np.arange(skip, stop, nth, dtype=np.int64)
    indices = np.minimum(np.floor((positions + 0.5) * sequence_fps / rate).astype(np.int64), len(files) - 1)
    if not len(indices):
        raise ValueError("選取後沒有影格；請檢查 skip_first_frames 與 frame_load_cap")
    unique, inverse = np.unique(indices, return_inverse=True)
    # The first frame also supplies the source metadata when the selection skips it.
    progress = None if preview_only else ProgressBar(len(unique) + int(unique[0] != 0) + 1)

    def read(path):
        if preview_only and _sequence_signature(path) != signature:
            raise ValueError(f"序列影格必須具有相同尺寸、位元深度與通道：{path}")
        try:
            frame = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except cv2.error as error:
            raise RuntimeError(f"OpenCV 無法讀取序列影格；EXR 需要啟用 OpenEXR 的版本：{path}") from error
        if frame is None:
            hint = "；請確認 OpenCV 編譯時包含 OpenEXR 支援" if path.suffix.lower() == ".exr" else ""
            raise ValueError(f"OpenCV 無法讀取序列影格{hint}：{path}")
        if frame.ndim not in (2, 3) or (frame.ndim == 3 and frame.shape[-1] not in (3, 4)):
            raise ValueError(f"序列僅支援灰階、RGB 或 RGBA：{path}")
        if frame.dtype not in (np.uint8, np.uint16, np.float32):
            raise ValueError(f"不支援的序列像素型別 {frame.dtype}：{path}")
        if progress is not None:
            progress.update(1)
        if preview_only and max(frame.shape[:2]) > 480:
            height, width = frame.shape[:2]
            scale = 480 / max(height, width)
            frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        return frame

    first_frame = read(first)
    has_alpha = first_frame.ndim == 3 and first_frame.shape[-1] == 4
    if not has_alpha and missing_alpha == "error":
        raise ValueError("來源沒有 alpha；若需要全白遮罩，請將 missing_alpha 設為 ones")
    # 每個檔案僅 imread 一次；整批 stack、通道重排與數值轉換均向量化。
    frames = [first_frame if index == 0 else read(files[index]) for index in unique]
    if any(frame.shape != first_frame.shape or frame.dtype != first_frame.dtype for frame in frames):
        raise ValueError("序列影格必須具有相同尺寸、位元深度與通道")
    array = np.stack(frames)[inverse].astype(np.float32)
    if first_frame.dtype in (np.uint8, np.uint16):
        array /= np.iinfo(first_frame.dtype).max
    if first_frame.ndim == 2:
        rgb = np.repeat(array[..., None], 3, axis=-1)
    else:
        rgb = array[..., [2, 1, 0]]
    alpha = array[..., 3:] if has_alpha else np.ones((*array.shape[:3], 1), dtype=np.float32)
    rgba = torch.from_numpy(np.concatenate((rgb, alpha), axis=-1))
    info = {"width": first_frame.shape[1], "height": first_frame.shape[0], "has_alpha": has_alpha,
            "pix_fmt": f"opencv_{'BGRA' if has_alpha else ('gray' if first_frame.ndim == 2 else 'BGR')}_{first_frame.dtype}"}
    return rgba, info, progress


def _process_error(log, operation):
    log.seek(0, os.SEEK_END)
    log.seek(max(0, log.tell() - 6000))
    return RuntimeError(f"ffmpeg {operation}失敗：" + log.read().decode("utf-8", errors="replace"))


def _decode(args, width, height, floating, ffmpeg, progress, has_alpha=True, grayscale=False):
    channels = (1 if grayscale else (4 if has_alpha else 3)) if floating else 4
    dtype = np.dtype("<f4" if floating else "<u2")
    frame_bytes = width * height * channels * dtype.itemsize
    chunk = memoryview(bytearray(frame_bytes * max(1, _BUFFER_BYTES // frame_bytes)))
    pixel_format = ("grayf32le" if grayscale else ("gbrapf32le" if has_alpha else "gbrpf32le")) if floating else "rgba64le"
    command = [ffmpeg, "-v", "error", "-nostdin", "-hwaccel", "none", "-noautorotate", "-threads", "2",
               *args, "-map", "0:v:0", "-an", "-sn", "-dn", "-pix_fmt", pixel_format,
               "-threads", "2", "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1"]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=log)
        try:
            while True:
                filled = 0
                while filled < len(chunk):
                    end = filled + frame_bytes
                    read = process.stdout.readinto(chunk[filled:end])
                    if not read:
                        break
                    if read != frame_bytes:
                        raise RuntimeError("ffmpeg 傳回不完整影格，請確認序列尺寸與格式一致")
                    filled = end
                    if progress is not None:
                        loaded = progress.current + 1
                        progress.update_absolute(loaded, max(progress.total, loaded + 1))
                if not filled:
                    break
                raw = chunk[:filled]
                if floating:
                    planes = np.frombuffer(raw, dtype=dtype).reshape(-1, channels, height, width)
                    order = [0, 0, 0] if grayscale else ([2, 0, 1, 3] if has_alpha else [2, 0, 1])
                    array = planes[:, order].transpose(0, 2, 3, 1).copy()
                    if not has_alpha:
                        array = np.concatenate((array, np.ones((*array.shape[:-1], 1), dtype=np.float32)), axis=-1)
                else:
                    array = np.frombuffer(raw, dtype=dtype).reshape(-1, height, width, 4).astype(np.float32)
                    array /= 65535.0
                yield torch.from_numpy(array)
            if process.wait():
                raise _process_error(log, "解碼")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
            process.wait()


def _video_load_estimate(info, force_rate, skip, nth, cap):
    reported = str(info.get("nb_frames", ""))
    frames = int(reported) if reported.isdecimal() else 0
    if force_rate or not frames:
        duration = info["duration_seconds"] or frames / info["fps"]
        frames = round(duration * (force_rate or info["fps"]))
    selected = max(0, (frames - skip + nth - 1) // nth)
    return min(cap, selected) if cap and selected else selected or cap


def _preview_selection(rate, nth, cap):
    sample, _ = _preview_sampling(rate / nth)
    count = len(range(0, sample.stop, sample.step))
    return nth * sample.step, min(count, (cap + sample.step - 1) // sample.step) if cap else count


def _load_audio(path, info, start, duration, ffmpeg):
    sample_rate, channels = int(info["sample_rate"]), int(info["channels"])
    if sample_rate <= 0 or channels <= 0:
        raise ValueError("音軌的採樣率或通道數無效")
    command = [ffmpeg, "-v", "error", "-nostdin", "-ss", str(start), "-i", str(path),
               "-t", str(duration), "-map", "0:a:0", "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode:
        raise RuntimeError("ffmpeg 音軌解碼失敗：" + result.stderr.decode("utf-8", errors="replace"))
    if not result.stdout:
        return None
    pcm = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, channels).T.copy()
    return {"waveform": torch.from_numpy(pcm).unsqueeze(0), "sample_rate": sample_rate}


class LoadVideoWithMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": (["", *_input_media()], {"tooltip": "從 ComfyUI input 選擇影片、序列目錄或首格"}),
            "frame_load_cap": ("INT", {"default": 0, "min": 0}),
            "skip_first_frames": ("INT", {"default": 0, "min": 0}),
            "select_every_nth": ("INT", {"default": 1, "min": 1}),
            "force_rate": ("FLOAT", {"default": 0.0, "min": 0.0}),
            "custom_width": ("INT", {"default": 0, "min": 0}),
            "custom_height": ("INT", {"default": 0, "min": 0}),
            "sequence_fps": ("FLOAT", {"default": 24.0, "min": 0.01}),
            "missing_alpha": (["ones", "error"], {"default": "ones"}),
        }, "optional": {
            "source_path": ("STRING", {"default": "", "tooltip": "選填，覆寫下拉選項。支援絕對路徑、序列首格或 %08d／####／* pattern"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "FLOAT", "STRING", "AUDIO")
    RETURN_NAMES = ("images", "mask", "frame_count", "fps", "video_info", "audio")
    FUNCTION = "load_video"
    CATEGORY = "video/mask"
    DESCRIPTION = "CPU 高精度影片載入；PNG16／EXR 序列使用 OpenCV IMREAD_UNCHANGED。MASK 直接等於 alpha（白色不透明），不反相；無 alpha 可選全白或報錯。"

    @classmethod
    def VALIDATE_INPUTS(cls, video, source_path=""):
        try:
            _resolve_source(source_path.strip() or video)
        except (ValueError, OSError, RuntimeError) as error:
            return str(error)
        return True

    @classmethod
    def IS_CHANGED(cls, video, source_path=""):
        try:
            first, files = _resolve_source(source_path.strip() or video)
            stats = [(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                     for path in files or (first,) for stat in (path.stat(),)]
        except (ValueError, OSError, RuntimeError):
            return float("nan")
        return sha256(json.dumps(stats).encode()).hexdigest()

    def load_video(self, video, frame_load_cap=0, skip_first_frames=0, select_every_nth=1,
                   force_rate=0.0, custom_width=0, custom_height=0, sequence_fps=24.0, missing_alpha="ones",
                   source_path="", _preview_only=False):
        cap = _integer(frame_load_cap, "frame_load_cap", 0)
        skip = _integer(skip_first_frames, "skip_first_frames", 0)
        nth = _integer(select_every_nth, "select_every_nth", 1)
        width = _integer(custom_width, "custom_width", 0)
        height = _integer(custom_height, "custom_height", 0)
        force_rate = _rate(force_rate, "force_rate", allow_zero=True)
        sequence_fps = _rate(sequence_fps, "sequence_fps")
        if missing_alpha not in ("ones", "error"):
            raise ValueError("missing_alpha 必須是 ones 或 error")
        first, files = _resolve_source(source_path.strip() or video, input_only=_preview_only)
        unmodified = not (cap or skip or nth != 1 or force_rate or width or height)
        reuse_preview = False
        if files is not None:
            if _preview_only:
                nth, cap = _preview_selection(force_rate or sequence_fps, nth, cap)
            rgba, info, progress = _load_sequence(first, files, sequence_fps, force_rate, skip, nth, cap,
                                                  missing_alpha, preview_only=_preview_only)
        else:
            ffmpeg = find_ffmpeg()
            input_flags = _PREVIEW_INPUT_FLAGS if _preview_only else []
            info = _probe(first, ffmpeg, input_flags)
            reuse_preview = (unmodified and first.suffix.lower() == ".webm" and info.get("codec_name") == "vp9"
                             and info["has_alpha"] and _media_file(first))
        if not info["has_alpha"] and missing_alpha == "error":
            raise ValueError("來源沒有 alpha；若需要全白遮罩，請將 missing_alpha 設為 ones")
        source_fps = sequence_fps if files is not None else info["fps"]
        if source_fps <= 0:
            raise ValueError("無法取得來源 fps；請先轉為有有效影格率的影片或影像序列")
        source_w, source_h = info["width"], info["height"]
        rotation = next((item["rotation"] for item in info.get("side_data_list", []) if "rotation" in item), 0)
        if not math.isfinite(rotation) or abs(rotation / 90 - round(rotation / 90)) > 1e-6:
            raise ValueError("目前僅支援 90 度倍數的影片旋轉資訊")
        turns = round(rotation / 90) % 4
        display_w, display_h = (source_h, source_w) if turns % 2 else (source_w, source_h)
        source_frames = len(files) if files is not None else _video_load_estimate(info, 0, 0, 1, 0)
        source = None
        if files is None:
            source = {"width": display_w, "height": display_h, "fps": source_fps,
                      "frames": source_frames, "has_audio": info["audio"] is not None}
        if _preview_only and reuse_preview:
            return _media_ui(first, (first,), source=source)
        width = width or (max(1, round(display_w * height / display_h)) if height else display_w)
        height = height or max(1, round(display_h * width / display_w))
        rate = force_rate or source_fps
        if _preview_only:
            if files is None:
                nth, cap = _preview_selection(rate, nth, cap)
            scale = min(1, 480 / max(width, height))
            width, height = max(1, round(width * scale)), max(1, round(height * scale))
        fps = rate / nth
        floating = first.suffix.lower() == ".exr"
        filters = ([f"fps={force_rate}"] if force_rate else [])
        filters += [f"trim=start_frame={skip}", f"select=not(mod(n\\,{nth}))", f"setpts=N/({fps}*TB)"]
        batches = []
        if files is not None:
            if (width, height) != (display_w, display_h):
                rgba = F.interpolate(rgba.movedim(-1, 1), size=(height, width), mode="bilinear",
                                     align_corners=False).movedim(1, -1)
            batches.append(rgba)
        else:
            progress = None if _preview_only else ProgressBar(_video_load_estimate(info, force_rate, skip, nth, cap) + 1)
            inputs = ["-i", str(first)]
            # VP9 原生解碼器可能忽略 WebM alpha；明確使用 libvpx 解碼。
            if info.get("codec_name") == "vp9" and info["has_alpha"]:
                inputs = ["-c:v", "libvpx-vp9", *inputs]
            args = [*input_flags, *inputs, "-vf", ",".join(filters)]
            if cap:
                args += ["-frames:v", str(cap)]
            for batch in _decode(args, source_w, source_h, floating, ffmpeg, progress,
                                 has_alpha=info["has_alpha"], grayscale=info.get("pix_fmt", "").startswith("gray")):
                if turns:
                    batch = torch.rot90(batch, turns, dims=(1, 2))
                if (width, height) != (display_w, display_h):
                    batch = F.interpolate(batch.movedim(-1, 1), size=(height, width), mode="bilinear",
                                          align_corners=False).movedim(1, -1)
                batches.append(batch)
        if not batches:
            raise ValueError("選取後沒有影格；請檢查 skip_first_frames 與 frame_load_cap")
        rgba = torch.cat(batches, dim=0)
        alpha = rgba[..., 3].contiguous() if info["has_alpha"] else torch.ones_like(rgba[..., 0])
        archive_root = Path(os.path.commonpath([path.parent for path in files])) if files else None
        preview = first
        if not reuse_preview:
            preview = _output_directory("mask_video_preview", False) / "preview.webm"
            _write_preview(preview, rgba, alpha, fps, find_ffmpeg() if files is not None else ffmpeg,
                           audio_path=first if source and source["has_audio"] else None,
                           audio_start=skip / rate, input_flags=input_flags if files is None else ())
        ui = _media_ui(preview, files or (first,), archive_root, source=source)
        if _preview_only:
            return ui
        audio = (_load_audio(first, info["audio"], skip / rate, len(rgba) / fps, ffmpeg)
                 if files is None and info["audio"] else None)
        metadata = {"source": str(first), "source_fps": source_fps, "fps": fps,
                    "source_width": display_w, "source_height": display_h, "source_frame_count": source_frames,
                    "source_duration": source_frames / source_fps if files is not None else info["duration_seconds"],
                    "loaded_duration": len(rgba) / fps,
                    "frame_count": len(rgba), "width": width, "height": height,
                    "has_alpha": info["has_alpha"], "source_pix_fmt": info.get("pix_fmt"),
                    "source_rotation": rotation,
                    "decoder": "opencv_imread_unchanged" if files is not None else "ffmpeg",
                    "alpha_convention": "mask=alpha", "sequence_files": len(files) if files else None,
                    "has_audio": audio is not None}
        result = {"ui": ui, "result": (rgba[..., :3].contiguous(), alpha, len(rgba), fps,
                                         json.dumps(metadata, ensure_ascii=False), audio)}
        completed = progress.current + 1
        progress.update_absolute(completed, completed)
        return result


async def _serve_load_preview(request):
    fields = {"video", "source_path", "frame_load_cap", "skip_first_frames", "select_every_nth",
              "force_rate", "custom_width", "custom_height", "sequence_fps", "missing_alpha"}
    try:
        settings = await request.json()
        if (not isinstance(settings, dict) or not isinstance(settings.get("video"), str)
                or not isinstance(settings.get("source_path", ""), str) or settings.keys() - fields):
            raise ValueError("無效的影片預覽設定")
        async with _load_preview_slots:
            ui = await asyncio.to_thread(LoadVideoWithMask().load_video, **settings, _preview_only=True)
    except PermissionError as error:
        return web.json_response({"error": str(error)}, status=403)
    except (TypeError, ValueError, OSError, RuntimeError) as error:
        return web.json_response({"error": str(error)}, status=400)
    return web.json_response(ui)


def _validate_images(images, mask):
    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[-1] not in (3, 4) or min(images.shape) < 1:
        raise ValueError("images 必須是非空的 [T,H,W,3] 或 [T,H,W,4]")
    if images.is_complex():
        raise ValueError("images 必須是實數")
    if mask is not None:
        if not isinstance(mask, torch.Tensor) or mask.is_complex():
            raise ValueError("mask 必須是實數張量")
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.shape == images.shape[:3]:
            return mask
        if mask.ndim == 3 and mask.shape[0] == 1 and mask.shape[1:] == images.shape[1:3]:
            return mask.expand(images.shape[:3])
        raise ValueError("mask 必須與 images 有相同 H,W，且格數為 1 或與 images 相同；"
                         f"images={tuple(images.shape[:3])}，mask={tuple(mask.shape)}")
    return mask


def _validate_audio(audio):
    if not isinstance(audio, Mapping):
        raise ValueError("audio 必須為 ComfyUI AUDIO")
    waveform = audio.get("waveform")
    if (not isinstance(waveform, torch.Tensor) or waveform.ndim != 3 or waveform.shape[0] != 1
            or min(waveform.shape) < 1 or not waveform.is_floating_point()):
        raise ValueError("audio waveform 必須為非空的 [1, channels, samples] 浮點張量")
    return waveform, _integer(audio.get("sample_rate"), "audio sample_rate", 1)


def _rgba_chunks(images, mask, repetitions, half=False, mask_only=False, preview_size=None, order=None):
    height, width = images.shape[1:3]
    count = len(images) if order is None else len(order)
    batch_size = max(1, _BUFFER_BYTES // (height * width * 4 * 4))
    # 只依有界 I/O 批次與播放次數迭代；轉換在整個批次上向量化，不逐像素處理。
    for _ in range(repetitions):
        for start in range(0, count, batch_size):
            chunk = slice(start, start + batch_size)
            indices = chunk if order is None else order[chunk]
            rgb = images[indices, ..., :3].detach().to(device="cpu", dtype=torch.float32)
            if mask is not None:
                alpha = mask[indices].detach().to(device="cpu", dtype=torch.float32).unsqueeze(-1)
            elif images.shape[-1] == 4:
                alpha = images[indices, ..., 3:].detach().to(device="cpu", dtype=torch.float32)
            else:
                alpha = torch.ones_like(rgb[..., :1])
            if preview_size is None and (not torch.isfinite(rgb).all() or not torch.isfinite(alpha).all()):
                raise ValueError("images 與 mask 不可包含 NaN 或 Infinity")
            alpha = alpha.clamp(0, 1)
            rgba = torch.cat((alpha.expand_as(rgb) if mask_only else rgb, alpha), dim=-1)
            if preview_size is not None:
                rgba.nan_to_num_().clamp_(0, 1)
                if preview_size != (height, width):
                    rgba = F.interpolate(rgba.movedim(-1, 1), size=preview_size, mode="bilinear",
                                         align_corners=False).movedim(1, -1)
            rgba = rgba.numpy()
            if half and np.any(np.abs(rgba) > np.finfo(np.float16).max):
                raise ValueError("EXR half 數值超過 float16 範圍；請改用 bit_depth=float")
            yield rgba.astype(np.float16).astype(np.float32) if half else rgba


def _encode(destination, kind, depth, images, mask, frame_rate, repetitions, ffmpeg,
            mask_only=False, preview_size=None, progress=None, audio_args=(), audio=None, order=None, crf=18):
    height, width = preview_size or images.shape[1:3]
    floating = kind == "exr"
    raw_format = "rgba" if kind == "preview" else ("gbrapf32le" if floating else "rgba64le")
    command = [ffmpeg, "-v", "error", "-nostdin", "-n", "-f", "rawvideo", "-pixel_format", raw_format,
               "-video_size", f"{width}x{height}", "-framerate", str(frame_rate), "-i", "pipe:0"]
    output = []
    if kind in ("prores", "prores_xq"):
        output += ["-c:v", "prores_ks", "-profile:v", "4444xq" if kind == "prores_xq" else "4444",
                   "-pix_fmt", "yuva444p12le", "-alpha_bits", "16"]
    elif kind in ("vp9", "preview"):
        output += ["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0"]
        if kind == "preview":
            output += ["-deadline", "realtime", "-cpu-used", "6", "-b:v", "300k", "-crf", "38"]
        else:
            output += ["-lossless", "1"]
    elif kind == "png":
        output += ["-c:v", "png", "-pix_fmt", "rgba64be", "-start_number", "0"]
    elif kind == "exr":
        output += ["-c:v", "exr", "-pix_fmt", "gbrapf32le", "-format", "half" if depth == "half" else "float",
                   "-compression", "zip16", "-start_number", "0"]
    elif kind == "h264":
        output += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart"]
    with tempfile.TemporaryFile() as log, ExitStack() as files:
        if audio is not None:
            waveform, sample_rate = audio
            frame_count = (len(images) if order is None else len(order)) * repetitions
            duration = frame_count / frame_rate
            samples = max(1, math.ceil(duration * sample_rate))
            audio_path = Path(files.enter_context(tempfile.TemporaryDirectory())) / "waveform.f32"
            pcm = waveform[0, :, :samples].detach().to(device="cpu", dtype=torch.float32).transpose(0, 1).numpy()
            np.asarray(pcm, dtype="<f4").tofile(audio_path)
            codec = {"vp9": "libopus", "prores": "pcm_s16le", "prores_xq": "pcm_s16le", "h264": "aac"}[kind]
            audio_args = ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(waveform.shape[1]), "-i", str(audio_path),
                          "-map", "0:v:0", "-map", "1:a:0", "-af", f"apad,atrim=duration={duration}", "-c:a", codec]
        command += [*(audio_args or ("-an",)), *output, "-threads", "2", str(destination)]
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            try:
                for rgba in _rgba_chunks(images, mask, repetitions, half=depth == "half", mask_only=mask_only,
                                         preview_size=preview_size, order=order):
                    if floating:
                        data = np.ascontiguousarray(rgba[..., [1, 2, 0, 3]].transpose(0, 3, 1, 2), dtype="<f4")
                    elif kind == "preview":
                        data = np.rint(rgba * 255).astype(np.uint8)
                    else:
                        data = np.rint(np.clip(rgba, 0, 1) * 65535).astype("<u2")
                    process.stdin.write(data.tobytes())
                    if progress is not None:
                        progress.update(len(data))
                process.stdin.close()
            except BrokenPipeError:
                process.wait()
                raise _process_error(log, "編碼") from None
            if process.wait():
                raise _process_error(log, "編碼")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass


def _preview_sampling(frame_rate):
    stride = max(1, math.ceil(frame_rate / 15))
    fps = frame_rate / stride
    stop = max(1, math.floor(8 * fps)) * stride
    return slice(0, stop, stride), fps


def _write_preview(destination, images, mask, frame_rate, ffmpeg, progress=None,
                   *, audio_path=None, audio_start=0, input_flags=(), order=None):
    sample, fps = _preview_sampling(frame_rate)
    scale = min(1, 480 / max(images.shape[1:3]))
    size = tuple(max(2, 2 * math.floor(dimension * scale / 2)) for dimension in images.shape[1:3])
    selection = sample if order is None else order[sample]
    frames = images[selection]
    audio_args = ()
    if audio_path is not None:
        audio_args = [*input_flags, "-ss", str(audio_start), "-i", str(audio_path),
                      "-map", "0:v:0", "-map", "1:a:0", "-af", f"atrim=duration={len(frames) / fps}",
                      "-c:a", "libopus", "-b:a", "64k"]
    _encode(destination, "preview", "8", frames, None if mask is None else mask[selection],
            fps, 1, ffmpeg, preview_size=size, progress=progress, audio_args=audio_args)


def _output_directory(prefix, save_output):
    if not isinstance(prefix, str) or not prefix.strip():
        raise ValueError("filename_prefix 不可為空")
    root = Path(folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()).resolve()
    relative = Path(prefix)
    if relative.is_absolute() or ".." in relative.parts or any(char in prefix for char in ("%", "\\", ":")):
        raise ValueError("filename_prefix 必須是 output 內的相對路徑，不可包含 ..、%、反斜線或磁碟代號")
    parent = (root / relative.parent).resolve()
    if not parent.is_relative_to(root):
        raise ValueError("filename_prefix 不可超出 ComfyUI 輸出目錄")
    parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=relative.name + "_", dir=parent))


class VideoCombineWithMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "frame_rate": ("FLOAT", {"default": 24.0, "min": 0.01}),
            "loop_count": ("INT", {"default": 0, "min": 0}),
            "filename_prefix": ("STRING", {"default": "mask_video"}),
            "format": (list(FORMATS), {"default": "[mask] ProRes 4444 (.mov)"}),
            "save_output": ("BOOLEAN", {"default": True}),
            "mask_mode": (["error", "separate_png", "ignore"], {"default": "error", "tooltip": "不支援 alpha 的格式：報錯／另存 PNG16 遮罩序列／忽略遮罩"}),
            "save_mask_sequence": ("BOOLEAN", {"default": False, "tooltip": "不論主格式是否支援 alpha，額外保存可重載 MASK 的 PNG16 序列"}),
            "bit_depth": (["auto", "8", "12", "16", "half", "float"], {"default": "auto", "tooltip": "auto：ProRes 要求 12-bit（實際精度受編碼器限制）、VP9/H.264 8、PNG 16、EXR float；EXR 另可選 half"}),
            "pingpong": ("BOOLEAN", {"default": False, "tooltip": "正向播放後反向播放，首尾格不重複"}),
        }, "optional": {"mask": ("MASK",), "audio": ("AUDIO",),
                        "crf": ("INT", {"default": 18, "min": 0, "max": 51, "tooltip": "僅 H.264。數值越低，畫質越高"})}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filenames",)
    FUNCTION = "combine_video"
    OUTPUT_NODE = True
    CATEGORY = "video/mask"
    DESCRIPTION = "CPU 串流輸出影片／高精度序列。MASK 直接作 alpha；可另存 PNG16 遮罩序列。ProRes 以 RGBA16 輸入並要求 yuva444p12le，實際精度受編碼器限制；EXR 保留 float 數值。"

    def combine_video(self, images, frame_rate=24.0, loop_count=0, filename_prefix="mask_video",
                      format="[mask] ProRes 4444 (.mov)", save_output=True, mask_mode="error",
                      save_mask_sequence=False, bit_depth="auto", mask=None, audio=None, pingpong=False, crf=18):
        mask = _validate_images(images, mask)
        frame_rate = _rate(frame_rate, "frame_rate")
        repetitions = _integer(loop_count, "loop_count", 0) + 1
        if format not in FORMATS:
            raise ValueError(f"不支援的 format：{format}")
        if mask_mode not in ("error", "separate_png", "ignore"):
            raise ValueError("mask_mode 必須是 error、separate_png 或 ignore")
        kind, extension, default_depth = FORMATS[format]
        sequence = kind in ("png", "exr")
        audio = _validate_audio(audio) if audio is not None and not sequence else None
        if kind == "h264" and _integer(crf, "crf", 0) > 51:
            raise ValueError("H.264 crf 必須介於 0 和 51")
        depth = default_depth if bit_depth == "auto" else bit_depth
        if depth not in (("half", "float") if kind == "exr" else (default_depth,)):
            raise ValueError(f"{format} 不支援 bit_depth={bit_depth}；有效選項為 auto 或 "
                             + ("half／float" if kind == "exr" else default_depth))
        has_mask = mask is not None or images.shape[-1] == 4
        if kind == "h264" and has_mask and mask_mode == "error":
            raise ValueError("H.264 不支援 alpha；mask_mode 請選 separate_png 或 ignore")
        if kind in ("h264", "vp9") and any(size % 2 for size in images.shape[1:3]):
            raise ValueError("H.264／VP9 的 4:2:0 格式需要偶數寬高；不會自動裁切或補邊")
        ffmpeg = find_ffmpeg()
        extra_mask = has_mask and (save_mask_sequence or (kind == "h264" and mask_mode == "separate_png"))
        order = None
        if pingpong and len(images) > 2:
            order = torch.cat((torch.arange(len(images)), torch.arange(len(images) - 2, 0, -1)))
        cycle = len(images) if order is None else len(order)
        frame_count = cycle * repetitions
        preview_frames = len(range(cycle)[_preview_sampling(frame_rate)[0]]) if kind != "vp9" else 0
        # Finish only after FFmpeg has flushed its last pass and the output is registered.
        progress = ProgressBar(frame_count * (1 + extra_mask) + preview_frames + 1)
        directory = _output_directory(filename_prefix, save_output)
        if sequence:
            (directory / "frames").mkdir()
            destination = directory / "frames" / f"%08d.{extension}"
        else:
            destination = directory / f"video.{extension}"
        _encode(destination, kind, depth, images, mask, frame_rate, repetitions, ffmpeg,
                progress=progress, audio=audio, order=order, crf=crf)
        paths = [str(destination)]
        if extra_mask:
            (directory / "mask").mkdir()
            mask_destination = directory / "mask" / "%08d.png"
            _encode(mask_destination, "png", "16", images, mask, frame_rate, repetitions, ffmpeg,
                    mask_only=True, progress=progress, order=order)
            paths.append(str(mask_destination))
        metadata = {"format": format, "bit_depth": depth, "fps": frame_rate,
                    "frame_count": frame_count, "alpha_convention": "mask=alpha",
                    "mask_sequence": len(paths) > 1, "has_audio": audio is not None, "files": paths}
        manifest = directory / "manifest.json"
        manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.append(str(manifest))
        preview = destination
        if kind != "vp9":
            preview = directory / "preview.webm"
            _write_preview(preview, images, mask, frame_rate, ffmpeg, progress=progress,
                           audio_path=destination if audio is not None else None, order=order)
        raw = [destination]
        if sequence:
            raw = [manifest, *(destination.with_name(destination.name % index) for index in range(frame_count))]
            if extra_mask:
                raw.extend(mask_destination.with_name(mask_destination.name % index) for index in range(frame_count))
        source = {"width": images.shape[2], "height": images.shape[1], "fps": frame_rate,
                  "frames": frame_count, "has_audio": audio is not None}
        ui = _media_ui(preview, raw, directory if sequence else None, source=source)
        progress.update(1)
        return {"ui": ui, "result": (json.dumps(paths, ensure_ascii=False),)}


NODE_CLASS_MAPPINGS = {LOAD_ID: LoadVideoWithMask, SAVE_ID: VideoCombineWithMask}
NODE_DISPLAY_NAME_MAPPINGS = {LOAD_ID: LOAD_ID, SAVE_ID: SAVE_ID}

_prompt_server = getattr(getattr(sys.modules.get("server"), "PromptServer", None), "instance", None)
if _prompt_server is not None:
    _register_media_route(_prompt_server.routes)
