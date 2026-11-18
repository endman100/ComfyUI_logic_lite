import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD_ID = "Load Video With Mask (Upload) (endman100)";
const SAVE_ID = "Video Combine With Mask (endman100)";
const PREVIEW_STATE = Symbol("logicLiteVideoMaskPreview");
const LOAD_PREVIEW = Symbol("logicLiteLoadPreviewRequest");
const UPLOAD_PROGRESS = Symbol("logicLiteUploadProgress");
const LOAD_SETTINGS = ["video", "source_path", "frame_load_cap", "skip_first_frames", "select_every_nth",
    "force_rate", "custom_width", "custom_height", "sequence_fps", "missing_alpha"];
const SEQUENCE_FILE = /\.(png|exr)$/i;
const VIDEO_FILE = /\.(avi|flv|gif|m4v|mkv|mov|mp4|mpeg|mpg|mts|m2ts|ts|webm|wmv)$/i;
const previews = new Set();

function rememberVideo(widget, selected) {
    const values = widget.options.values ?? [];
    if (selected && !values.includes(selected)) widget.options.values = [...values, selected];
}

function stopLoadPreview(node) {
    const request = node[LOAD_PREVIEW];
    if (request) {
        clearTimeout(request.timer);
        request.controller.abort();
        delete node[LOAD_PREVIEW];
    }
}

function refreshLoadPreview(node) {
    stopLoadPreview(node);
    const settings = Object.fromEntries((node.widgets ?? []).filter(widget => LOAD_SETTINGS.includes(widget.name))
        .map(widget => [widget.name, widget.value]));
    if (!(settings.source_path?.trim() || settings.video)) {
        showPreviews(node, []);
        return;
    }
    const request = {controller: new AbortController()};
    node[LOAD_PREVIEW] = request;
    request.timer = setTimeout(async () => {
        try {
            const response = await api.fetchApi("/logiclite/video-mask/load-preview", {
                method: "POST", headers: {"Content-Type": "application/json"},
                body: JSON.stringify(settings), signal: request.controller.signal,
            });
            const data = await response.json();
            if (node[LOAD_PREVIEW] !== request) return;
            if (!response.ok && response.status !== 403) throw new Error(data.error ?? `HTTP ${response.status}`);
            showPreviews(node, data.mask_video_previews ?? []);
        } catch (error) {
            if (node[LOAD_PREVIEW] === request) {
                showPreviews(node, []);
                app.extensionManager?.toast?.add({severity: "error", summary: "影片預覽失敗", detail: error.message});
            }
        }
    }, 300);
}

function isImageSequence(files) {
    const extension = files[0]?.name.match(SEQUENCE_FILE)?.[0].toLowerCase();
    return Boolean(extension && files.every(file => file.name.toLowerCase().endsWith(extension)));
}

function acceptsDrop(files) {
    if (files.length !== 1) return isImageSequence(files);
    const file = files[0];
    return SEQUENCE_FILE.test(file.name) || VIDEO_FILE.test(file.name) || file.type?.startsWith("video/");
}

function uploadFile(form, onProgress) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", api.apiURL("/upload/image"));
        if (api.user) xhr.setRequestHeader("Comfy-User", api.user);
        xhr.responseType = "json";
        xhr.upload.onprogress = event => {
            if (event.lengthComputable && event.total) onProgress(event.loaded / event.total);
        };
        xhr.onload = () => {
            if (xhr.status >= 200 && xhr.status < 300 && xhr.response?.name) resolve(xhr.response);
            else reject(new Error(`上傳失敗：HTTP ${xhr.status}`));
        };
        xhr.onerror = xhr.onabort = () => reject(new Error("上傳連線中斷"));
        xhr.send(form);
    });
}

async function uploadFiles(node, files, progress) {
    const sequence = files.length > 1;
    if (sequence && !isImageSequence(files)) {
        throw new Error("多檔上傳只接受同一格式的 PNG 或 EXR 序列；影片請一次選取一個檔案。");
    }
    const token = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
    const subfolder = `logic_lite_${token}`;
    const total = files.reduce((bytes, file) => bytes + (file.size || 1), 0);
    let sent = 0;
    let selected;
    for (const file of files) {
        const form = new FormData();
        form.append("image", file);
        form.append("type", "input");
        form.append("subfolder", subfolder);
        form.append("overwrite", "false");
        const size = file.size || 1;
        const saved = await uploadFile(form, fraction => { progress.value = (sent + size * fraction) / total; });
        sent += size;
        progress.value = sent / total;
        selected = sequence ? saved.subfolder : [saved.subfolder, saved.name].filter(Boolean).join("/");
    }
    const widget = node.widgets?.find(widget => widget.name === "video");
    if (widget) {
        rememberVideo(widget, selected);
        const previous = widget.value;
        widget.value = selected;
        const path = node.widgets.find(widget => widget.name === "source_path");
        if (path) path.value = "";
        widget.callback?.(selected, app.canvas, node);
        node.onWidgetChanged?.("video", selected, previous, widget);
        app.graph?.setDirtyCanvas(true, true);
    }
}

async function uploadWithFeedback(node, files) {
    if (!files.length) return;
    let upload = node[UPLOAD_PROGRESS];
    if (!upload) {
        const container = document.createElement("div");
        const progress = document.createElement("progress");
        progress.setAttribute("aria-label", "上傳進度");
        progress.title = "上傳進度";
        progress.max = 1;
        progress.style.width = "100%";
        container.style.pointerEvents = "none";
        container.append(progress);
        const widget = node.addDOMWidget("mask_video_upload_progress", "progress", container, {
            serialize: false, getMinHeight: () => 26, getMaxHeight: () => 26,
        });
        upload = node[UPLOAD_PROGRESS] = {container, progress, widget};
    } else if (!upload.widget.hidden) return;
    upload.widget.hidden = upload.container.hidden = false;
    upload.progress.value = 0;
    resizeNode(node);
    try {
        await uploadFiles(node, files, upload.progress);
    } catch (error) {
        app.extensionManager?.toast?.add({severity: "error", summary: "遮罩影片上傳失敗", detail: error.message});
        console.error("遮罩影片上傳失敗", error);
    } finally {
        upload.widget.hidden = upload.container.hidden = true;
        if (node.graph) resizeNode(node);
    }
}

function releaseVideos(state) {
    for (const video of state.videos) {
        video.pause();
        video.removeAttribute("src");
        video.load();
    }
    state.videos = [];
    state.container.replaceChildren();
}

function resizeNode(node) {
    node.setSize([node.size[0], node.computeSize()[1]]);
    node.setDirtyCanvas(true, true);
}

function updateMaskModeVisibility(node) {
    const format = node.widgets?.find(widget => widget.name === "format")?.value;
    const maskMode = node.widgets?.find(widget => widget.name === "mask_mode");
    const hidden = typeof format !== "string" || !format.startsWith("[no mask]");
    if (maskMode && Boolean(maskMode.hidden) !== hidden) {
        maskMode.hidden = hidden;
        resizeNode(node);
    }
}

function updatePlayback(state) {
    for (const video of state.videos) {
        video.autoplay = !state.paused && !state.hidden;
        if (!video.autoplay) {
            video.muted = true;
            video.pause();
        } else video.play().catch(() => {});
    }
}

function showPreviews(node, media, isLoader = true) {
    let state = node[PREVIEW_STATE];
    if (!state && !media.length) return;
    if (!state) {
        const container = document.createElement("div");
        container.style.pointerEvents = "none";
        state = {node, container, isLoader, videos: [], raw: [], paused: false, hidden: false, muted: false};
        const previewHeight = () => {
            const padding = (state.widget?.margin ?? 10) * 2;
            const width = Math.max(1, node.size[0] - padding);
            return padding + state.videos.reduce((height, video) =>
                height + width * (video.videoHeight / video.videoWidth || 9 / 16), 0);
        };
        let lastWidth = node.size[0];
        state.widget = node.addDOMWidget("mask_video_preview", "video", container, {
            serialize: false, getMinHeight: previewHeight, getMaxHeight: previewHeight,
            afterResize: () => {
                if (lastWidth === node.size[0]) return;
                lastWidth = node.size[0];
                if (!state.widget.hidden && node.graph) resizeNode(node);
            },
        });
        node[PREVIEW_STATE] = state;
        previews.add(state);
    }
    releaseVideos(state);
    state.raw = media.map(item => item.raw);
    state.source = media[0]?.source;
    for (const item of media) {
        const video = document.createElement("video");
        video.loop = video.muted = video.playsInline = true;
        Object.assign(video.style, {display: "block", width: "100%", height: "auto"});
        if (state.isLoader) {
            video.style.pointerEvents = "auto";
            const playHoverAudio = () => {
                if (state.muted || state.paused || state.hidden) return;
                video.muted = false;
                video.play().catch(() => {
                    // A browser may require activation before audible playback.
                    video.muted = true;
                    if (!state.paused && !state.hidden) video.play().catch(() => {});
                });
            };
            video.addEventListener("pointerenter", playHoverAudio);
            video.addEventListener("pointerleave", () => { video.muted = true; });
            video.addEventListener("click", playHoverAudio);
            video.addEventListener("contextmenu", event => {
                event.preventDefault();
                app.canvas.adjustMouseEvent(event);
                app.canvas.processContextMenu(node, event);
            });
            video.addEventListener("wheel", event => app.canvas.processMouseWheel(event), {passive: false});
            for (const type of ["dragover", "drop"]) {
                video.addEventListener(type, event => {
                    if (!app.canvas.canvas.dispatchEvent(new DragEvent(type, event))) event.preventDefault();
                });
            }
        }
        video.addEventListener("loadedmetadata", () => resizeNode(node));
        video.src = api.apiURL(item.preview);
        state.videos.push(video);
        state.container.append(video);
    }
    state.widget.hidden = state.container.hidden = state.hidden || !state.videos.length;
    updatePlayback(state);
    resizeNode(node);
}

function downloadFile(url, filename = "") {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.append(link);
    link.click();
    link.remove();
}

app.registerExtension({
    name: "endman100.VideoMaskIO",
    setup() {
        // Let LiteGraph finish its node-level hooks before adding our entries.
        const canvas = globalThis.LGraphCanvas.prototype;
        const getNodeMenuOptions = canvas.getNodeMenuOptions;
        canvas.getNodeMenuOptions = function (node) {
            const options = getNodeMenuOptions.apply(this, arguments);
            const state = node[PREVIEW_STATE];
            if (!state?.videos.length) return options;
            const items = [{
                content: state.paused ? "Resume preview" : "Pause preview",
                callback: () => {
                    state.paused = !state.paused;
                    updatePlayback(state);
                },
            }, {
                content: state.hidden ? "Show preview" : "Hide preview",
                callback: () => {
                    state.hidden = !state.hidden;
                    state.widget.hidden = state.container.hidden = state.hidden;
                    updatePlayback(state);
                    resizeNode(node);
                },
            }, {
                content: "Sync preview",
                callback: () => {
                    for (const peer of previews) {
                        if (peer.node.graph !== node.graph) continue;
                        for (const video of peer.videos) video.currentTime = 0;
                    }
                },
            }];
            const downloads = state.raw.flatMap((url, index) => url ? [{
                content: `Result ${index + 1}`, callback: () => downloadFile(api.apiURL(url)),
            }] : []);
            if (downloads.length === 1) items.push({...downloads[0], content: "Download raw"});
            if (downloads.length > 1) items.push({content: "Download raw", has_submenu: true, submenu: {options: downloads}});
            if (state.isLoader) {
                const video = state.videos[0];
                items.push({content: "Open preview", callback: () => window.open(video.src, "_blank", "noopener")},
                    {content: "Save preview", callback: () => downloadFile(video.src, "preview.webm")});
                const source = state.source;
                if (source?.has_audio) items.push({
                    content: state.muted ? "Unmute preview" : "Mute preview",
                    callback: () => { state.muted = !state.muted; if (state.muted) video.muted = true; },
                });
                if (source) items.push({disabled: true,
                    content: `${source.width}×${source.height} @ ${Number(source.fps.toFixed(2))} fps` +
                        (source.frames ? ` · ${source.frames} frames` : ""),
                });
            }
            return [...items, null, ...options];
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LOAD_ID && nodeData.name !== SAVE_ID) return;
        if (nodeData.name === LOAD_ID) {
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = original?.apply(this, arguments);
                const dragOver = this.onDragOver;
                this.onDragOver = function (event) {
                    const transfer = event.dataTransfer;
                    const files = Array.from(transfer?.files ?? []);
                    // The browser hides filenames during dragover. Validate the actual files on drop.
                    if (files.length ? acceptsDrop(files) : Array.from(transfer?.types ?? []).includes("Files")) return true;
                    return dragOver?.apply(this, arguments) ?? false;
                };
                const dragDrop = this.onDragDrop;
                this.onDragDrop = function (event) {
                    const files = Array.from(event.dataTransfer?.files ?? []);
                    if (!acceptsDrop(files)) return dragDrop?.apply(this, arguments) ?? false;
                    event.preventDefault();
                    event.stopPropagation();
                    void uploadWithFeedback(this, files);
                    return true;
                };
                this.addWidget("button", "上傳影片／PNG／EXR 序列", null, () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = "video/*,.gif,.mkv,.mts,.m2ts,.ts,.png,.exr";
                    input.multiple = true;
                    input.hidden = true;
                    input.addEventListener("cancel", () => input.remove(), {once: true});
                    input.onchange = async () => {
                        try {
                            await uploadWithFeedback(this, Array.from(input.files ?? []));
                        } finally {
                            input.remove();
                        }
                    };
                    document.body.append(input);
                    input.click();
                }, {serialize: false});
                return result;
            };
            const configured = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = configured?.apply(this, arguments);
                const video = this.widgets?.find(widget => widget.name === "video");
                if (video) rememberVideo(video, video.value);
                refreshLoadPreview(this);
                return result;
            };
            const changed = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (name) {
                const result = changed?.apply(this, arguments);
                if (LOAD_SETTINGS.includes(name)) refreshLoadPreview(this);
                return result;
            };
        } else {
            const created = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = created?.apply(this, arguments);
                updateMaskModeVisibility(this);
                return result;
            };
            const configured = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = configured?.apply(this, arguments);
                updateMaskModeVisibility(this);
                return result;
            };
            const changed = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (name) {
                const result = changed?.apply(this, arguments);
                if (name === "format") updateMaskModeVisibility(this);
                return result;
            };
        }
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = executed?.apply(this, arguments);
            showPreviews(this, output?.mask_video_previews ?? [], nodeData.name === LOAD_ID);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            stopLoadPreview(this);
            const state = this[PREVIEW_STATE];
            if (state) {
                releaseVideos(state);
                previews.delete(state);
                delete this[PREVIEW_STATE];
            }
            return removed?.apply(this, arguments);
        };
    },
});
