import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD_ID = "Load Video With Mask (Upload) (endman100)";
const SAVE_ID = "Video Combine With Mask (endman100)";
const PREVIEW_STATE = Symbol("logicLiteVideoMaskPreview");
const LOAD_PREVIEW = Symbol("logicLiteLoadPreviewRequest");
const LOAD_SETTINGS = ["video", "source_path", "frame_load_cap", "skip_first_frames", "select_every_nth",
    "force_rate", "custom_width", "custom_height", "sequence_fps", "missing_alpha"];
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

async function uploadFiles(node, files) {
    if (!files.length) return;
    const sequence = files.length > 1;
    const extension = files[0].name.match(/\.(png|exr)$/i)?.[0].toLowerCase();
    if (sequence && (!extension || !files.every(file => file.name.toLowerCase().endsWith(extension)))) {
        throw new Error("多檔上傳只接受同一格式的 PNG 或 EXR 序列；影片請一次選取一個檔案。");
    }
    const token = globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
    const subfolder = `logic_lite_${token}`;
    let selected;
    for (const file of files) {
        const form = new FormData();
        form.append("image", file);
        form.append("type", "input");
        form.append("subfolder", subfolder);
        form.append("overwrite", "false");
        const response = await api.fetchApi("/upload/image", {method: "POST", body: form});
        if (!response.ok) throw new Error(`上傳失敗：HTTP ${response.status}`);
        const saved = await response.json();
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
        if (!video.autoplay) video.pause();
        else video.play().catch(() => {});
    }
}

function showPreviews(node, media) {
    let state = node[PREVIEW_STATE];
    if (!state && !media.length) return;
    if (!state) {
        const container = document.createElement("div");
        container.style.pointerEvents = "none";
        state = {node, container, videos: [], raw: [], paused: false, hidden: false};
        state.widget = node.addDOMWidget("mask_video_preview", "video", container, {
            serialize: false,
            getMinHeight: () => state.videos.reduce((sum, video) => {
                const ratio = video.videoHeight / video.videoWidth || 9 / 16;
                return sum + Math.min(220, Math.max(1, node.size[0] - 20) * ratio);
            }, 0),
        });
        node[PREVIEW_STATE] = state;
        previews.add(state);
    }
    releaseVideos(state);
    state.raw = media.map(item => item.raw);
    for (const item of media) {
        const video = document.createElement("video");
        video.loop = video.muted = video.playsInline = true;
        Object.assign(video.style, {display: "block", width: "100%", maxHeight: "220px", objectFit: "contain"});
        video.addEventListener("loadedmetadata", () => resizeNode(node));
        video.src = api.apiURL(item.preview);
        state.videos.push(video);
        state.container.append(video);
    }
    state.widget.hidden = state.container.hidden = state.hidden || !state.videos.length;
    updatePlayback(state);
    resizeNode(node);
}

function downloadRaw(url) {
    const link = document.createElement("a");
    link.href = api.apiURL(url);
    link.download = "";
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
                content: `Result ${index + 1}`, callback: () => downloadRaw(url),
            }] : []);
            if (downloads.length === 1) items.push({...downloads[0], content: "Download raw"});
            if (downloads.length > 1) items.push({content: "Download raw", has_submenu: true, submenu: {options: downloads}});
            return [...items, null, ...options];
        };
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LOAD_ID && nodeData.name !== SAVE_ID) return;
        if (nodeData.name === LOAD_ID) {
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = original?.apply(this, arguments);
                this.addWidget("button", "上傳影片／PNG／EXR 序列", null, () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = "video/*,.gif,.mkv,.mts,.m2ts,.ts,.png,.exr";
                    input.multiple = true;
                    input.hidden = true;
                    input.addEventListener("cancel", () => input.remove(), {once: true});
                    input.onchange = async () => {
                        try {
                            await uploadFiles(this, Array.from(input.files ?? []));
                        } catch (error) {
                            app.extensionManager?.toast?.add({severity: "error", summary: "遮罩影片上傳失敗", detail: error.message});
                            console.error("遮罩影片上傳失敗", error);
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
            showPreviews(this, output?.mask_video_previews ?? []);
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
