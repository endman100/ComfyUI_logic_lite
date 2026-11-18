import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD_ID = "Load Video With Mask (Upload) (endman100)";
const SAVE_ID = "Video Combine With Mask (endman100)";
const SEEK_BAR_HEIGHT = 14;
const previews = new Map();

async function uploadFiles(node, files) {
    if (!files.length) return;
    const sequence = files.length > 1;
    if (sequence && !files.every(file => /\.(png|exr)$/i.test(file.name))) {
        throw new Error("多檔上傳只接受 PNG／EXR 序列；影片請一次選取一個檔案。");
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
        widget.value = selected;
        widget.callback?.(selected);
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

function createSeekBar(video) {
    const bar = document.createElement("input");
    Object.assign(bar, {type: "range", min: "0", max: "1", step: "any", value: "0", disabled: true});
    bar.setAttribute("aria-label", "Seek preview");
    Object.assign(bar.style, {display: "block", width: "100%", height: `${SEEK_BAR_HEIGHT}px`, margin: "0",
        accentColor: "#72b7e9", pointerEvents: "auto"});

    function updateProgress() {
        const duration = video.duration;
        bar.disabled = !Number.isFinite(duration) || duration <= 0;
        bar.value = bar.disabled ? "0" : String(video.currentTime / duration);
    }

    for (const event of ["durationchange", "timeupdate", "seeked"]) {
        video.addEventListener(event, updateProgress);
    }
    bar.addEventListener("input", () => {
        if (!bar.disabled) video.currentTime = bar.valueAsNumber * video.duration;
    });
    bar.addEventListener("pointerdown", event => event.stopPropagation());
    bar.addEventListener("keydown", event => event.stopPropagation());
    return bar;
}

function updatePlayback(state) {
    for (const video of state.videos) {
        video.autoplay = !state.paused && !state.hidden;
        if (!video.autoplay) video.pause();
        else video.play().catch(() => {});
    }
}

function showPreviews(node, media) {
    let state = previews.get(node);
    if (!state && !media.length) return;
    if (!state) {
        const container = document.createElement("div");
        container.style.pointerEvents = "none";
        state = {container, videos: [], raw: [], paused: false, hidden: false};
        state.widget = node.addDOMWidget("mask_video_preview", "video", container, {
            serialize: false,
            getMinHeight: () => state.videos.reduce((sum, video) => {
                const ratio = video.videoHeight / video.videoWidth || 9 / 16;
                return sum + Math.min(220, Math.max(1, node.size[0] - 20) * ratio) + SEEK_BAR_HEIGHT;
            }, 0),
        });
        previews.set(node, state);
    }
    releaseVideos(state);
    state.raw = media.map(item => item.raw);
    for (const item of media) {
        const video = document.createElement("video");
        video.loop = video.muted = video.playsInline = true;
        Object.assign(video.style, {display: "block", width: "100%", maxHeight: "220px", objectFit: "contain"});
        video.addEventListener("loadedmetadata", () => resizeNode(node));
        const seekBar = createSeekBar(video);
        video.src = api.apiURL(item.preview);
        state.videos.push(video);
        state.container.append(video, seekBar);
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
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== LOAD_ID && nodeData.name !== SAVE_ID) return;
        if (nodeData.name === LOAD_ID) {
            const original = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = original?.apply(this, arguments);
                this.addWidget("button", "上傳影片／PNG／EXR 序列", null, () => {
                    const input = document.createElement("input");
                    input.type = "file";
                    input.accept = ".mov,.webm,.mp4,.mkv,.avi,.png,.exr";
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
        const menu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            const result = menu?.apply(this, arguments);
            const state = previews.get(this);
            if (!state?.videos.length) return result;
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
                    resizeNode(this);
                },
            }, {
                content: "Sync preview",
                callback: () => {
                    for (const [node, peer] of previews) {
                        if (node.graph !== this.graph) continue;
                        for (const video of peer.videos) video.currentTime = 0;
                    }
                },
            }];
            const downloads = state.raw.flatMap((url, index) => url ? [{
                content: `Result ${index + 1}`, callback: () => downloadRaw(url),
            }] : []);
            if (downloads.length === 1) items.push({...downloads[0], content: "Download raw"});
            if (downloads.length > 1) items.push({content: "Download raw", has_submenu: true, submenu: {options: downloads}});
            options.unshift(...items, null);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            const state = previews.get(this);
            if (state) {
                releaseVideos(state);
                previews.delete(this);
            }
            return removed?.apply(this, arguments);
        };
    },
});
