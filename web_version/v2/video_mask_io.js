import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const LOAD_ID = "Load Video With Mask (Upload) (endman100)";
const SAVE_ID = "Video Combine With Mask (endman100)";

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

function releasePreview(node) {
    node._logicLiteMaskMedia?.querySelectorAll("video").forEach(video => {
        video.pause();
        video.removeAttribute("src");
        video.load();
    });
}

app.registerExtension({
    name: "endman100.VideoMaskIO",
    async beforeRegisterNodeDef(nodeType, nodeData) {
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
        }
        if (nodeData.name !== SAVE_ID) return;
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            const result = executed?.apply(this, arguments);
            releasePreview(this);
            if (!this._logicLiteMaskMedia) {
                this._logicLiteMaskMedia = document.createElement("div");
                this.addDOMWidget("mask_video_preview", "div", this._logicLiteMaskMedia, {serialize: false});
            }
            const container = this._logicLiteMaskMedia;
            container.replaceChildren();
            for (const file of output?.mask_videos ?? []) {
                const video = document.createElement("video");
                video.controls = true;
                video.loop = true;
                video.style.width = "100%";
                const query = new URLSearchParams({filename: file.filename, subfolder: file.subfolder, type: file.type});
                video.src = api.apiURL(`/view?${query}`);
                container.append(video);
            }
            const paths = document.createElement("pre");
            paths.style.whiteSpace = "pre-wrap";
            paths.style.overflowWrap = "anywhere";
            paths.textContent = (output?.mask_video_files ?? []).join("\n");
            container.append(paths);
            this.setSize?.(this.computeSize());
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            releasePreview(this);
            return removed?.apply(this, arguments);
        };
    },
});
