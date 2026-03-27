import { app } from "../../scripts/app.js";

// 快取 input 資料夾的圖片清單，避免重複請求
let _imageListCache = null;
async function getImageList() {
    if (_imageListCache) return _imageListCache;
    try {
        const resp = await fetch("/object_info/LoadImage");
        if (resp.ok) {
            const data = await resp.json();
            _imageListCache = data?.LoadImage?.input?.required?.image?.[0] ?? [];
        }
    } catch (_) {}
    if (!_imageListCache || _imageListCache.length === 0) {
        _imageListCache = ["none"];
    }
    return _imageListCache;
}

function splitRaw(value) {
    return String(value ?? "").replace(/\r/g, "\n").split("\n");
}

function stringifyPaths(paths) {
    return paths.join("\n");
}

function removeDynamicWidgets(node) {
    const widgets = node.widgets ?? [];
    for (let i = widgets.length - 1; i >= 0; i--) {
        if (widgets[i]._logiclitePathRow || widgets[i]._logiclitePathControl) {
            widgets.splice(i, 1);
        }
    }
}

// 依 storageWidget 目前的值，從 ComfyUI /view 載入所有圖片並更新預覽
function refreshPreview(node, storageWidget) {
    const paths = splitRaw(storageWidget.value).filter(Boolean);
    if (!paths.length) {
        node._logiclitePreviewImgs = [];
        node.setSize([node.size[0], node._logicliteBaseHeight ?? node.size[1]]);
        app.graph.setDirtyCanvas(true, false);
        return;
    }
    Promise.all(paths.map(p => new Promise(resolve => {
        // p 可能是 ComfyUI input 目錄的 filename，或 subfolder/filename
        const parts = p.split("/");
        const filename = parts.pop();
        const subfolder = parts.join("/");
        const url = `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=${encodeURIComponent(subfolder)}&rand=${Math.random()}`;
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => resolve(null);
        img.src = url;
    }))).then(imgs => {
        const valid = imgs.filter(Boolean);
        node._logiclitePreviewImgs = valid;
        const base = node._logicliteBaseHeight ?? node.size[1];
        const nodeW = node.size[0];
        let extraH = valid.length ? 8 : 0;
        for (const img of valid) {
            extraH += Math.round(img.naturalHeight * (nodeW / img.naturalWidth)) + 4;
        }
        node.setSize([nodeW, base + extraH]);
        app.graph.setDirtyCanvas(true, false);
    });
}

async function buildPathUI(node, storageWidget) {
    // 清空預覽圖，確保 computeSize 反映的是純 widget 高度
    node._logiclitePreviewImgs = [];
    removeDynamicWidgets(node);
    const imageList = await getImageList();
    console.log("[LLB] buildPathUI imageList count:", imageList.length);
    const paths = splitRaw(storageWidget.value);

    for (let i = 0; i < paths.length; i++) {
        const idx = i;
        const current = paths[idx] || "none";
        // 永遠在頂部加入 "none" 選項，其後才是實際圖片清單
        const baseOptions = imageList.includes("none") ? imageList : ["none", ...imageList];
        const options = baseOptions.includes(current) ? baseOptions : [current, ...baseOptions];
        const row = node.addWidget(
            "combo",
            `path_${idx + 1}`,
            current,
            (value) => {
                console.log("[LLB] combo[" + idx + "] changed to:", value);
                const latest = splitRaw(storageWidget.value);
                if (idx < latest.length) latest[idx] = value;
                storageWidget.value = stringifyPaths(latest);
                refreshPreview(node, storageWidget);
            },
            { values: options }
        );
        row._logiclitePathRow = true;

        const uploadBtn = node.addWidget(
            "button",
            `📤 Upload → slot ${idx + 1}`,
            null,
            () => {
                const input = document.createElement("input");
                input.type = "file";
                input.accept = "image/*";
                input.onchange = async () => {
                    const file = input.files?.[0];
                    if (!file) return;
                    const formData = new FormData();
                    formData.append("image", file);
                    formData.append("overwrite", "false");
                    try {
                        const resp = await fetch("/upload/image", { method: "POST", body: formData });
                        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
                        const result = await resp.json();
                        const filename = result.name ?? file.name;
                        _imageListCache = null;
                        const latest = splitRaw(storageWidget.value);
                        if (idx < latest.length) latest[idx] = filename;
                        storageWidget.value = stringifyPaths(latest);
                        await buildPathUI(node, storageWidget);
                    } catch (e) {
                        console.error("[LLB] Upload failed:", e);
                    }
                };
                input.click();
            },
            { serialize: false }
        );
        uploadBtn._logiclitePathRow = true;
    }

    const addBtn = node.addWidget(
        "button",
        "➕ Add Path",
        null,
        () => {
            const latest = splitRaw(storageWidget.value);
            latest.push("none");
            storageWidget.value = stringifyPaths(latest);
            buildPathUI(node, storageWidget);
        },
        { serialize: false }
    );
    addBtn._logiclitePathControl = true;

    const removeBtn = node.addWidget(
        "button",
        "➖ Remove Last",
        null,
        () => {
            const latest = splitRaw(storageWidget.value);
            if (latest.length > 0) latest.pop();
            storageWidget.value = stringifyPaths(latest);
            buildPathUI(node, storageWidget);
        },
        { serialize: false }
    );
    removeBtn._logiclitePathControl = true;

    // buildPathUI 結束後，此時 _logiclitePreviewImgs 為空，
    // computeSize 回傳的即為純 widget 基礎高度
    node.setSize(node.computeSize());
    node._logicliteBaseHeight = node.size[1];
    console.log("[LLB] buildPathUI done, _logicliteBaseHeight=", node._logicliteBaseHeight);
    // 立即顯示目前選取的圖片
    refreshPreview(node, storageWidget);
}

app.registerExtension({
    name: "LogicLite.LoadImageBatch",

    getCustomWidgets(app) {
        console.log("[LLB] getCustomWidgets called");
        return {
            LIST(node, inputName, inputData, _app) {
                if (node.comfyClass !== "logic loadImageBatch") {
                    return {};
                }
                console.log("[LLB] LIST widget factory called for", node.comfyClass);
                const storageWidget = node.addWidget(
                    "text",
                    inputName,
                    "",
                    () => {},
                    { serialize: true }
                );
                storageWidget.computeSize = () => [0, -4];
                storageWidget._logicliteStorage = true;
                return { widget: storageWidget };
            }
        };
    },

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "logic loadImageBatch") return;
        console.log("[LLB] beforeRegisterNodeDef hooking logic loadImageBatch");

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);
            // workflow 載入時 app.configuringGraph 為 true，onConfigure 會處理，此處跳過避免重複
            if (app.configuringGraph) return;
            const storageWidget = (this.widgets ?? []).find(w => w._logicliteStorage);
            console.log("[LLB] onNodeCreated, storageWidget found:", !!storageWidget);
            if (storageWidget) buildPathUI(this, storageWidget);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origOnConfigure?.apply(this, arguments);
            const storageWidget = (this.widgets ?? []).find(w => w._logicliteStorage);
            if (storageWidget) buildPathUI(this, storageWidget);
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            origOnExecuted?.apply(this, arguments);
            // 圖片 UI 由下拉選單即時控制，不在執行後更新
        };

        // onDrawBackground: (0,0) = 節點 body 頂端（title 以下）
        const origOnDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            origOnDrawBackground?.apply(this, arguments);
            const imgs = this._logiclitePreviewImgs;
            if (!imgs?.length || this.flags?.collapsed) return;
            const nodeW = this.size[0];
            const base = this._logicliteBaseHeight ?? this.size[1];
            let y = base + 4;
            for (const img of imgs) {
                const scale = nodeW / img.naturalWidth;
                const drawH = Math.round(img.naturalHeight * scale);
                ctx.drawImage(img, 0, y, nodeW, drawH);
                y += drawH + 4;
            }
        };
    },
});

