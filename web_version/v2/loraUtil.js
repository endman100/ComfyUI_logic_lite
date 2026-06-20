import { app } from "../../scripts/app.js";

const NODE_CLASS = "logic loraUtilLoader";
const STORAGE_INPUTS = ["lora_paths", "model_weights", "clip_weights"];

let loraListCache = null;

async function getLoraList() {
    if (loraListCache) return loraListCache;

    try {
        const resp = await fetch("/object_info/LoraLoader");
        if (resp.ok) {
            const data = await resp.json();
            loraListCache = data?.LoraLoader?.input?.required?.lora_name?.[0] ?? null;
        }
    } catch (_) {}

    if (!loraListCache) {
        try {
            const resp = await fetch("/models/loras");
            if (resp.ok) {
                loraListCache = await resp.json();
            }
        } catch (_) {}
    }

    if (!Array.isArray(loraListCache) || loraListCache.length === 0) {
        loraListCache = ["none"];
    }
    return loraListCache;
}

function splitRaw(value) {
    if (value && typeof value === "object" && !Array.isArray(value) && Array.isArray(value.__value__)) {
        value = value.__value__;
    }
    if (Array.isArray(value)) return value.map(v => String(v ?? ""));
    return String(value ?? "").replace(/\r/g, "\n").replace(/,/g, "\n").split("\n");
}

function splitPaths(value) {
    return splitRaw(value).map(v => String(v ?? "").trim()).filter(Boolean);
}

function splitWeights(value) {
    return splitRaw(value)
        .map(v => String(v ?? "").trim())
        .filter(Boolean)
        .map(v => {
            const n = Number(v);
            return Number.isFinite(n) ? n : 1;
        });
}

function stringifyList(values) {
    return values.join("\n");
}

function isLoraUtilNode(node) {
    return node?.comfyClass === NODE_CLASS || node?.type === NODE_CLASS || node?.constructor?.type === NODE_CLASS;
}

function storageFor(node, inputName) {
    return (node.widgets ?? []).find(w => w._logicliteLoraStorage === inputName);
}

function createStorageWidget(node, inputName) {
    const existing = storageFor(node, inputName);
    if (existing) return existing;

    const storageWidget = node.addWidget(
        "text",
        inputName,
        "",
        () => {},
        { serialize: true }
    );
    storageWidget.computeSize = () => [0, -4];
    storageWidget._logicliteLoraStorage = inputName;
    Object.defineProperty(storageWidget, "serializeValue", {
        configurable: true,
        value: () => storageWidget.value,
    });
    return storageWidget;
}

function ensureStorageWidgets(node) {
    if (!isLoraUtilNode(node)) return;
    for (const inputName of STORAGE_INPUTS) {
        createStorageWidget(node, inputName);
    }
}

function getStorages(node) {
    return {
        lora: storageFor(node, "lora_paths"),
        model: storageFor(node, "model_weights"),
        clip: storageFor(node, "clip_weights"),
    };
}

function removeDynamicWidgets(node) {
    const widgets = node.widgets ?? [];
    for (let i = widgets.length - 1; i >= 0; i--) {
        if (widgets[i]._logicliteLoraRow || widgets[i]._logicliteLoraControl) {
            widgets.splice(i, 1);
        }
    }
}

function writeRows(storages, rows) {
    storages.lora.value = stringifyList(rows.map(row => row.lora));
    storages.model.value = stringifyList(rows.map(row => row.modelWeight));
    storages.clip.value = stringifyList(rows.map(row => row.clipWeight));
}

function readRows(storages) {
    const loras = splitPaths(storages.lora.value);
    const modelWeights = splitWeights(storages.model.value);
    const clipWeights = splitWeights(storages.clip.value);
    const count = Math.max(loras.length, modelWeights.length, clipWeights.length, 1);
    const rows = [];

    for (let i = 0; i < count; i++) {
        rows.push({
            lora: loras[i] || "none",
            modelWeight: modelWeights[i] ?? modelWeights[modelWeights.length - 1] ?? 1,
            clipWeight: clipWeights[i] ?? clipWeights[clipWeights.length - 1] ?? modelWeights[i] ?? 1,
        });
    }
    return rows;
}

async function buildLoraUI(node, buildId = node._logicliteLoraBuildId ?? 0) {
    ensureStorageWidgets(node);
    const storages = getStorages(node);
    if (!storages.lora || !storages.model || !storages.clip) return;

    removeDynamicWidgets(node);
    const loraList = await getLoraList();
    if (node._logicliteLoraBuildId !== buildId) return;

    const baseOptions = loraList.includes("none") ? loraList : ["none", ...loraList];
    const rows = readRows(storages);
    writeRows(storages, rows);

    for (let i = 0; i < rows.length; i++) {
        const rowIndex = i;
        const row = rows[rowIndex];
        const options = baseOptions.includes(row.lora) ? baseOptions : [row.lora, ...baseOptions];

        const loraWidget = node.addWidget(
            "combo",
            `lora_${rowIndex + 1}`,
            row.lora,
            (value) => {
                const latest = readRows(storages);
                latest[rowIndex].lora = value;
                writeRows(storages, latest);
            },
            { values: options, serialize: false }
        );
        loraWidget._logicliteLoraRow = true;

        const modelWidget = node.addWidget(
            "number",
            `model_weight_${rowIndex + 1}`,
            row.modelWeight,
            (value) => {
                const latest = readRows(storages);
                latest[rowIndex].modelWeight = Number(value);
                writeRows(storages, latest);
            },
            { min: -100, max: 100, step: 0.01, precision: 3, serialize: false }
        );
        modelWidget._logicliteLoraRow = true;

        const clipWidget = node.addWidget(
            "number",
            `clip_weight_${rowIndex + 1}`,
            row.clipWeight,
            (value) => {
                const latest = readRows(storages);
                latest[rowIndex].clipWeight = Number(value);
                writeRows(storages, latest);
            },
            { min: -100, max: 100, step: 0.01, precision: 3, serialize: false }
        );
        clipWidget._logicliteLoraRow = true;
    }

    const addBtn = node.addWidget(
        "button",
        "+ Add Lora",
        null,
        () => {
            const latest = readRows(storages);
            latest.push({ lora: "none", modelWeight: 1, clipWeight: 1 });
            writeRows(storages, latest);
            buildLoraUI(node);
        },
        { serialize: false }
    );
    addBtn._logicliteLoraControl = true;

    const removeBtn = node.addWidget(
        "button",
        "- Remove Last",
        null,
        () => {
            const latest = readRows(storages);
            if (latest.length > 1) latest.pop();
            writeRows(storages, latest);
            buildLoraUI(node);
        },
        { serialize: false }
    );
    removeBtn._logicliteLoraControl = true;

    node.setSize(node.computeSize());
    app.graph.setDirtyCanvas(true, true);
}

function scheduleBuild(node) {
    if (!isLoraUtilNode(node)) return;
    const buildId = (node._logicliteLoraBuildId ?? 0) + 1;
    node._logicliteLoraBuildId = buildId;
    requestAnimationFrame(() => buildLoraUI(node, buildId));
}

app.registerExtension({
    name: "LogicLite.LoraUtilLoader",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);
            if (!app.configuringGraph) scheduleBuild(this);
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            origOnConfigure?.apply(this, arguments);
            scheduleBuild(this);
        };
    },

    nodeCreated(node) {
        scheduleBuild(node);
    },

    loadedGraphNode(node) {
        scheduleBuild(node);
    },

    setup() {
        setTimeout(() => {
            for (const node of app.graph?._nodes ?? []) {
                scheduleBuild(node);
            }
        }, 500);
    },
});
