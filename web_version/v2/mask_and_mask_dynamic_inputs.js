import { app } from "../../scripts/app.js";

const NODE_NAMES = new Set([
    "Mask Or Mask (endman100)",
    "Mask And Mask (endman100)",
    "Mask Sub Mask (endman100)",
    "Mask Concat (endman100)",
    "Temporal Mask Or (endman100)",
    "Temporal Mask And (endman100)",
]);
const MASK_INPUT_RE = /^mask[1-9]\d*$/;

function isMaskInput(input) {
    return input?.type === "MASK" && MASK_INPUT_RE.test(input.name ?? "");
}

function getMaskSlots(node) {
    return (node.inputs ?? [])
        .map((input, index) => ({ input, index }))
        .filter(({ input }) => isMaskInput(input));
}

function isConnected(input) {
    return input?.link != null;
}

function stabilizeMaskInputs(node) {
    if (node.__maskHandleStabilizing) {
        return;
    }

    node.__maskHandleStabilizing = true;
    try {
        if (!node.inputs) {
            node.inputs = [];
        }

        if (!getMaskSlots(node).some(({ input }) => input.name === "mask1")) {
            node.addInput("mask1", "MASK");
        }

        const maskSlots = getMaskSlots(node);
        const lastConnected = Math.max(0, ...maskSlots
            .filter(({ input }) => isConnected(input))
            .map(({ input }) => Number(input.name.slice(4))));
        const spareName = `mask${lastConnected + 1}`;

        // Only trim empty slots beyond the spare. Never compact middle gaps:
        // mask1 remains Sub's base even when mask2+ stay wired after a disconnect.
        for (const { input, index } of maskSlots.reverse()) {
            if (!isConnected(input) && Number(input.name.slice(4)) > lastConnected + 1) {
                node.removeInput(index);
            }
        }

        if (!getMaskSlots(node).some(({ input }) => input.name === spareName)) {
            node.addInput(spareName, "MASK");
        }

        node.setSize?.(node.computeSize());
        node.graph?.setDirtyCanvas(true, true);
    } finally {
        node.__maskHandleStabilizing = false;
    }
}

function scheduleMaskInputs(node) {
    if (node.__maskHandleStabilizing || node.__maskHandleScheduled) {
        return;
    }
    node.__maskHandleScheduled = true;
    requestAnimationFrame(() => {
        node.__maskHandleScheduled = false;
        // Link restoration and same-slot reconnection must finish before trimming.
        if (app.configuringGraph) {
            scheduleMaskInputs(node);
        } else {
            stabilizeMaskInputs(node);
        }
    });
}

app.registerExtension({
    name: "endman100.DynamicMaskInputs",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            if (app.configuringGraph) {
                scheduleMaskInputs(this);
            } else {
                stabilizeMaskInputs(this);
            }
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            scheduleMaskInputs(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            const result = onConnectionsChange?.apply(this, arguments);
            if (type === LiteGraph.INPUT) {
                scheduleMaskInputs(this);
            }
            return result;
        };
    },
});
