const { app } = window.comfyAPI.app;

const NODE_NAMES = new Set([
    "Mask Or Mask (endman100)",
    "Mask And Mask (endman100)",
    "Mask Sub Mask (endman100)",
    "Mask Concat (endman100)",
    "Mask Concat Long Image (endman100)",
]);
const MASK_INPUT_RE = /^mask\d+$/;

function isMaskInput(input) {
    return input?.type === "MASK" && MASK_INPUT_RE.test(input.name ?? "");
}

function getMaskSlots(node) {
    return (node.inputs ?? [])
        .map((input, index) => ({ input, index }))
        .filter(({ input }) => isMaskInput(input));
}

function renameMaskInputs(node) {
    getMaskSlots(node).forEach(({ input }, index) => {
        input.name = `mask${index + 1}`;
    });
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

        let maskSlots = getMaskSlots(node);
        while (maskSlots.length < 2) {
            node.addInput(`mask${maskSlots.length + 1}`, "MASK");
            maskSlots = getMaskSlots(node);
        }

        renameMaskInputs(node);
        maskSlots = getMaskSlots(node);

        while (
            maskSlots.length > 2 &&
            !isConnected(maskSlots[maskSlots.length - 1].input) &&
            !isConnected(maskSlots[maskSlots.length - 2].input)
        ) {
            node.removeInput(maskSlots[maskSlots.length - 1].index);
            maskSlots = getMaskSlots(node);
        }

        if (maskSlots.length > 0 && isConnected(maskSlots[maskSlots.length - 1].input)) {
            node.addInput(`mask${maskSlots.length + 1}`, "MASK");
        }

        renameMaskInputs(node);
        node.setSize?.(node.computeSize());
        node.graph?.setDirtyCanvas(true, true);
    } finally {
        node.__maskHandleStabilizing = false;
    }
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
            stabilizeMaskInputs(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            stabilizeMaskInputs(this);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type) {
            const result = onConnectionsChange?.apply(this, arguments);
            if (type === LiteGraph.INPUT) {
                stabilizeMaskInputs(this);
            }
            return result;
        };
    },
});
