import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

// Displays output text for "Show Any" and "Show Tensor Shape" nodes.
// Mirrors the proven pattern from pysssss/ShowText.

function formatValue(v) {
    if (v !== null && typeof v === "object") {
        return JSON.stringify(v, null, 2);
    }
    return String(v);
}

function populate(text) {
    // Clear all existing dynamically-added text widgets
    if (this.widgets) {
        for (let i = this.widgets.length - 1; i >= 0; i--) {
            if (this.widgets[i].__logiclite_text) {
                this.widgets[i].onRemove?.();
                this.widgets.splice(i, 1);
            }
        }
    }

    const values = Array.isArray(text) ? text : [text];
    for (const val of values) {
        let list = Array.isArray(val) ? val : [val];
        for (const item of list) {
            const w = ComfyWidgets["STRING"](
                this,
                "text_" + (this.widgets?.length ?? 0),
                ["STRING", { multiline: true }],
                app
            ).widget;
            w.inputEl.readOnly = true;
            w.inputEl.style.opacity = 0.6;
            w.value = formatValue(item);
            w.__logiclite_text = true;
        }
    }

    requestAnimationFrame(() => {
        const sz = this.computeSize();
        if (sz[0] < this.size[0]) sz[0] = this.size[0];
        if (sz[1] < this.size[1]) sz[1] = this.size[1];
        this.onResize?.(sz);
        app.graph.setDirtyCanvas(true, false);
    });
}

function addTextDisplay(nodeType) {
    // Called when node executes and returns ui.text
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        if (message?.text) {
            populate.call(this, message.text);
        }
    };

    // Store widget values before configure clears them (new frontend behaviour)
    const VALUES = Symbol();
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function () {
        this[VALUES] = arguments[0]?.widgets_values;
        return configure?.apply(this, arguments);
    };

    // Restore saved values after configure (e.g. when loading a saved workflow)
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        onConfigure?.apply(this, arguments);
        if (this[VALUES]?.length) {
            requestAnimationFrame(() => {
                populate.call(this, this[VALUES]);
            });
        }
    };
}

app.registerExtension({
    name: "LogicLite.ShowAny",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (
            nodeData.name === "logic showAnything" ||
            nodeData.name === "logic showTensorShape"
        ) {
            addTextDisplay(nodeType);
        }
    },
});
