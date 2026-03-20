import { app } from "../../scripts/app.js";

const MAX_IMAGES = 16;

// Hide a widget by collapsing its size to zero (standard pattern used by many ComfyUI extensions)
function hideWidget(widget) {
    if (widget._origComputeSize === undefined) {
        widget._origComputeSize = widget.computeSize;
    }
    widget.computeSize = () => [0, -4];
}

// Restore a previously hidden widget
function showWidget(widget) {
    if (widget._origComputeSize !== undefined) {
        widget.computeSize = widget._origComputeSize;
        delete widget._origComputeSize;
    } else {
        delete widget.computeSize;
    }
}

app.registerExtension({
    name: "LogicLite.LoadImageBatch",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "logic loadImageBatch") return;

        // Find the combo widget AND its upload button for a given slot index (2..MAX_IMAGES)
        // The upload button is an addWidget("button",...) added right after the combo
        // with { serialize: false, canvasOnly: true }
        nodeType.prototype._getImageWidgets = function (slotIndex) {
            const name = `image_${slotIndex}`;
            const widgets = this.widgets ?? [];
            const result = [];
            for (let i = 0; i < widgets.length; i++) {
                if (widgets[i].name === name) {
                    result.push(widgets[i]);  // the combo
                    // upload button is right after: type "button" with serialize:false
                    const next = widgets[i + 1];
                    if (next && next.type === "button" && next.options?.serialize === false) {
                        result.push(next);
                    }
                    break;
                }
            }
            return result;
        };

        // Show/hide image widgets according to this._shownCount
        // _shownCount = number of EXTRA slots shown beyond image_1
        nodeType.prototype._applyVisibility = function () {
            const count = this._shownCount ?? 0;
            for (let i = 2; i <= MAX_IMAGES; i++) {
                const visible = i - 1 <= count;
                for (const w of this._getImageWidgets(i)) {
                    visible ? showWidget(w) : hideWidget(w);
                }
            }
        };

        // Calculate how many extra slots should be visible based on current widget values
        // Used for migration of old workflows that don't have extra._shownCount
        nodeType.prototype._countActiveSlots = function () {
            let maxActive = 0;
            for (let i = 2; i <= MAX_IMAGES; i++) {
                const widgets = this.widgets ?? [];
                const w = widgets.find(w => w.name === `image_${i}`);
                if (w && w.value && w.value !== "none") {
                    maxActive = i - 1;
                }
            }
            return maxActive;
        };

        // Fetch all currently selected images and render them as node preview
        nodeType.prototype._refreshPreview = function () {
            const node = this;
            const names = [];
            for (let i = 1; i <= MAX_IMAGES; i++) {
                const w = (this.widgets ?? []).find(w => w.name === `image_${i}`);
                if (w && w.value && w.value !== "none") {
                    names.push(w.value);
                }
            }
            if (!names.length) {
                node.imgs = [];
                app.graph.setDirtyCanvas(true, false);
                return;
            }
            Promise.all(
                names.map(
                    name =>
                        new Promise(resolve => {
                            const img = new Image();
                            img.onload = () => resolve(img);
                            img.onerror = () => resolve(null);
                            img.src = `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=&rand=${Math.random()}`;
                        })
                )
            ).then(imgs => {
                node.imgs = imgs.filter(Boolean);
                node.imageIndex = 0;
                node.setSizeForImage?.();
                app.graph.setDirtyCanvas(true, false);
            });
        };

        // Hook into every image_i combo widget so preview updates on value change
        nodeType.prototype._setupLivePreview = function () {
            const node = this;
            for (let i = 1; i <= MAX_IMAGES; i++) {
                const w = (this.widgets ?? []).find(w => w.name === `image_${i}`);
                if (!w || w._livePreviewHooked) continue;
                const origCallback = w.callback;
                w.callback = function () {
                    origCallback?.apply(this, arguments);
                    node._refreshPreview();
                };
                w._livePreviewHooked = true;
            }
        };

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            // For fresh nodes, use 0; for pre-configured nodes extra might already be set
            this._shownCount = this.extra?._shownCount ?? 0;
            this._applyVisibility();
            this._setupLivePreview();

            // "➕ Add Image" button (not serialized)
            this.addWidget("button", "➕ Add Image", null, () => {
                if (this._shownCount < MAX_IMAGES - 1) {
                    this._shownCount++;
                    if (!this.extra) this.extra = {};
                    this.extra._shownCount = this._shownCount;
                    this._applyVisibility();
                    this._setupLivePreview();  // hook newly visible widgets
                    this.setSize(this.computeSize());
                    app.graph.setDirtyCanvas(true, false);
                }
            }, { serialize: false });
        };

        // Restore _shownCount when a saved workflow is loaded
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            origOnConfigure?.apply(this, arguments);
            if (info.extra?._shownCount !== undefined) {
                // New workflow: use saved count
                this._shownCount = info.extra._shownCount;
            } else {
                // Migration: old workflow without _shownCount
                // Count how many slots have actual (non-"none") values
                this._shownCount = this._countActiveSlots();
                if (!this.extra) this.extra = {};
                this.extra._shownCount = this._shownCount;
            }
            this._applyVisibility();
            this._setupLivePreview();
            this.setSize(this.computeSize());
            // Show preview for already-saved values
            this._refreshPreview();
        };

        // Right-click menu: remove last image slot
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            origGetExtraMenuOptions?.apply(this, arguments);
            if ((this._shownCount ?? 0) > 0) {
                options.push({
                    content: "➖ Remove Last Image",
                    callback: () => {
                        this._shownCount--;
                        if (!this.extra) this.extra = {};
                        this.extra._shownCount = this._shownCount;
                        this._applyVisibility();
                        this.setSize(this.computeSize());
                        app.graph.setDirtyCanvas(true, false);
                    },
                });
            }
        };

        // Display all preview images returned by the backend
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            origOnExecuted?.apply(this, arguments);
            const node = this;
            if (!output?.images?.length) return;
            Promise.all(
                output.images.map(
                    src =>
                        new Promise(resolve => {
                            const img = new Image();
                            img.onload = () => resolve(img);
                            img.onerror = () => resolve(null);
                            img.src = `/view?filename=${encodeURIComponent(src.filename)}&type=${src.type}&subfolder=${encodeURIComponent(src.subfolder ?? "")}&rand=${Math.random()}`;
                        })
                )
            ).then(imgs => {
                node.imgs = imgs.filter(Boolean);
                node.imageIndex = 0;
                node.setSizeForImage?.();
                app.graph.setDirtyCanvas(true, false);
            });
        };
    },
});
