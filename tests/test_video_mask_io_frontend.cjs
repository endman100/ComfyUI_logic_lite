const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const file = path.join(__dirname, "../web_version/v2/video_mask_io.js");
const source = fs.readFileSync(file, "utf8")
    .replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__app;")
    .replace('import { api } from "../../scripts/api.js";', "const api = globalThis.__api;");
const LOAD_ID = "Load Video With Mask (Upload) (endman100)";
const SAVE_ID = "Video Combine With Mask (endman100)";
const PRORES = "[mask] ProRes 4444 (.mov)";
const H264 = "[no mask] H.264 (.mp4)";

async function fixture(name, status = 200) {
    let extension;
    const elements = [];
    const uploads = [];
    const errors = [];
    class Canvas {
        getNodeMenuOptions(node) {
            const options = [{content: "Existing item"}];
            const extra = node.getExtraMenuOptions(this, options);
            return Array.isArray(extra) ? [...extra, ...options] : options;
        }
    }
    const app = {
        registerExtension(value) { extension = value; },
        canvas: new Canvas(),
        graph: {setDirtyCanvas() {}},
        extensionManager: {toast: {add(error) { errors.push(error); }}},
    };
    const api = {
        apiURL(value) { return `/prefix${value}`; },
        async fetchApi(url, options) {
            uploads.push({url, options});
            return {ok: status === 200, status, async json() {
                return {name: options.body.values.get("image").name, subfolder: options.body.values.get("subfolder")};
            }};
        },
    };
    const document = {
        body: {append() {}},
        createElement(tag) {
            const listeners = new Map();
            const element = {tag, style: {}, children: [], attributes: {}, removed: false,
                append(...values) { this.children.push(...values); },
                replaceChildren() { this.children = []; },
                addEventListener(type, callback) { listeners.set(type, callback); },
                emit(type) { listeners.get(type)?.(); },
                setAttribute(key, value) { this.attributes[key] = value; },
                get valueAsNumber() { return Number(this.value); },
                click() { this.clicked = true; }, remove() { this.removed = true; },
                pause() { this.paused = true; }, removeAttribute(key) { delete this[key]; }, load() {},
                async play() { this.paused = false; },
            };
            elements.push(element);
            return element;
        },
    };
    class FormData {
        constructor() { this.values = new Map(); }
        append(key, value) { this.values.set(key, value); }
    }
    vm.runInNewContext(source, {__app: app, __api: api, document, FormData, LGraphCanvas: Canvas,
        crypto: {randomUUID: () => "test-group"}, URLSearchParams, console: {error() {}}}, {filename: file});
    extension.setup?.();
    async function makeNode(name, installEarlierHook) {
        const combineWidgets = () => [
            {name: "format", value: PRORES},
            {name: "mask_mode", type: "combo", value: "error", options: {serialize: true}},
        ];
        class Node {
            constructor() {
                this.widgets = name === SAVE_ID ? combineWidgets() : [{name: "video", value: "original.mov"}];
                this.dom = [];
                this.size = [320, 300];
                this.graph = app.graph;
                this.canvas = app.canvas;
            }
            onNodeCreated() { return "created"; }
            onConfigure({format, maskMode, recreateWidgets}) {
                if (recreateWidgets) this.widgets = combineWidgets();
                this.widgets[0].value = format;
                this.widgets[1].value = maskMode;
                return "configured";
            }
            onWidgetChanged() { return "changed"; }
            onExecuted() { return "executed"; }
            onRemoved() { return "removed"; }
            getExtraMenuOptions() { return "menu"; }
            addWidget(type, name, value, callback, options) { this.widgets.push({type, name, value, callback, options}); }
            addDOMWidget(name, type, container, options) {
                const widget = {name, container, options};
                this.dom.push(widget);
                return widget;
            }
            computeSize() { return [320, 400]; }
            setSize(value) { this.size = value; }
            setDirtyCanvas() {}
        }
        const original = Node.prototype.onNodeCreated;
        installEarlierHook?.(Node);
        await extension.beforeRegisterNodeDef(Node, {name});
        return {node: new Node(), Node, original};
    }
    return {...await makeNode(name), makeNode, elements, uploads, errors};
}

test("單檔上傳使用 ComfyUI 端點並保留伺服器子資料夾", async () => {
    const {node, elements, uploads} = await fixture(LOAD_ID);
    assert.equal(node.onNodeCreated(), "created");
    const button = node.widgets.at(-1);
    assert.equal(button.options.serialize, false);
    button.callback();
    const input = elements.at(-1);
    assert.equal(input.multiple, true);
    input.files = [{name: "alpha.mov"}];
    await input.onchange();
    assert.equal(uploads[0].url, "/upload/image");
    assert.equal(uploads[0].options.body.values.get("overwrite"), "false");
    assert.equal(node.widgets[0].value, "logic_lite_test-group/alpha.mov");
    assert.equal(input.removed, true);
});

test("多張高精度影格上傳至同一唯一序列資料夾", async () => {
    const {node, elements, uploads} = await fixture(LOAD_ID);
    node.onNodeCreated(); node.widgets.at(-1).callback();
    const input = elements.at(-1);
    input.files = [{name: "0001.exr"}, {name: "0002.exr"}];
    await input.onchange();
    assert.equal(uploads.length, 2);
    assert.equal(node.widgets[0].value, "logic_lite_test-group");
});

test("上傳失敗或多部影片不會改寫已選路徑", async () => {
    for (const [status, files] of [[500, [{name: "a.mov"}]], [200, [{name: "a.mov"}, {name: "b.mov"}]]]) {
        const {node, elements, errors} = await fixture(LOAD_ID, status);
        node.onNodeCreated(); node.widgets.at(-1).callback();
        const input = elements.at(-1);
        input.files = files;
        await input.onchange();
        assert.equal(node.widgets[0].value, "original.mov");
        assert.equal(errors.length, 1);
        assert.equal(input.removed, true);
    }
});

const media = {preview: "/logiclite/video-mask/test/preview", raw: "/logiclite/video-mask/test/raw"};
const output = {mask_video_previews: [media], mask_video_files: ["D:\\output\\shot\\manifest.json"]};

function menuItems(node) {
    const options = node.canvas.getNodeMenuOptions(node);
    assert.equal(options.at(-1).content, "Existing item");
    return Array.from(options).filter(Boolean).slice(0, -1);
}

test("both preview menus survive chained third-party hooks and a wrapped node receiver", async () => {
    const {makeNode} = await fixture(LOAD_ID);
    for (const type of [LOAD_ID, SAVE_ID]) {
        const {node, Node} = await makeNode(type, Node => {
            Node.prototype.getExtraMenuOptions = function (_canvas, options) {
                options.unshift({content: "Earlier extension"});
                return [{content: "Earlier returned item"}];
            };
        });
        const previous = Node.prototype.getExtraMenuOptions;
        Node.prototype.getExtraMenuOptions = function (canvas, options) {
            const extra = previous.call(new Proxy(this, {}), canvas, options);
            return [...extra, {content: "Later extension"}];
        };
        node.onExecuted(output);
        const items = menuItems(new Proxy(node, {}));
        assert.deepEqual(Array.from(items, item => item.content), [
            "Pause preview", "Hide preview", "Sync preview", "Download raw",
            "Earlier returned item", "Later extension", "Earlier extension",
        ]);
        items[0].callback();
        assert.equal(node.dom[0].container.children[0].paused, true);
    }
});

test("a node-level menu replacement still keeps the loaded preview actions", async () => {
    for (const type of [LOAD_ID, SAVE_ID]) {
        const {node, Node} = await fixture(type);
        Node.prototype.getExtraMenuOptions = () => [{content: "Replacement extension"}];
        node.onExecuted(output);
        assert.deepEqual(Array.from(menuItems(node), item => item.content), [
            "Pause preview", "Hide preview", "Sync preview", "Download raw", "Replacement extension",
        ]);
    }
});

test("both nodes play muted looping video without path text and release old media", async () => {
    for (const type of [LOAD_ID, SAVE_ID]) {
        const {node} = await fixture(type);
        assert.equal(node.onExecuted(output), "executed");
        assert.equal(node.dom.length, 1);
        const {container, options} = node.dom[0];
        assert.equal(options.serialize, false);
        assert.equal(container.children.length, 2);
        const video = container.children[0];
        assert.equal(video.tag, "video");
        assert.equal(video.src, `/prefix${media.preview}`);
        assert.ok(video.loop && video.autoplay && video.muted && video.playsInline);
        assert.equal(video.paused, false);
        assert.equal(container.textContent, undefined);
        assert.equal(video.textContent, undefined);
        assert.equal(node.title, undefined);
        node.onExecuted(output);
        assert.equal(video.paused, true);
        assert.equal(video.src, undefined);
        assert.equal(node.dom.length, 1);
        const latest = container.children[0];
        assert.equal(node.onRemoved(), "removed");
        assert.equal(latest.paused, true);
        assert.equal(latest.src, undefined);
        assert.deepEqual(menuItems(node), []);
    }
});

test("both nodes track playback and seek from their preview bar", async () => {
    for (const type of [LOAD_ID, SAVE_ID]) {
        const {node} = await fixture(type);
        node.onExecuted(output);
        const [video, bar] = node.dom[0].container.children;
        assert.equal(bar.tag, "input");
        assert.equal(bar.type, "range");
        assert.equal(bar.attributes["aria-label"], "Seek preview");
        assert.equal(bar.disabled, true);
        assert.equal(bar.value, "0");

        video.duration = 8;
        video.currentTime = 2;
        video.emit("durationchange");
        assert.equal(bar.disabled, false);
        assert.equal(bar.valueAsNumber, 0.25);

        bar.value = "0.75";
        bar.emit("input");
        assert.equal(video.currentTime, 6);
        assert.equal(video.paused, false);

        video.currentTime = 0;
        video.emit("seeked");
        assert.equal(bar.valueAsNumber, 0);
        video.currentTime = 1;
        video.emit("timeupdate");
        assert.equal(bar.valueAsNumber, 0.125);

        video.duration = Infinity;
        video.emit("durationchange");
        assert.equal(bar.disabled, true);
        bar.value = "0.5";
        bar.emit("input");
        assert.equal(video.currentTime, 1);
    }
});

test("Combine shows mask_mode only for no-mask formats and keeps its value and type", async () => {
    const {node} = await fixture(SAVE_ID);
    const [format, maskMode] = node.widgets;
    assert.equal(node.onNodeCreated(), "created");
    assert.equal(maskMode.hidden, true);
    maskMode.value = "separate_png";

    for (const alphaFormat of [PRORES, "[mask] ProRes 4444 XQ (.mov)", "[mask] VP9 alpha (.webm)",
        "[mask] PNG16 RGBA sequence", "[mask] EXR float sequence"]) {
        let previous = format.value;
        format.value = H264;
        assert.equal(node.onWidgetChanged("format", H264, previous, format), "changed");
        assert.equal(maskMode.hidden, false);
        previous = format.value;
        format.value = alphaFormat;
        node.onWidgetChanged("format", alphaFormat, previous, format);
        assert.equal(maskMode.hidden, true);
    }
    assert.equal(maskMode.value, "separate_png");
    assert.equal(maskMode.type, "combo");
    assert.equal(maskMode.options.serialize, true);
});

test("Combine reapplies mask_mode visibility after restoring values and recreating widgets", async () => {
    const {node} = await fixture(SAVE_ID);
    node.onNodeCreated();
    assert.equal(node.onConfigure({format: H264, maskMode: "ignore"}), "configured");
    assert.equal(node.widgets[1].hidden, false);
    assert.equal(node.widgets[1].value, "ignore");
    const oldMaskMode = node.widgets[1];
    node.onConfigure({format: PRORES, maskMode: "ignore", recreateWidgets: true});
    assert.notEqual(node.widgets[1], oldMaskMode);
    assert.equal(node.widgets[1].hidden, true);
    assert.equal(node.widgets[1].value, "ignore");
    node.onConfigure({format: H264, maskMode: "ignore"});
    assert.equal(node.widgets[1].hidden, false);
    assert.equal(node.widgets[1].value, "ignore");
});

test("legacy raw paths alone create no widget or menu", async () => {
    const {node} = await fixture(LOAD_ID);
    node.onExecuted({mask_video_files: ["/output/<frame>.exr"]});
    assert.equal(node.dom.length, 0);
    assert.deepEqual(menuItems(node), []);
});

test("both node menus pause, hide, sync the current graph and download raw", async () => {
    const {node: load, makeNode, elements} = await fixture(LOAD_ID);
    const {node: save} = await makeNode(SAVE_ID);
    const {node: otherGraph} = await makeNode(SAVE_ID);
    otherGraph.graph = {};
    for (const node of [load, save, otherGraph]) node.onExecuted(output);
    const video = node => node.dom[0].container.children[0];
    const action = (node, label) => menuItems(node).find(item => item.content === label).callback();
    for (const node of [load, save]) {
        assert.deepEqual(menuItems(node).map(item => item.content), ["Pause preview", "Hide preview", "Sync preview", "Download raw"]);
        action(node, "Pause preview");
        assert.equal(video(node).paused, true);
        assert.equal(video(node).autoplay, false);
        action(node, "Hide preview");
        assert.equal(node.dom[0].hidden, true);
        assert.equal(node.dom[0].container.hidden, true);
        action(node, "Show preview");
        assert.equal(node.dom[0].hidden, false);
        assert.equal(node.dom[0].container.hidden, false);
        assert.equal(video(node).paused, true);
        action(node, "Resume preview");
        assert.equal(video(node).paused, false);
        video(load).currentTime = 4;
        video(save).currentTime = 6;
        video(otherGraph).currentTime = 7;
        action(node, "Sync preview");
        assert.equal(video(load).currentTime, 0);
        assert.equal(video(save).currentTime, 0);
        assert.equal(video(otherGraph).currentTime, 7);
        action(node, "Download raw");
        const link = elements.at(-1);
        assert.equal(link.tag, "a");
        assert.equal(link.href, `/prefix${media.raw}`);
        assert.equal(link.download, "");
        assert.ok(link.clicked && link.removed);
    }
});

test("each batch result has its own raw download when available", async () => {
    const {node, elements} = await fixture(SAVE_ID);
    node.onExecuted({mask_video_previews: [{...media, raw: null}]});
    assert.ok(!menuItems(node).some(item => item.content === "Download raw"));
    const second = {...media, raw: "/logiclite/video-mask/second/raw"};
    node.onExecuted({mask_video_previews: [media, second]});
    assert.deepEqual(node.dom[0].container.children.map(element => element.tag), ["video", "input", "video", "input"]);
    const {submenu} = menuItems(node).find(item => item.content === "Download raw");
    submenu.options[1].callback();
    assert.equal(elements.at(-1).href, `/prefix${second.raw}`);
});

test("既有影片與遮罩節點不被新擴充掛鉤", async () => {
    for (const name of ["Video Mask Or (endman100)", "Temporal Mask Or (endman100)", "VHS_LoadVideo"]) {
        const {Node, original} = await fixture(name);
        assert.equal(Node.prototype.onNodeCreated, original);
    }
});
