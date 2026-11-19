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

async function fixture(name, status = 200, deferUpload = false) {
    let extension;
    const elements = [];
    const uploads = [];
    const errors = [];
    const previewRequests = [];
    const timers = new Map();
    let nextTimer = 0;
    async function flushPreviews() {
        const pending = [...timers.values()];
        timers.clear();
        await Promise.all(pending.map(callback => callback()));
    }
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
            assert.equal(url, "/logiclite/video-mask/load-preview");
            previewRequests.push(JSON.parse(options.body));
            return {ok: true, async json() { return {mask_video_previews: [media]}; }};
        },
    };
    class XMLHttpRequest {
        upload = {};
        open(method, url) { Object.assign(this, {method, url}); }
        send(body) {
            uploads.push({url: this.url, options: {method: this.method, body}, xhr: this});
            this.status = status;
            this.response = {name: body.values.get("image").name, subfolder: body.values.get("subfolder")};
            if (!deferUpload) queueMicrotask(() => this.onload());
        }
    }
    const document = {
        body: {append() {}},
        createElement(tag) {
            const element = {tag, style: {}, children: [], removed: false, events: {},
                append(value) { this.children.push(value); },
                replaceChildren() { this.children = []; },
                addEventListener(type, callback) { this.events[type] = callback; },
                setAttribute(key, value) { this[key] = value; },
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
    vm.runInNewContext(source, {__app: app, __api: api, document, FormData, XMLHttpRequest, LGraphCanvas: Canvas,
        crypto: {randomUUID: () => "test-group"}, URLSearchParams, AbortController,
        setTimeout(callback) { timers.set(++nextTimer, callback); return nextTimer; },
        clearTimeout(timer) { timers.delete(timer); }, console: {error() {}}}, {filename: file});
    extension.setup?.();
    async function makeNode(name, installEarlierHook) {
        const combineWidgets = () => [
            {name: "format", value: PRORES},
            {name: "mask_mode", type: "combo", value: "error", options: {serialize: true}},
            {name: "crf", type: "number", value: 18, options: {serialize: true}},
        ];
        class Node {
            constructor() {
                this.widgets = name === SAVE_ID ? combineWidgets() : [
                    {name: "video", value: "original.mov", options: {values: ["", "original.mov"]}},
                    {name: "source_path", value: ""},
                ];
                this.dom = [];
                this.size = [320, 300];
                this.graph = app.graph;
                this.canvas = app.canvas;
            }
            onNodeCreated() { return "created"; }
            onConfigure({format, maskMode, recreateWidgets, video}) {
                if (name === LOAD_ID) {
                    this.widgets[0].value = video;
                    return "configured";
                }
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
            computeSize() { return [320, 400 + this.dom.reduce((height, widget) =>
                height + (widget.hidden ? 0 : widget.options.getMinHeight()), 0)]; }
            setSize(value) {
                this.size = value;
                this.dom.forEach(widget => widget.options.afterResize?.());
            }
            setDirtyCanvas() {}
        }
        const original = Node.prototype.onNodeCreated;
        installEarlierHook?.(Node);
        await extension.beforeRegisterNodeDef(Node, {name});
        return {node: new Node(), Node, original};
    }
    return {...await makeNode(name), makeNode, elements, uploads, errors, previewRequests, flushPreviews};
}

test("單檔上傳使用 ComfyUI 端點並加入下拉選單", async () => {
    const {node, elements, uploads, previewRequests, flushPreviews} = await fixture(LOAD_ID);
    assert.equal(node.onNodeCreated(), "created");
    node.widgets[1].value = "old-source.mov";
    const button = node.widgets.at(-1);
    assert.equal(button.options.serialize, false);
    button.callback();
    const input = elements.at(-1);
    assert.equal(input.multiple, true);
    input.files = [{name: "alpha.mov"}];
    await input.onchange();
    assert.equal(uploads[0].url, "/prefix/upload/image");
    assert.equal(uploads[0].options.body.values.get("overwrite"), "false");
    assert.equal(node.widgets[0].value, "logic_lite_test-group/alpha.mov");
    assert.ok(node.widgets[0].options.values.includes(node.widgets[0].value));
    assert.equal(node.widgets[1].value, "");
    await flushPreviews();
    assert.equal(previewRequests[0].video, "logic_lite_test-group/alpha.mov");
    assert.equal(node.dom.find(widget => widget.name === "mask_video_preview").container.children[0].tag, "video");
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
    assert.ok(node.widgets[0].options.values.includes(node.widgets[0].value));
});

test("拖放影片或同格式序列會上傳、選中並重新預覽", async () => {
    for (const [files, selected] of [
        [[{name: "clip.MOV"}], "logic_lite_test-group/clip.MOV"],
        [[{name: "0001.exr"}, {name: "0002.EXR"}], "logic_lite_test-group"],
    ]) {
        const {node, uploads, previewRequests, flushPreviews} = await fixture(LOAD_ID);
        node.onNodeCreated();
        assert.equal(node.onDragOver({dataTransfer: {types: ["Files"], files: []}}), true);
        let prevented = false;
        let stopped = false;
        assert.equal(node.onDragDrop({dataTransfer: {files},
            preventDefault() { prevented = true; }, stopPropagation() { stopped = true; }}), true);
        assert.ok(prevented && stopped);
        await new Promise(setImmediate);
        assert.equal(uploads.length, files.length);
        assert.ok(uploads.every(({url, options}) => url === "/prefix/upload/image" && options.body.values.get("type") === "input"));
        assert.equal(node.widgets[0].value, selected);
        assert.ok(node.widgets[0].options.values.includes(selected));
        await flushPreviews();
        assert.equal(previewRequests[0].video, selected);
    }
});

test("上傳按鈕和拖放共用節點原生進度，序列按檔案大小累計並在結束時清除", async () => {
    for (const drag of [false, true]) {
        const {node, elements, uploads} = await fixture(LOAD_ID, 200, true);
        node.onNodeCreated();
        const files = [{name: "one.png", size: 100}, {name: "two.png", size: 300}];
        if (drag) {
            node.onDragDrop({dataTransfer: {files}, preventDefault() {}, stopPropagation() {}});
        } else {
            node.widgets.at(-1).callback();
            const input = elements.at(-1);
            input.files = files;
            void input.onchange();
        }
        assert.equal(node.dom.length, 0);
        assert.ok(!elements.some(element => element.tag === "progress"));
        uploads[0].xhr.upload.onprogress({lengthComputable: true, loaded: 50, total: 100});
        assert.equal(node.progress, 0.125);
        uploads[0].xhr.onload();
        await new Promise(setImmediate);
        uploads[1].xhr.upload.onprogress({lengthComputable: true, loaded: 150, total: 300});
        assert.equal(node.progress, 0.625);
        uploads[1].xhr.onload();
        await new Promise(setImmediate);
        assert.equal(node.progress, undefined);
        assert.equal(node.dom.length, 0);
    }
});

test("拖放工作流、連結或混合序列仍交由原本的拖放處理", async () => {
    const {makeNode, uploads} = await fixture(LOAD_ID);
    const forwarded = [];
    const {node} = await makeNode(LOAD_ID, Node => {
        Node.prototype.onDragDrop = event => { forwarded.push(event); return false; };
    });
    node.onNodeCreated();
    for (const files of [[], [{name: "workflow.json"}], [{name: "01.png"}, {name: "02.exr"}]]) {
        const event = {dataTransfer: {files},
            preventDefault() { assert.fail("unsupported drop was cancelled"); },
            stopPropagation() { assert.fail("unsupported drop was consumed"); }};
        assert.equal(node.onDragOver(event), false);
        assert.equal(node.onDragDrop(event), false);
        assert.equal(forwarded.at(-1), event);
    }
    assert.equal(uploads.length, 0);
});

test("上傳失敗或多部影片不會改寫已選路徑", async () => {
    for (const [status, files] of [[500, [{name: "a.mov"}]], [200, [{name: "a.mov"}, {name: "b.mov"}]],
        [200, [{name: "a.png"}, {name: "b.exr"}]]]) {
        const {node, elements, errors, uploads} = await fixture(LOAD_ID, status, true);
        node.onNodeCreated(); node.widgets.at(-1).callback();
        const input = elements.at(-1);
        input.files = files;
        const upload = input.onchange();
        if (status === 500) {
            uploads[0].xhr.upload.onprogress({lengthComputable: true, loaded: 1, total: 2});
            assert.equal(node.progress, 0.5);
            uploads[0].xhr.onload();
        }
        await upload;
        assert.equal(node.progress, undefined);
        assert.equal(node.widgets[0].value, "original.mov");
        assert.equal(errors.length, 1);
        assert.equal(input.removed, true);
    }
});

test("還原舊來源及變更選幀參數會更新預覽", async () => {
    const {node, previewRequests, flushPreviews} = await fixture(LOAD_ID);
    assert.equal(node.onConfigure({video: "legacy/sequence"}), "configured");
    assert.ok(node.widgets[0].options.values.includes("legacy/sequence"));
    for (const [name, value] of Object.entries({frame_load_cap: 3, skip_first_frames: 2, select_every_nth: 2, force_rate: 12})) {
        node.widgets.push({name, value});
        node.onWidgetChanged(name);
    }
    await flushPreviews();
    assert.deepEqual(previewRequests, [{video: "legacy/sequence", source_path: "", frame_load_cap: 3,
        skip_first_frames: 2, select_every_nth: 2, force_rate: 12}]);
});

const media = {preview: "/logiclite/video-mask/test/preview", raw: "/logiclite/video-mask/test/raw"};
const output = {mask_video_previews: [media], mask_video_files: ["D:\\output\\shot\\manifest.json"]};
const previewActions = ["Pause preview", "Hide preview", "Sync preview", "Download raw"];
const fileActions = ["Open preview", "Save preview"];

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
            ...previewActions, ...fileActions,
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
            ...previewActions, ...fileActions, "Replacement extension",
        ]);
    }
});

test("預覽依實際比例配合節點寬度，並只在滑鼠移入時開啟聲音", async () => {
    for (const type of [LOAD_ID, SAVE_ID]) await checkAudioPreview(type);
});

async function checkAudioPreview(type) {
    const {node, elements} = await fixture(type);
    const source = {width: 200, height: 300, fps: 29.97, frames: 60, has_audio: true};
    node.onExecuted({mask_video_previews: [{...media, source}]});
    const {container, options} = node.dom[0];
    const video = container.children[0];
    Object.assign(video, {videoWidth: 200, videoHeight: 300});
    video.events.loadedmetadata();
    assert.equal(options.getMinHeight(), 470);
    assert.equal(options.getMaxHeight(), 470);
    assert.equal(node.size[1], 870);
    node.setSize([420, 400]);
    assert.equal(node.size[1], 1020);
    assert.equal(video.muted, true);
    video.events.pointerenter();
    assert.equal(video.muted, false);
    video.events.pointerleave();
    assert.equal(video.muted, true);
    const action = label => menuItems(node).find(item => item.content === label).callback();
    action("Mute preview");
    video.events.pointerenter();
    assert.equal(video.muted, true);
    action("Unmute preview");
    video.events.pointerenter();
    assert.equal(video.muted, false);
    action("Hide preview");
    assert.equal(video.muted, true);
    assert.ok(menuItems(node).some(item => item.disabled && item.content === "200×300 @ 29.97 fps · 60 frames"));
    action("Save preview");
    assert.equal(elements.at(-1).href, `/prefix${media.preview}`);
    assert.equal(elements.at(-1).download, "preview.webm");
}

test("both nodes play muted looping video without path text and release old media", async () => {
    for (const type of [LOAD_ID, SAVE_ID]) {
        const {node} = await fixture(type);
        assert.equal(node.onExecuted(output), "executed");
        assert.equal(node.dom.length, 1);
        const {container, options} = node.dom[0];
        assert.equal(options.serialize, false);
        assert.equal(options.hideOnZoom, false);
        assert.equal(container.children.length, 1);
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

test("Combine shows mask_mode only for no-mask formats and keeps its value and type", async () => {
    const {node} = await fixture(SAVE_ID);
    const [format, maskMode, crf] = node.widgets;
    assert.equal(node.onNodeCreated(), "created");
    assert.equal(maskMode.hidden, true);
    assert.equal(crf.hidden, true);
    crf.value = 24;
    maskMode.value = "separate_png";

    for (const alphaFormat of [PRORES, "[mask] ProRes 4444 XQ (.mov)", "[mask] VP9 alpha (.webm)",
        "[mask] PNG16 RGBA sequence", "[mask] EXR float sequence"]) {
        let previous = format.value;
        format.value = H264;
        assert.equal(node.onWidgetChanged("format", H264, previous, format), "changed");
        assert.equal(maskMode.hidden, false);
        assert.equal(crf.hidden, false);
        previous = format.value;
        format.value = alphaFormat;
        node.onWidgetChanged("format", alphaFormat, previous, format);
        assert.equal(maskMode.hidden, true);
        assert.equal(crf.hidden, true);
    }
    assert.equal(maskMode.value, "separate_png");
    assert.equal(maskMode.type, "combo");
    assert.equal(maskMode.options.serialize, true);
    assert.equal(crf.value, 24);
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
        assert.deepEqual(menuItems(node).map(item => item.content), [...previewActions, ...fileActions]);
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
    assert.deepEqual(node.dom[0].container.children.map(element => element.tag), ["video", "video"]);
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
