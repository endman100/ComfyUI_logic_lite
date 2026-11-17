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

async function fixture(name, status = 200) {
    let extension;
    const elements = [];
    const uploads = [];
    const errors = [];
    const app = {
        registerExtension(value) { extension = value; },
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
            const element = {tag, style: {}, children: [], removed: false,
                append(value) { this.children.push(value); },
                replaceChildren() { this.children = []; },
                querySelectorAll(tag) { return this.children.filter(child => child.tag === tag); },
                addEventListener() {}, click() {}, remove() { this.removed = true; },
                pause() { this.paused = true; }, removeAttribute(key) { delete this[key]; }, load() {},
            };
            elements.push(element);
            return element;
        },
    };
    class FormData {
        constructor() { this.values = new Map(); }
        append(key, value) { this.values.set(key, value); }
    }
    vm.runInNewContext(source, {__app: app, __api: api, document, FormData,
        crypto: {randomUUID: () => "test-group"}, URLSearchParams, console: {error() {}}}, {filename: file});
    class Node {
        constructor() { this.widgets = [{name: "video", value: "original.mov"}]; this.dom = []; }
        onNodeCreated() { return "created"; }
        onExecuted() { return "executed"; }
        onRemoved() { return "removed"; }
        addWidget(type, name, value, callback, options) { this.widgets.push({type, name, value, callback, options}); }
        addDOMWidget(name, type, container, options) { this.dom.push({name, container, options}); }
        computeSize() { return [320, 400]; }
        setSize(value) { this.size = value; }
    }
    const original = Node.prototype.onNodeCreated;
    await extension.beforeRegisterNodeDef(Node, {name});
    const node = new Node();
    return {node, Node, original, elements, uploads, errors};
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

test("預覽使用 /view 並於更新或移除時釋放影片", async () => {
    const {node} = await fixture(SAVE_ID);
    const output = {mask_videos: [{filename: "video.webm", subfolder: "shot one", type: "temp"}],
        mask_video_files: ["/output/shot/video.webm"]};
    assert.equal(node.onExecuted(output), "executed");
    assert.equal(node.dom.length, 1);
    const container = node.dom[0].container;
    const video = container.children[0];
    assert.ok(video.src.startsWith("/prefix/view?"));
    assert.ok(video.src.includes("subfolder=shot+one"));
    assert.equal(node.dom[0].options.serialize, false);
    node.onExecuted(output);
    assert.equal(video.paused, true);
    assert.equal(video.src, undefined);
    assert.equal(node.dom.length, 1);
    const latest = container.children[0];
    assert.equal(node.onRemoved(), "removed");
    assert.equal(latest.paused, true);
});

test("不支援瀏覽器播放的格式仍顯示純文字路徑，不注入 HTML", async () => {
    const {node} = await fixture(SAVE_ID);
    node.onExecuted({mask_video_files: ["/output/<frame>.exr"]});
    const paths = node.dom[0].container.children[0];
    assert.equal(paths.tag, "pre");
    assert.equal(paths.textContent, "/output/<frame>.exr");
});

test("既有影片與遮罩節點不被新擴充掛鉤", async () => {
    for (const name of ["Video Mask Or (endman100)", "Temporal Mask Or (endman100)", "VHS_LoadVideo"]) {
        const {Node, original} = await fixture(name);
        assert.equal(Node.prototype.onNodeCreated, original);
    }
});
