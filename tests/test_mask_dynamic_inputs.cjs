const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const file = path.join(__dirname, "../web_version/v2/mask_and_mask_dynamic_inputs.js");
const source = fs.readFileSync(file, "utf8");
const importLine = 'import { app } from "../../scripts/app.js";';
assert.ok(source.startsWith(importLine), "Use the public app module, without requiring window.comfyAPI");
const names = ["Mask Or Mask", "Mask And Mask", "Mask Sub Mask", "Mask Concat", "Temporal Mask Or", "Temporal Mask And"]
    .map(name => `${name} (endman100)`);

async function fixture(name = names[0]) {
    let extension;
    const frames = [];
    const app = {configuringGraph: false, registerExtension(value) { extension = value; }};
    // No window.comfyAPI: the app module import works independently of that global.
    vm.runInNewContext(source.replace(importLine, "const app = globalThis.__app;"), {
        __app: app, LiteGraph: {INPUT: 1}, requestAnimationFrame(callback) { frames.push(callback); },
    }, {filename: file});
    class Node {
        constructor() {
            this.inputs = ["mask1", "mask2"].map(name => ({name, type: "MASK", link: null}));
            this.calls = [];
            this.graph = {setDirtyCanvas() {}};
        }
        addInput(name, type) {
            this.inputs.push({name, type, link: null});
            this.onConnectionsChange(1, this.inputs.length - 1, false);
        }
        removeInput(index) {
            this.inputs.splice(index, 1);
            this.onConnectionsChange(1, index, false);
        }
        computeSize() { return [200, this.inputs.length * 20]; }
        setSize(size) { this.size = size; }
        onNodeCreated(...args) { this.calls.push(["created", ...args]); return "created-result"; }
        onConfigure(data) {
            this.calls.push(["configure", data]);
            if (data?.inputs) this.inputs = data.inputs.map(input => ({...input}));
            return "configure-result";
        }
        onConnectionsChange(...args) { this.calls.push(["connection", ...args]); return "connection-result"; }
    }
    const originalCreated = Node.prototype.onNodeCreated;
    await extension.beforeRegisterNodeDef(Node, {name});
    const node = new Node();
    function frame() { frames.splice(0).forEach(callback => callback()); }
    function flush() {
        let count = 0;
        while (frames.length) {
            assert.ok(++count < 20, "scheduled stabilization must converge");
            frame();
        }
    }
    function connect(name, link) {
        const index = node.inputs.findIndex(input => input.name === name);
        assert.notEqual(index, -1, name);
        node.inputs[index].link = link;
        return node.onConnectionsChange(1, index, link != null);
    }
    return {app, node, frames, frame, flush, connect, originalCreated, Node};
}

const slots = node => node.inputs.filter(input => input.type === "MASK").map(input => input.name);

test("六種節點皆從 mask1 開始，連接後可擴充至 N 個輸入", async () => {
    for (const name of names) {
        const {node, connect, flush} = await fixture(name);
        assert.equal(node.onNodeCreated("argument"), "created-result");
        assert.deepEqual(slots(node), ["mask1"]);
        for (let i = 1; i <= 12; i++) {
            assert.equal(connect(`mask${i}`, i), "connection-result");
            flush();
            assert.equal(node.inputs.length, i + 1);
            assert.equal(node.inputs.at(-1).name, `mask${i + 1}`);
            assert.equal(node.inputs.at(-1).link, null);
        }
        assert.ok(node.calls.some(call => call[0] === "created" && call[1] === "argument"));
    }
});

test("disconnecting tail trims redundant empty slots, retaining one spare", async () => {
    const {node, connect, flush} = await fixture();
    node.onNodeCreated();
    for (let i = 1; i <= 3; i++) { connect(`mask${i}`, i); flush(); }
    connect("mask3", null); flush();
    assert.deepEqual(slots(node), ["mask1", "mask2", "mask3"]);
    connect("mask2", null); flush();
    assert.deepEqual(slots(node), ["mask1", "mask2"]);
    connect("mask1", null); flush();
    assert.deepEqual(slots(node), ["mask1"]);
});

test("middle gaps and Sub's disconnected mask1 never change names or links", async () => {
    const {node, connect, flush} = await fixture(names[2]);
    node.onNodeCreated();
    for (let i = 1; i <= 3; i++) { connect(`mask${i}`, i * 10); flush(); }
    connect("mask2", null); flush();
    assert.deepEqual(slots(node), ["mask1", "mask2", "mask3", "mask4"]);
    assert.equal(node.inputs[2].link, 30);
    connect("mask1", null); flush();
    assert.equal(node.inputs[0].name, "mask1");
    assert.equal(node.inputs[0].link, null);
    assert.equal(node.inputs[2].name, "mask3");
    assert.equal(node.inputs[2].link, 30);
});

test("old workflows restore mask1/mask2 and mask3+ before any trimming", async () => {
    for (const count of [2, 3, 10]) {
        const {app, node, frame, flush, connect} = await fixture();
        app.configuringGraph = true;
        node.onNodeCreated();
        const saved = {inputs: Array.from({length: count}, (_, index) => ({
            name: `mask${index + 1}`, type: "MASK", link: null,
        }))};
        assert.equal(node.onConfigure(saved), "configure-result");
        frame();
        assert.equal(node.inputs.length, count);
        for (let i = 1; i <= count; i++) connect(`mask${i}`, i - 1);
        frame();
        assert.equal(node.inputs.length, count);
        app.configuringGraph = false;
        flush();
        assert.deepEqual(node.inputs.slice(0, count).map(input => [input.name, input.link]),
            saved.inputs.map((input, index) => [input.name, index]));
        assert.equal(node.inputs.at(-1).name, `mask${count + 1}`);
    }
});

test("onConfigure defers cleanup even without the configuringGraph flag", async () => {
    const {node, flush} = await fixture();
    node.onNodeCreated();
    node.onConfigure({inputs: ["mask1", "mask2", "mask3"].map(name => ({name, type: "MASK", link: null}))});
    assert.equal(node.inputs.length, 3);
    node.inputs[0].link = 101;
    node.inputs[1].link = 102;
    node.inputs[2].link = 103;
    flush();
    assert.deepEqual(node.inputs.map(input => input.link), [101, 102, 103, null]);
});

test("sparse saved names remain stable with one trailing spare", async () => {
    const {node, flush} = await fixture();
    node.onConfigure({inputs: [
        {name: "mask1", type: "MASK", link: 1},
        {name: "mask2", type: "MASK", link: null},
        {name: "mask10", type: "MASK", link: 10},
        {name: "mask11", type: "MASK", link: null},
        {name: "mask12", type: "MASK", link: null},
    ]});
    flush();
    assert.deepEqual(slots(node), ["mask1", "mask2", "mask10", "mask11"]);
    assert.equal(node.inputs[2].link, 10);
});

test("same-frame disconnect and reconnect does not remove a live input", async () => {
    const {node, connect, frames, flush} = await fixture();
    node.onNodeCreated();
    connect("mask1", 1); flush();
    connect("mask2", 2); flush();
    connect("mask2", null);
    connect("mask2", 22);
    assert.equal(frames.length, 1, "connection changes should coalesce");
    flush();
    assert.deepEqual(node.inputs.map(input => input.link), [1, 22, null]);
});

test("add/remove callbacks cannot re-enter stabilization or leave the guard set", async () => {
    const {node, frames, flush} = await fixture();
    node.inputs.push(...["mask3", "mask4", "mask5"].map(name => ({name, type: "MASK", link: null})));
    node.onNodeCreated();
    assert.deepEqual(slots(node), ["mask1"]);
    assert.equal(node.__maskHandleStabilizing, false);
    assert.equal(frames.length, 0);
    node.onConfigure({inputs: []}); flush();
    assert.deepEqual(slots(node), ["mask1"]);
    assert.equal(node.__maskHandleStabilizing, false);
    assert.equal(node.__maskHandleScheduled, false);
});

test("non-mask inputs survive tail cleanup and keep their identities", async () => {
    const {node, flush} = await fixture();
    const other = {name: "other", type: "IMAGE", link: 77};
    node.inputs.splice(1, 0, other);
    node.onNodeCreated(); flush();
    assert.strictEqual(node.inputs[1], other);
    assert.equal(other.link, 77);
    assert.deepEqual(slots(node), ["mask1"]);
});

test("output connection events leave inputs untouched", async () => {
    const {node, frames} = await fixture();
    node.onNodeCreated();
    assert.equal(node.onConnectionsChange(2, 0, true), "connection-result");
    assert.equal(frames.length, 0);
    assert.deepEqual(slots(node), ["mask1"]);
});

test("unrelated nodes are not wrapped", async () => {
    const {Node, originalCreated} = await fixture("logic repeatMask");
    assert.strictEqual(Node.prototype.onNodeCreated, originalCreated);
});

test("影片形態學節點維持單一輸入，不套用動態接頭", async () => {
    const {Node, originalCreated} = await fixture("Temporal Mask Morphology (endman100)");
    assert.strictEqual(Node.prototype.onNodeCreated, originalCreated);
});
