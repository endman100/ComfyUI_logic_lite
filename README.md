# ComfyUI_logic_lite

Includes all 17 mask nodes from [comfyui-mask-handle](https://github.com/endman100/comfyui-mask-handle): mask selection, OR/AND/subtraction, concatenation, inversion, repetition, morphology, borders, temporal smoothing, area filling, background blending, alpha masks, and region extraction. Original node IDs, class names, and display names are preserved for existing workflows, including dynamic mask inputs. These nodes use the dependencies already listed in `requirements.txt` and PyTorch provided by ComfyUI.

Or / And / Sub and Mask Concat accept 1..N inputs. New nodes show only `mask1`; connecting the last socket adds a spare. Disconnecting trims only spare tail sockets and preserves middle socket numbers and saved links. Python keeps `mask2` optional for older workflows; extra numbered inputs are validated at runtime. Tensors and lists of tensors normalize to `[B,H,W]`; empty batches and invalid shapes fail explicitly.

For logical operations, `merge_all=false` combines matching batch indices and requires identical B/H/W (no broadcasting). `merge_all=true` reduces all inputs and batches to `[1,H,W]`, allowing different B but requiring equal H/W. Sub subtracts the union of the remaining masks from the first: in true mode, flatten in socket order (`mask1`'s entire batch, then `mask2`, etc.) and compute `flat[0] - union(flat[1:])`, including subtraction within a single input's batch. A single input in false mode passes through after thresholding. Inversion is applied last. Corrected rules can change older results that relied on broadcasting or previous Sub reductions.

Mask Concat preserves soft values and joins 1..N inputs horizontally, requiring the same B and the same height after optional resizing; widths may differ. It returns the mask and RGB preview, without a `merge_all` switch.

CPU checks: `python -B tests/run_cpu_tests.py --comfy-root /path/to/ComfyUI` and `node tests/test_mask_dynamic_inputs.cjs`. The CPU runner uses real PyTorch and AST-loads the existing image tests' reference classes from that ComfyUI checkout with a minimal schema/output adapter; it does not start ComfyUI. Frontend tests simulate connection and workflow restoration events.

新增三個 `mask/video` 節點，輸出皆為原裝置上的 float32 二值 `[T,H,W]`，支援 `[H,W]`（視為單格）、`[T,H,W]`、`[T,H,W,1]` 與清單；清單中的張量沿時間串接。

- `Temporal Mask Or (endman100)`／`Temporal Mask And (endman100)`：支援動態 `mask1..N`，所有輸入須有相同 T/H/W 與裝置。先依 `threshold=0.5`（嚴格大於）二值化，再逐格取聯集／交集，無 `merge_all`。`radius=0` 跳過時間投票；大於 0 時，對前後各 r 格的合併結果取平均，邊界只計入實際影格，平均值須嚴格大於 `vote_threshold=0.5` 才輸出 1。
- `Temporal Mask Morphology (endman100)`：單一 `mask`，使用真正 3D 形態學，沒有時間平均投票參數。`method` 可選 `erosion`／`dilation`（預設）／`open`／`close`，`kernel_shape=rect`（僅提供精確矩形核心），`kernel_size=3`、`frame_kernel_size=1` 均須為正奇數；時間核心 1 僅處理每格空間，5 表示前後各 2 格。`iterations=1`，open／close 的次數分別套用於侵蝕與膨脹階段；T/H/W 邊界皆複製邊緣值填補。先以 `threshold=0.5` 二值化。例如 `[1,1,0,1,1]` 的五格全白／全黑影片，使用時間核心 3 的 close 可填補中央一格缺口。

三者皆於最後套用 `mask_inverse=False`；合併、時間投票與 3D 池化均以 PyTorch 向量化執行，不逐影格或逐像素處理。
