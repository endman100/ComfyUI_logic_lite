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

新增 `video/mask` I/O（獨立實作，無須安裝 VHS）：

- `Load Video With Mask (Upload) (endman100)`：上傳影片或多選 PNG／EXR 影格，亦可在 `video` 填入 input 相對路徑、絕對路徑、序列資料夾、首格或 `%08d`／`####`／`*` pattern。序列依數字排序，首格路徑由該格載入至結尾；同一序列須有相同格式、尺寸、位元深度與通道。PNG／EXR 使用 OpenCV `IMREAD_UNCHANGED` 逐檔讀取，再以 NumPy 批次 stack、BGR(A)→RGB(A) 與數值轉換，沒有 Python 像素迴圈，也不經 ffmpeg concat。載入前設定 `OPENCV_IO_ENABLE_OPENEXR=1`；缺少 OpenCV 或 EXR codec 時明確報錯，不降精度回退。`sequence_fps=24` 指定序列影格率。`force_rate=0` 保持來源 fps；若指定，先重採樣（序列以影格中心時間取最近鄰，不插值像素），再跳過 `skip_first_frames=0`、每 `select_every_nth=1` 格選一格，最後套用 `frame_load_cap=0`（不限）。輸出 IMAGE、MASK、frame_count、抽幀後 fps 與 JSON video_info。`custom_width/custom_height=0` 保持尺寸，只設一邊時維持比例，兩邊都設則縮放，不裁切；縮放會改變像素。`missing_alpha=ones` 於無 alpha 時輸出全白，可改 `error`。
- `Video Combine With Mask (endman100)`：輸入 `images`（RGB／RGBA）與選填 `mask`，`frame_rate=24`、`loop_count=0`（額外重播次數）、`filename_prefix=mask_video`、`save_output=true`。`[mask]` 標示 ProRes 4444、4444 XQ、VP9 alpha、PNG16 RGBA 序列、EXR float 序列；`[no mask]` 為 H.264。選到 H.264 且有遮罩時，`mask_mode=error` 可改為 `separate_png`（另存）或 `ignore`。`save_mask_sequence=false` 可開啟，無論主格式是否支援 alpha 都另存 PNG16 RGBA 遮罩序列（RGB 與 A 皆為遮罩，可直接重載 MASK）。PNG16 保留 uint16 精度，任意 float 仍會量化；精確 float 封存請用 EXR float。`bit_depth=auto` 依格式要求 ProRes **12**、VP9／H.264 8、PNG 16、EXR float；ProRes 可明確選 `12`，EXR 另可選 half，其餘不相容深度會報錯。`mask` 寬高須與影格相同，單格遮罩套用至每一格，其餘須與影片格數相同；`mask` 優先於 RGBA 的 A。

遮罩極性請分清楚：**本套件兩個 I/O 節點：MASK = alpha = 前景白色／不透明，不反相；VHS FFmpeg 重載：MASK = 1−alpha；核心「Join Image with Alpha」：A = 1−mask**。將本套件 MASK 接到該核心 Join 節點前，須先反相。

RGBA 影片走 16-bit pipe，PNG16 不經 Pillow 量化；OpenCV EXR32 保留 float32、負值與 HDR 範圍，不套用色彩轉換；支援一般單層 scanline EXR，不支援 tiled／deep／multipart。整數格式限制至 0..1。ProRes 4444／XQ 使用 `rgba64le → -c:v prores_ks -profile:v 4444/4444xq -pix_fmt yuva444p12le -alpha_bits 16`，**不強制 BT.709 scale／range 轉換或色彩標籤**；不會先降成 RGBA8。`bit_depth=12` 是要求值，不保證編碼器保留完整 12-bit：本機 ffmpeg 7.1.5 的 `prores_ks` 會警告並自動改用 `yuva444p10le`，解碼顯示 `yuva444p12le` 也不能證明全程 12-bit 精度。3 格 32×24 合成樣本的兩種 profile 往返最大差皆為 RGB 約 0.00170711、alpha 約 0.00122583（約 80/65535），均非數值無損；XQ 名稱也不代表已做 HDR 色彩管理。VP9 alpha 為 8-bit、4:2:0，讀取時強制 libvpx 解碼；VP9／H.264 需要偶數寬高。這些節點只處理影像，未保留音訊。

輸出在 ComfyUI output（`save_output=false` 則 temp）下的唯一資料夾，不覆寫既有檔案；含主檔／序列、fps 與路徑的 `manifest.json`，並回傳 JSON 檔案路徑。MP4／WebM 提供瀏覽器預覽，其餘提供縮圖及路徑；僅縮圖是 8-bit，不影響母檔。ffmpeg 依序尋找 `LOGIC_LITE_FFMPEG` 明確覆寫 → 選填 imageio-ffmpeg → `VHS_FORCE_FFMPEG_PATH`／已載入 VHS 的路徑 → PATH；不匯入 VHS。ffprobe 使用 `LOGIC_LITE_FFPROBE` → ffmpeg 同目錄 → PATH。需有 CPU `prores_ks`、`libvpx-vp9`、`libx264`、`png`、`exr` 編碼器；缺少時顯示 ffmpeg 錯誤。編碼以有界批次串流到 pipe；載入仍需容納最終 IMAGE／MASK 批次，可用 cap 限制記憶體。

授權參考：本儲存庫未找到授權檔；VideoHelperSuite 為 GPLv3。本次僅參考其功能與介面，未複製 GPL Python／JS 程式碼。I/O 測試可併入既有 CPU runner；前端測試用 `node --test tests/*.cjs`。

本機既有 venv 的 OpenCV 5.0.0 未編入 OpenEXR；完整測試使用 `/tmp/logic_lite_opencv411` 的隔離 `opencv-python-headless==4.11.0.86`（含 OpenEXR），未修改既有 venv。重跑時在上述 CPU runner 命令前加 `PYTHONPATH=/tmp/logic_lite_opencv411`；部署環境也須使用編入 OpenEXR 的 OpenCV，僅設定環境變數無法補上缺少的 codec。
