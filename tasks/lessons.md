# 教訓

## 2026-08-13

- 本機 96GB DDR4 只能在 2133 MT/s 穩定運作；不可建議或自動啟用 XMP／記憶體超頻。效能調校必須以 DDR4-2133 的實際頻寬為硬限制。
- 不把模型卡容量、第三方 TPS 或品質百分比當成實證；固定 revision 後驗證實檔大小，TPS 必須比對 context、硬體、量化、輸出長度與原始 log。
- DFlash 必須以 acceptance 與端到端 decode A/B 決定；CPU/RAM MoE 下 verification 可能放大 expert weight traffic，不能預設一定加速。
- llama.cpp 的 Laguna experts 是 per-layer packed tensor；粗粒度 `--n-cpu-moe`／tensor split 不能等同 hot-expert placement。
- 256K hybrid attention 後端必須驗證 KV allocator 是否真的對 SWA 層使用 ring buffer，不能只看 attention kernel 有 window 參數。
- SSE benchmark 不能把連線中斷當成功；沒有 `[DONE]` 或 `finish_reason` 必須回傳結構化失敗。
- benchmark 工具顯示的模型 context 不等於實際 allocation；必須從 runtime log 證明 `n_ctx` 與 KV 容量，避免把 512 context 誤報成 256K。
- hot-expert cache hit 高不代表一定達標；仍須用足夠暖機與固定輸出長度量測端到端低分位吞吐。
- Q4 KV 能釋放 VRAM並提高 expert cache hit，但不能把容量收益直接等同吞吐或品質；本機實測約快 30%，仍離 20 tok/s 很遠。
- Windows Ninja build 必須自行載入 MSVC INCLUDE/LIB；可用 vswhere 找到 vcvars64，不能假設 shell 是 Developer Prompt。
- 有限 A/B 低於目標只能淘汰該組設定，不能推論整個 backend 已到極限；宣告硬體 roofline 前，必須先完成高價值參數搜尋、逐階段 profiling 與 Amdahl 上限分析。
- 使用者要求持續逼近 20 tok/s 時，不能因 CLI 參數搜尋收斂就停止；必須繼續檢查未利用的框架級並行性，特別是同層雙 GPU 與 CPU expert lane 是否真正重疊。
- 把最後 1–2 層 MoE 留在 GPU（`n_cpu_moe=47/46`）會與 12GiB cache 搶 VRAM，本機實測掉到 11.2–11.6 tok/s。
- 同層 expert split 讓兩張 GPU 都能對同一層 Top-K 做 MMV，但不保證接近 2 倍：本機 pair_both 約 49/51，p5 只從 14.12 升到 14.59。collect 仍逐卡同步，attention 仍序列。

## 2026-08-14

- 逐 lane `cudaStreamSynchronize` 是真瓶頸，但 prefetch D2H 後 collect-wait 只剩約 2 ms/token；更大的一刀是把 down MMV 接在 fused swiglu 後面，dispatch 次數減半，p5 從 14.59 升到 17.99（`-v`）／19.32（INFO）。
- llama-bench `-v` 會把每 token 的 `CUDA Graph id reused` DEBUG 灌進 log，單獨吃掉約 1.4 tok/s。核對 `n_ctx` 用 INFO 即可，不要用 DEBUG 當正式數字。
- 長時間雙卡 CUDA 後 RTX 3060 可能 `GPU is lost`；`nvidia-smi` 寫明要重開機。單卡數字不能拿來取代雙卡 20 tok/s 驗收。
- 3060 故障後槽位可能變成 3070 Ti 8GB。8GB 卡不能當 12GB MoE-cache 驗收替代；換卡前暫停雙卡 10x，不要在錯誤硬體上硬跑。
- Windows 清 `.cache` 時 cargo 套件可能有超長路徑；`rd /s /q` 失敗後改用 `\\?\` 前綴再刪。
- 20 tok/s 路徑只需要 APEX i-compact。subcompact（Pulsar）與 IQ4_XS（品質候選）可刪；`config/models.json` 當重抓目錄，不要為了以後可能用到就佔一百多 GB。

## 2026-08-21

- 3060 無法修復、槽位變成 8GB 時應**封存倉庫**，不要在錯誤硬體上續跑。解封條件寫進 `tasks/todo.md`，不要另留一份 `handoff.md` 當第二個真相來源。

