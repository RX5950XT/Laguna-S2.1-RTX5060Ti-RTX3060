# Laguna-S2.1-RTX5060Ti-RTX3060 todo
最後更新：2026-11-19

## 現況（交接）：封存（2026-08-21）
- 2026-08-14 RTX 3060 12GB `GPU is lost`；重開機後為 GPU0 RTX 5060 Ti 16GB + GPU1 RTX 3070 Ti **8GB**。8GB 不能當 12GB MoE-cache 雙卡驗收，要換 **≥12GB** 第二張卡才解封。
- 封存期間：**不要**跑雙卡 10x、不要再掃 CLI、不要在 8GB 卡上硬測 20 tok/s。
- 暫停前最快雙卡：fused-down、INFO-only、無 STATS，r3 `19.506 / 19.6213 / 19.3222`，p5 **19.32**（帶 `-v` 為 17.99）。離 20 約 3%（約 2 ms/token）。**10x 未跑。**
- 磁碟只留 APEX i-compact（50.64GB）；subcompact、IQ4_XS 已刪，manifest 在 `config/models.json`，需要時用 `scripts/download-model.ps1` 重抓。官方 Q4_K_M 實檔 96.03GB（非模型卡寫的 68GB），未下載。

## 硬限制
- 目標硬體：RTX 5060 Ti 16GB + 12GB 級第二張卡、Ryzen 7 5700X、96GB DDR4-2133；禁止 XMP／記憶體超頻，不以 PBO 作驗收條件。
- context 固定 `262144`；目標 decode p5 `>=20 tok/s`、品質損失 `<5%`。
- 模型、cache、tmp、build、benchmark 大型產物只能在專案內；服務只聽 `127.0.0.1`。

## 關鍵版本
- Poolside llama.cpp Laguna fork `06f8cebd7fe728687be3d19f8bdedb70d75883af`，Windows CUDA 13.3、`sm_86 + sm_120a`。
- Pulsar 上游 `c7245425dbc46cd8b310370a71d05fa30a9d9b34`（Windows port，含 Laguna SWA ring）。
- leloch MoE-cache `e3096b046bb809f7f80bc47801f6579aed1cbc60` + `patches/moe-cache/0001`–`0004`（context、split、fused down、CUDA graph）。`test-moe-cache`（split／collect／fused-down，含 `graph-replays>0`）通過。

## 解封後接續
解封條件：`nvidia-smi` 為 GPU0 RTX 5060 Ti 16GB，**GPU1 ≥12GB 且健康**。若仍是 8GB／lost／只剩一張卡：**停止**。

Winner（原樣重跑，不要再轉旋鈕）：

| 項目 | 值 |
|---|---|
| 模型 | `models/apex_compact/Laguna-S-2.1-APEX-i-compact.gguf`（APEX i-compact，不是官方 Q4_K_M） |
| context | **262144**（禁止縮短 context／輸出） |
| KV | K 與 V 皆 **Q4_0**（`-ctk q4_0 -ctv q4_0`） |
| MoE-cache | 12288 MiB，reserve 2048 MiB |
| threads / ubatch / n_cpu_moe | 4 / 160 / 48 |
| load | mmap |
| split | `GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS=1`（fused-down + CUDA graph） |

```powershell
.\scripts\tune-moe-cache.ps1 -Stage confirm -Case fused-down
```
用 **pwsh** 直接跑；不要用 `powershell.exe`（UTF-8 繁中會壞）。官方 tune **不要加 `-v`**。

1. fused-down 256K r3，不得比 19.32／17.99 慢。
2. 兩次**獨立啟動** 10x（warmup 256、`-n 256`、`n_ctx=262144`），兩次 p5 都 ≥ 20；一次過一次掛不算。
3. 每次證據鏈：stderr `llama_context: n_ctx = 262144`、MoE-cache enabled、JSON `n_ctx=262144`、`samples_ts` 長度符合、fill/dispatch/collect fail=0。
4. JSON/log 寫進 `benchmarks/`，摘要更新本檔。

若 10x 仍 < 20：**禁止再掃 CLI**。下一刀有 profile 再切 scheduler 的 attn↔MoE host bounce，或把獨立 shared-expert CUDA 疊進剩餘 collect wait（約 1.6 ms/token，不是主因）。一次一刀；失敗或沒增益就完整撤回。不得留下比 split p5 14.59 更慢的 winner。

## 待辦（驗收門檻，皆封存中）
- [ ] 暖機後 10 次單 request decode p5 `>= 20 tok/s`（解封後依上方 SOP）。
- [ ] 服務回報 `n_ctx = 262144`，約 260K prompt 可完成至少 512-token generation。
- [ ] 30 分鐘 soak 無 crash、OOM 或持續熱降頻。
- [ ] Q4 KV 與候選量化相對官方 Q4：perplexity、coding pass@1、長文 retrieval 損失均 `< 5%`（需先下載 96.03GB 官方 Q4_K_M 建基準；另案）。
- [ ] `/v1/models`、chat completion、streaming、thinking、tool calling 全部通過（小 context 先驗 chat template、tool parser、UTF-8；另案）。
- [ ] 逐 token timestamp／資源遙測自動調校（另案）。

## 已知結論／刻意不處理
- 速度路徑（皆 256K + Q4_0 KV + i-compact）：baseline ~7.23 → Pulsar ~4.06 → cache p5 12.09 → layer-pin **14.12** → split **14.59** → fused-down `-v` **17.99** → INFO **19.32**。
- Q4_0 KV 把 KV VRAM 由約 6.45GiB 降到約 3.41GiB；Q8 KV 實際 6.42GiB。DDR4-2133 下 Pulsar 6 threads 最佳，llama.cpp 12 threads 反而較慢。
- 已淘汰：DFlash（acceptance 約 5–9% 且更慢）、`n_cpu_moe=47/46`（11.56／11.21，更慢）、CLI sequential-halving 搜尋（已收斂到 t=4／ub=160／ncmoe=48；`config/moe-cache-search.json` 保留給 confirm）。
- Pulsar 公開 22.4 tok/s 是 2K／32GB／9900X／IQ2 短輸出，不能當本機 256K 成果。
- 禁止：8GB 當 12GB 替代、縮 ctx／縮輸出湊 TPS、XMP／記憶體超頻、`-v` DEBUG 當正式數字、單卡數字當雙卡驗收。
