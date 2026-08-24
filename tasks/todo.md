# Laguna S 2.1 部署計畫

## 封存（2026-08-21）

本倉庫**封存**。RTX 3060 12GB 故障，重開機後槽位是 RTX 3070 Ti **8GB**，不能當 12GB MoE-cache 雙卡驗收。要換 **≥12GB** 的第二張卡才解封。

- 封存期間：**不要**跑雙卡 10x、不要再掃 CLI、不要在 8GB 卡上硬測 20 tok/s。
- 暫停前最快雙卡：**fused-down、INFO-only、無 STATS、p5 19.32 tok/s**（19.506 / 19.6213 / 19.3222，平均 19.483）。離 20 約 3%。帶 `-v` 為 p5 17.99。**10x 未跑。**
- 解封 SOP、winner 旋鈕與禁止事項見下方「解封後接續」。歷史階段與 review 仍保留。

## 解封後接續

解封條件：`nvidia-smi` 為 GPU0 RTX 5060 Ti 16GB，**GPU1 ≥12GB 且健康**。若仍是 8GB／lost／只剩一張卡：**停止**。

Winner（原樣重跑，不要再轉旋鈕）：

| 項目 | 值 |
|---|---|
| 模型 | `models/apex_compact/Laguna-S-2.1-APEX-i-compact.gguf`（50.64 GB，APEX i-compact，不是官方 Q4_K_M） |
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

依序：

1. fused-down 256K r3。不得比 19.32／17.99 慢。
2. 兩次**獨立啟動** 10x（warmup 256、`-n 256`、`n_ctx=262144`）。兩次 p5 都 ≥ 20。一次過一次掛不算。
3. 每次證據鏈：stderr `llama_context: n_ctx = 262144`、MoE-cache enabled、JSON `n_ctx=262144`、`samples_ts` 長度符合、fill/dispatch/collect fail=0。
4. JSON/log 寫進 `benchmarks/` 與 `CONTEXT.md`。

若 10x 仍 < 20：**禁止再掃 CLI**。下一刀有 profile 再切 scheduler 的 attn↔MoE host bounce，或把獨立 shared-expert CUDA 疊進剩餘 collect wait（collect-wait 約 1.6 ms/token，不是主因）。一次一刀；失敗或沒增益就完整撤回。不得留下比 split p5 14.59 更慢的 winner。

解封後仍不做（另案）：30 分鐘 soak、260K+512、thinking／tool／完整 `/v1`、Q4 KV <5% 品質、下載 96GB 官方 Q4、復活 DFlash、把 Pulsar 22.4 當成本機成果。

禁止：8GB 當 12GB 替代、縮 ctx／縮輸出湊 TPS、XMP／記憶體超頻、`-v` DEBUG 當正式數字、單卡數字當雙卡驗收。

速度路徑（皆 256K + Q4_0 KV + i-compact）：baseline ~7.23 → Pulsar ~4.06 → cache p5 12.09 → layer-pin **14.12** → split **14.59** → fused-down `-v` **17.99** → INFO **19.32**。已淘汰 `n_cpu_moe=47/46`、DFlash。

## 驗收門檻

- [ ] 服務實際回報 `n_ctx = 262144`，約 260K prompt 可完成至少 512-token generation。
- [ ] 暖機後 10 次單 request decode 的低分位吞吐（p5）`>= 20 tok/s`。**封存中。**
- [ ] 30 分鐘 soak 無 crash、OOM 或持續熱降頻。
- [ ] 相對官方 Q4 基準：perplexity、coding pass@1 與長文 retrieval 損失均 `< 5%`。
- [ ] `/v1/models`、chat completion、streaming、thinking、tool calling 全部通過。
- [x] 模型、cache、tmp、build 與 benchmark 大型產物都位於專案內。

## 階段 1：盤點與專案邊界

- [x] 建立 Git repo、`AGENTS.md`、`CLAUDE.md`、`CONTEXT.md` 與任務文件。
- [x] 強化 `.gitignore` 排除規則（排除大型模型、專案快取、建置目錄、第三方原始碼、benchmark 產物、敏感檔與測試快取）並完成 initial commit。
- [x] 確認 GPU、VRAM、RAM、driver、磁碟與現有 toolchain。
- [x] 鎖定 DDR4-2133 為穩定上限，排除 XMP／記憶體超頻。
- [x] 產生可重跑的硬體與路徑清冊。

## 階段 2：Runtime

- [x] 固定 Poolside `llama.cpp` Laguna fork commit `06f8cebd7fe728687be3d19f8bdedb70d75883af`。
- [x] 原生 Windows CUDA 建置 `sm_86` + `sm_120a`。
- [x] 驗證 Laguna、DFlash、Flash Attention、MoE offload 與雙 GPU flags。

## 階段 3：模型

- [x] 將所有 cache/temp 環境變數固定在專案內。
- [x] 下載並驗證 APEX i-compact、Unsloth IQ4_XS 與兩款 DFlash checksum/revision。
- [ ] 視品質基準需求下載官方 96.03GB Q4_K_M；不得沿用過時的 68GB 容量估計。**封存中。**
- [x] 盤點較緊湊候選量化；只有品質損失 `<5%` 才可採用。

## 階段 4：功能與基準

- [ ] 小 context 驗證 chat template、thinking、tool parser、UTF-8 與 streaming。**封存中（另案）。**
- [ ] 建立官方 Q4 的品質與效能基準。**封存中（另案）。**
- [ ] 建立自動調校與逐 token timestamp／資源遙測。**封存中。**

## 階段 5：256K 與效能搜尋

- [x] 固定 262144 context，比較雙 GPU 非對稱分工。
- [x] 搜尋 CPU MoE offload、threads、mmap 與 Q8/Q4 KV baseline。
- [x] A/B 測試兩款 DFlash；因低 acceptance 與明顯降速而停用。
- [x] 稽核 Pulsar hot-expert placement 與 KTransformers 可行性。
- [x] 實作支援 Laguna 256K SWA KV 的 Windows Pulsar per-expert 後端，並驗證另一套 llama.cpp MoE-cache。
- [ ] 淘汰未通過 `<5%` 品質門檻的低位元候選。**封存中（另案）。**

## 階段 5.1：組合拳極限優化實測 (KV Q4 + Hot Cache 擴展)

- [x] 實驗 A：Q4_0 KV 將 KV VRAM 由約 6.45GiB 降到約 3.41GiB。
- [x] 實驗 B：GPU0 cache 由 1.8GiB 增到 4.9GiB，hit 提升到 77.9%；GPU1 hit 79.2%。
- [x] 實驗 C：固定 256K、256-token warmup、三次 256-token decode 與 cache teardown 遙測。
- [ ] 實驗 D：評估 KV Cache Q4 對於品質與 PPL 損耗之影響是否 `< 5%`。**封存中（另案）。**
- [x] 實驗 E：Q4 p5 僅 `12.09 tok/s`，未達 15/20，依預設停止門檻不執行 30 分鐘假驗收。

## 階段 5.3：同層雙 GPU expert split

- [x] 確認 CLI 參數層已收斂：背景服務可部署 winner 為 Q4 KV、12GiB cache、reserve 2GiB、`t=4`、`ubatch=160`，乾淨 r3 為 `14.34 / 14.30 / 14.10`，p5 `14.12 tok/s`。
- [x] 確認 `n_cpu_moe=47/46` 把最後 1–2 層留在 GPU 會變慢（11.56 / 11.21），淘汰。
- [x] 將同層 Top-K expert 依 expert id 拆到兩張 GPU，`GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS=1`（預設開）。
- [x] `test-moe-cache` 新增 `cache-same-layer-split` 並全數通過。
- [x] 固定 256K r3：`14.82 / 15.44 / 14.59`，平均 `14.95`、p5 `14.59`；相對 14.12 僅約 `+3%`，未達 20。

## 階段 5.4：collect 重疊 + fused down

- [x] Collect 在 dispatch 後立刻 D2H，雙 lane 先全部送再一起 sync；`test-moe-cache` 新增 `cache-same-layer-split-collect`。
- [x] `fused_begin_ex` 把 down MMV 接在 fused swiglu 後面；CPU fusion 自動偵測 GLU→down。`cache-fused-down` 通過。
- [x] 256K r3（`-v`）：`18.0075 / 18.1243 / 17.9905`，p5 `17.99`。INFO-only：`19.506 / 19.6213 / 19.3222`，p5 `19.32`。`n_ctx=262144`，failure=0。
- [x] fused-down kernel 序列收成 CUDA graph；`cache-fused-down` 要求 `graph-replays>0`。
- [ ] p5 `>=20`。**封存**：3060 故障，槽位是 3070 Ti 8GB。換 ≥12GB 卡後依「解封後接續」跑 r3 與兩次 10x。

## 階段 5.2：MoE-cache 系統化逼近 20 tok/s

**已取代／取消。** CLI sequential-halving 搜尋已跑過並收斂到 t=4／ub=160／ncmoe=48；後續改走 5.3／5.4 backend。封存期間不要重開這輪搜尋。腳本與 `config/moe-cache-search.json` 仍保留供解封後 confirm。

- [x] 建立 `config/moe-cache-search.json` 與 `scripts/tune-moe-cache.ps1`。
- [x] 參數層收斂；正式路徑改 fused-down confirm，不再掃 ubatch／threads／n_cpu_moe=47/46。
- [ ] 速度達標後才執行 Q4 KV `<5%` 品質 gate、10 次 p5、260K+512、30 分鐘 soak 與 API 全功能驗收。**封存中。**

## 階段 6：最終驗收

- [ ] 260K long-context、10 次吞吐與 30 分鐘 soak。**封存中。**
- [ ] 品質差異報告與 OpenAI-compatible API 驗證。**封存中（另案）。**
- [x] 專案外大型檔案路徑稽核；本任務模型、cache、tmp、build 與結果均在專案內。
- [x] 完成 README、CONTEXT 與本檔 review。

## 階段 7：遠端版本控制 (GitHub)

- [x] 建立 GitHub 存放庫 `RX5950XT/Laguna-S2.1-RTX5060Ti-RTX3060`。
- [x] 設定遠端 `origin` 並推送到 `main` 分支。
- [x] 驗證遠端 repo、gitignore 排除項與同步狀態。

## Review

- 2026-08-13：完成初始規劃。容量可行；主要不確定性是 DDR4-2133 下的 MoE expert streaming 吞吐，必須以 DFlash 與 GPU/CPU 分工實測，不以估算宣稱達標。
- 2026-08-13：HF API/HEAD 證實官方 Q4_K_M 實檔為 96.03GB，而非模型卡仍標示的 68GB；已固定各候選 revision、size 與 SHA-256。IQ4_XS（57.57GB）先作品質優先可運行候選。
- 2026-08-13：完成原生 Windows CUDA `sm_86` + `sm_120a` runtime。APEX i-compact 在 256K、Q4 KV、38 層 CPU MoE、90/10 split、8 threads、no-mmap 達目前最佳約 7.23 tok/s；12 threads 較慢。
- 2026-08-13：DFlash 並非必然加速。APEX Q4 draft acceptance 約 9% 且降至約 2.46 tok/s；IQ4_XS + BF16 draft acceptance 約 5–7% 且降至約 1.26 tok/s，均已淘汰。
- 2026-08-13：Pulsar 的 per-expert 架構有效，但公開 22.4 tok/s 是 2K context、32GB VRAM、9900X、特製 IQ2 的短輸出結果。未修改的上游對 Laguna 256K 配置 full-layer KV、只支援 Linux，Laguna DFlash 亦不相容，不能直接視為本機部署成果。
- 2026-08-13：已將 Pulsar port 到 Windows，完成 Laguna SWA ring、native K-quant、CPU lane 與 OpenAI API。256K Q8 KV 實際為 6.42GiB；5700X/DDR4-2133 以 6 threads 最佳，APEX subcompact 暖機 API 約 4.06 tok/s。
- 2026-08-13：第二套 llama.cpp MoE-cache 在固定 256K、Q8 KV、APEX compact、256-token warmup 後三次 decode 為 10.74/9.72/10.26 tok/s，平均 10.24、p5 約 9.77；兩卡 cache hit 75.8%/71.9%，無 fill/dispatch/collect failure。
- 2026-08-13：per-expert placement、CPU/GPU overlap 與雙卡 hot cache 均已實際運作，但仍離 20 tok/s 約一倍；因此未宣稱速度或 `<5%` 品質驗收完成，也不以無法改善速度的 96GB 官方 Q4 下載取代實證。
- 2026-08-13：Q4_0 KV + 12GiB cache 的最終 speed probe 為 12.07/12.34/12.22 tok/s，平均 12.21、p5 12.09，比 Q8 約快 30%，但仍不到 15/20，且 Q4 KV 品質未通過 `<5%`。
- 2026-08-13：完成固定 revision 的 MoE-cache build/start scripts；乾淨 Release build、雙架構 fatbin、63 個 library tests、實際 256K `/v1/models`、非串流 chat 與 SSE `[DONE]` 均通過。
- 2026-08-13：使用者要求不要把有限 A/B 當成結論；下一輪改以 sequential-halving 自動搜尋未測的 ubatch、reserve、拓撲、CPU overlap 與 admission，再以 profiling 證據決定 backend 深改，持續以 p5 `>=20 tok/s` 為成功門檻。
- 2026-08-14：接續 Codex。參數層可部署數字改為 p5 `14.12`；`n_cpu_moe=47/46` 淘汰。同層 expert split 已落地且雙卡都有 dispatch，256K r3 p5 `14.59`，相對 14.12 只快約 3%，仍遠低於 20。
- 2026-08-14：collect prefetch + fused down。帶 `-v` p5 `17.99`，INFO-only p5 `19.32`。`test-moe-cache` 全過。續跑時 RTX 3060 `GPU is lost`，未完成 10x。
- 2026-08-14：確認 3060 故障並暫停 20 tok/s 工作。重開機後 `nvidia-smi` 為 5060 Ti 16GB + 3070 Ti 8GB。已清 DFlash 草稿、`.audit-build`、`.cache`、Web UI `node_modules`、Pulsar `target`／worktree 與暫存；正式 GGUF、MoE-cache binary 與 winner 證據保留。
- 2026-08-15：磁碟只留 APEX i-compact。已刪 subcompact（44.58GB）與 IQ4_XS（53.61GB）；manifest 保留，需要時用 `scripts/download-model.ps1` 重抓。
- 2026-08-16：完成 Git 存放庫初始化，強化 `.gitignore` 排除規則（包含 `models/`, `.cache/`, `tmp/`, `logs/`, `build/`, `third_party/`, `benchmarks/moe-*/`, `benchmarks/results/`, `.pytest_cache/`, `.coverage`, 敏感檔案等），驗證 53 項單元測試全數通過並建立 initial commit。
- 2026-08-16：建立 GitHub 存放庫 `RX5950XT/Laguna-S2.1-RTX5060Ti-RTX3060` 並推送至 `main` 分支。
- 2026-08-21：將 `tasks/handoff.md` 併入本檔「解封後接續」並刪除 handoff。倉庫因 3060 故障、只能改換 ≥12GB 第二張卡而**封存**。winner 仍是 fused-down INFO p5 19.32；10x 未跑。
