# 開發交接

## 目前狀態：硬體暫停

- 2026-08-14：RTX 3060 12GB 故障。重開機後 `nvidia-smi` 為 GPU0 RTX 5060 Ti 16GB + GPU1 RTX 3070 Ti 8GB。
- 8GB 卡不能替代 12GB 的 MoE-cache 雙卡驗收。`p5 >= 20` 工作暫停，等換回 12GB 級第二張卡。
- 暫停前最快雙卡數字仍是 fused-down、INFO-only、無 STATS 的 p5 `19.32 tok/s`（平均 `19.48`）。10x 未跑。

## 目標與硬限制

- 目標硬體：RTX 5060 Ti 16GB + 12GB 級第二張卡、Ryzen 7 5700X、96GB DDR4。
- RAM 只能在 DDR4-2133 穩定運作；不得使用 XMP、記憶體超頻或以 PBO 作驗收條件。
- Laguna S 2.1 context 固定 `262144`，目標 decode p5 `>=20 tok/s`、品質損失 `<5%`。
- 所有模型、cache、tmp、build 與 benchmark 大型產物只能留在專案內。

## 已完成

- Poolside llama.cpp Laguna fork 固定 `06f8cebd7fe728687be3d19f8bdedb70d75883af`，Windows CUDA 13.3 建置 `sm_86 + sm_120a`。
- Pulsar 固定上游 `c7245425dbc46cd8b310370a71d05fa30a9d9b34`；leloch MoE-cache 固定 `e3096b046bb809f7f80bc47801f6579aed1cbc60`，另有專案 patch `0001`–`0004`（context、split、fused down、CUDA graph）。
- 可部署 CLI winner（背景服務）：APEX compact、256K、Q4 KV、12GiB cache、reserve 2GiB、`t=4`、`ubatch=160`、`n_cpu_moe=48`。乾淨 r3：`14.34 / 14.30 / 14.10`，平均 `14.25`、p5 `14.12`。
- `n_cpu_moe=47/46` 把最後 1–2 層留在 GPU：`11.56 / 11.21 tok/s`，已淘汰。
- 同層 expert split（`GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS=1`，預設開）：兩張 GPU 對同一層 Top-K 各自做 MMV。256K r3：`14.82 / 15.44 / 14.59`，平均 `14.95`、p5 `14.59`。
- Collect prefetch + 雙 lane 同步重疊，再把 down MMV 接在同一條 stream（`fused_begin_ex`）。帶 `-v` 的 256K r3：`18.0075 / 18.1243 / 17.9905`，平均 `18.041`、p5 `17.99`。關掉 DEBUG 後：`19.506 / 19.6213 / 19.3222`，平均 `19.483`、p5 `19.32`。`n_ctx=262144`，fill/dispatch/collect failure 均為 0。download-prefetch 從約 93 降到約 46 次/token。
- `test-moe-cache` 含 `cache-same-layer-split`、`cache-same-layer-split-collect`、`cache-fused-down` 全數通過。
- DFlash 因 acceptance 5–9% 且端到端更慢而停用。
- 官方 Q4_K_M 實檔 `96,031,829,760` bytes；尚未為形式驗收下載。
- 磁碟上只留 APEX i-compact（50.64GB）。subcompact 與 IQ4_XS 已刪；checksum 仍在 `config/models.json`，品質門檻或 Pulsar 對照時再下載。
- 完成 Git 存放庫初始化與 `.gitignore` 排除規則強化（包含大型模型、快取、暫存、build、third_party、搜尋與日誌產物、pytest 快取、敏感檔案等），全套 53 個單元測試均通過並建立 initial commit。

## 判定

per-expert cache、同層 split、collect prefetch 與 fused down 都已實機運作。目前最快雙卡數字是 fused-down、INFO log、無 STATS 的 p5 `19.32 tok/s`（平均 `19.48`），離 20 約 3%。帶 `-v` 的可核對證據鏈數字是 p5 `17.99`。collect-wait 只剩約 1.6 ms/token，不是主因。

`<5%` 品質門檻未過：Q4 KV 沒有 BF16-relative PPL。

## 仍未通過

- 暖機後 10 次 decode p5 `>=20 tok/s`。
- 30 分鐘 20 tok/s 條件 soak。
- 相對 BF16/官方品質基準的 PPL、coding 與 long-context 差異 `<5%`。
- thinking、tool calling 與 260K prompt + 512-token generation 的完整最終驗收。

## 換卡後下一輪

- 先確認 `nvidia-smi` 兩張卡健康且第二張 ≥12GB，再跑 fused-down 256K r3 與兩次獨立 10x。
- 不要再掃 CLI。離 20 只剩約 2 ms/token，下一刀是 scheduler 的 attn↔MoE host bounce，或把獨立的 shared-expert CUDA 工作疊進 collect wait。
- fused-down compute 已收成 CUDA graph（`graph-replays` 由 `cache-fused-down` 鎖定）。
- 可用 `GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS=0` 做 layer-pin 對照。

實測結果以 `tasks/todo.md` 與 `benchmarks/moe-cache-search/` 為準。
