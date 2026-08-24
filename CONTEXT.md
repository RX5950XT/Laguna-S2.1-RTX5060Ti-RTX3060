# 開發交接

**本倉庫已封存（2026-08-21）。** 3060 故障，只能改換 ≥12GB 第二張卡。解封 SOP 在 `tasks/todo.md`「解封後接續」。

## 目前狀態：封存

- 2026-08-14：RTX 3060 12GB `GPU is lost`。重開機後 GPU0 RTX 5060 Ti 16GB + GPU1 RTX 3070 Ti 8GB。
- 8GB 不能替代 12GB MoE-cache 雙卡驗收。封存期間不要跑雙卡 10x、不要再掃 CLI。
- 暫停前最快雙卡：fused-down、INFO-only、無 STATS、p5 `19.32 tok/s`（平均 `19.48`）。帶 `-v` p5 `17.99`。10x 未跑。
- 磁碟只留 APEX i-compact（50.64GB）。KV 為 Q4_0。context 固定 262144。

## 目標與硬限制

- 目標硬體：RTX 5060 Ti 16GB + **12GB 級**第二張卡、Ryzen 7 5700X、96GB DDR4-2133。
- 不得使用 XMP、記憶體超頻或以 PBO 作驗收條件。
- Laguna S 2.1 context 固定 `262144`，目標 decode p5 `>=20 tok/s`、品質損失 `<5%`。
- 所有模型、cache、tmp、build 與 benchmark 大型產物只能留在專案內。服務只聽 `127.0.0.1`。

## 已完成

- Poolside llama.cpp Laguna fork `06f8cebd7fe728687be3d19f8bdedb70d75883af`，Windows CUDA 13.3、`sm_86 + sm_120a`。
- Pulsar 上游 `c7245425dbc46cd8b310370a71d05fa30a9d9b34`；leloch MoE-cache `e3096b046bb809f7f80bc47801f6579aed1cbc60` + patches `0001`–`0004`。
- CLI 收斂：t=4、ubatch=160、n_cpu_moe=48、12GiB cache、reserve 2GiB、mmap、split-experts。layer-pin p5 `14.12`；split p5 `14.59`。
- Collect prefetch + fused-down + CUDA graph。`test-moe-cache`（split／collect／fused-down，含 `graph-replays`）通過。
- DFlash 與 `n_cpu_moe=47/46` 已淘汰。官方 Q4_K_M（96.03GB）未下載。GitHub `RX5950XT/Laguna-S2.1-RTX5060Ti-RTX3060`。

## 判定

最快數字離 20 約 3%（約 2 ms/token）。collect-wait 約 1.6 ms，不是主因。若解封後 10x 仍 < 20，下一刀是 scheduler attn↔MoE host bounce，不要再掃 CLI。

`<5%` 品質門檻未過。30 分鐘 soak、260K+512、完整 `/v1` 未做。

## 解封後

見 `tasks/todo.md`：確認 GPU1 ≥12GB → `tune-moe-cache.ps1 -Stage confirm -Case fused-down` → 兩次獨立 10x。pwsh、不要 `-v`。
