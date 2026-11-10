# Laguna S 2.1 本機部署

**目前狀態：** 原本的 RTX 3060 12GB 故障，現槽位是 RTX 3070 Ti 8GB，不能替代 12GB 雙卡驗收。換上 **≥12GB** 的第二張卡後再繼續衝 20 tok/s；在那之前不要在 8GB 卡上硬測。

接續步驟與 winner 旋鈕見 [`tasks/todo.md`](tasks/todo.md)。實測摘要見 [`CONTEXT.md`](CONTEXT.md)。

本專案針對 RTX 5060 Ti 16GB + 12GB 級第二張卡、Ryzen 7 5700X、96GB DDR4-2133，部署 Laguna S 2.1 並固定 `262144` context。模型、cache、暫存、build 與 benchmark 大型產物都限制在專案內。記憶體只能以 DDR4-2133 穩定運作；所有設定與驗收均排除 XMP、記憶體超頻及 PBO。

## 目前進度

最快雙卡（fused-down、INFO-only、無 STATS）：p5 **19.32 tok/s**（19.506 / 19.6213 / 19.3222）。帶 `-v` 為 p5 17.99。離 20 約 3%。**10x 未跑。**

Winner：APEX i-compact（50.64 GB，不是官方 Q4_K_M）、`n_ctx=262144`、KV **Q4_0**（K 與 V）、12GiB MoE-cache、reserve 2GiB、`t=4`、`ubatch=160`、`n_cpu_moe=48`、mmap、`GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS=1`。

```powershell
.\scripts\tune-moe-cache.ps1 -Stage confirm -Case fused-down
```

用 pwsh 直接跑，不要加 `-v`，不要用 `powershell.exe`。

## 實測結論

- 已完成 Windows 原生 CUDA `sm_86` + `sm_120a` runtime、模型 SHA-256 校驗、OpenAI-compatible API、SWA KV ring、per-expert hot cache、CPU/GPU overlap、同層雙 GPU expert split、collect prefetch、fused down、CUDA graph。
- Poolside llama.cpp baseline：APEX i-compact、256K、Q4 KV，約 `7.23 tok/s`。
- Pulsar：256K Q8 KV 實際 `6.42GiB`；APEX subcompact 暖機 API 約 `4.06 tok/s`。公開 22.4 tok/s 是 2K／32GB／9900X／IQ2，不能當成本機 256K 成果。
- MoE-cache layer-pin：`14.34 / 14.30 / 14.10`，p5 `14.12 tok/s`。
- 同層 split：p5 `14.59`。fused-down `-v` p5 `17.99`；INFO-only p5 `19.32`。fill/dispatch/collect failure 為 0。
- Q4 KV 尚未通過 `<5%` 品質 gate。DFlash 因 acceptance 約 `5–9%` 且更慢而停用。`n_cpu_moe=47/46` 已淘汰。

## 專案內容

- `scripts/start.ps1`：可重現的 Poolside llama.cpp baseline。
- `scripts/build-moe-cache.ps1`、`scripts/start-moe-cache.ps1`：目前最快後端；預設 Q4_0 KV、12GiB cache、`t=4`、`ubatch=160`、同層 expert split，固定 256K、loopback `127.0.0.1`、localhost CORS 並關閉 Web UI。可用 `GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS=0` 退回 layer-pin。
- `scripts/tune-moe-cache.ps1 -Stage confirm -Case fused-down`：換卡後第一條指令。
- `scripts/build-pulsar.ps1`、`scripts/start-pulsar.ps1`：固定 revision 與 patch set 的 Windows Pulsar。
- `patches/moe-cache/`：0001–0004（context、split、fused down、CUDA graph）。
- `config/models.json`：模型 revision、容量與 SHA-256。磁碟只留 APEX i-compact；其餘用 `scripts/download-model.ps1` 再抓。
- `tasks/todo.md`：目前狀態、換卡後接續 SOP、歷史階段。
- `CONTEXT.md`：交接摘要。

遠端：[`RX5950XT/Laguna-S2.1-RTX5060Ti-RTX3060`](https://github.com/RX5950XT/Laguna-S2.1-RTX5060Ti-RTX3060)。
