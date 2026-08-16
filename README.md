# Laguna S 2.1 本機部署

本專案針對 RTX 5060 Ti 16GB + 12GB 級第二張卡（原 RTX 3060 12GB）、Ryzen 7 5700X、96GB DDR4-2133，部署 Laguna S 2.1 並固定 `262144` context。模型、cache、暫存、build 與 benchmark 大型產物都限制在專案內。

**目前暫停**：3060 故障，現槽位是 3070 Ti 8GB，不能替代 12GB 雙卡驗收。等換卡後再續跑 20 tok/s。

## 實測結論

- 已完成 Windows 原生 CUDA `sm_86` + `sm_120a` runtime、模型 SHA-256 校驗、OpenAI-compatible API、SWA KV ring、per-expert hot cache、CPU/GPU overlap 與同層雙 GPU expert split。
- Poolside llama.cpp baseline：APEX i-compact、256K、Q4 KV，約 `7.23 tok/s`。
- Pulsar：256K Q8 KV 實際 `6.42GiB`；APEX subcompact 暖機 API 約 `4.06 tok/s`。
- MoE-cache 可部署 layer-pin winner（背景服務、Q4 KV、12GiB cache、`t=4`、`ubatch=160`）：`14.34 / 14.30 / 14.10`，p5 `14.12 tok/s`。
- 同層 expert split + collect prefetch + fused down（預設開）：帶 `-v` 的 256K r3 為 `18.0075 / 18.1243 / 17.9905`，p5 `17.99`；INFO-only 為 `19.506 / 19.6213 / 19.3222`，p5 `19.32`。failure 為 0。仍低於 20。
- Q4 KV 尚未通過 `<5%` 品質 gate。DFlash 因 acceptance 約 `5–9%` 且更慢而停用。

hot-expert 常駐與同層雙卡計算都已實機驗證，但本機尚未達成穩定 `20 tok/s`。

品質損失 `<5%` 也不能由量化名稱推定。官方 Q4_K_M 當前實檔是 `96,031,829,760` bytes，不是舊文件的 68GB；APEX 的 BF16-relative PPL 仍無公開完整結果，因此目前不宣稱任何較小量化已通過 `<5%`。

## 專案內容

- `scripts/start.ps1`：可重現的 Poolside llama.cpp baseline。
- `scripts/build-moe-cache.ps1`、`scripts/start-moe-cache.ps1`：目前最快後端；預設 Q4 KV、12GiB cache、`t=4`、`ubatch=160`、同層 expert split，固定 256K、loopback、localhost CORS 並關閉 Web UI。可用 `GGML_CUDA_MOE_CACHE_SPLIT_EXPERTS=0` 退回 layer-pin。
- `scripts/build-pulsar.ps1`、`scripts/start-pulsar.ps1`：固定 revision 與 patch set 的 Windows Pulsar。
- `patches/pulsar/`：Windows port、SWA KV ring、native K-quant、CPU lane 與 API server。
- `benchmarks/`：OpenAI API、SSE timestamp、長 context needle 與結構化結果。
- `config/models.json`：模型 revision、容量與 SHA-256 manifest。磁碟預設只留 APEX i-compact；其餘候選用 `scripts/download-model.ps1` 再抓。
- `CONTEXT.md`、`tasks/todo.md`：實測證據與未通過的驗收門檻。

記憶體只能以 DDR4-2133 穩定運作；所有設定與驗收均排除 XMP、記憶體超頻及 PBO。
