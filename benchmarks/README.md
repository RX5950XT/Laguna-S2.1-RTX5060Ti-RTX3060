# OpenAI-compatible API benchmark

本工具只使用 Python standard library，驗證 `/v1/models`、非串流 chat、
串流 chat 與長 context needle retrieval。每個 SSE event 都保留 UTC 時間與
相對 request 起點的 monotonic timestamp；tok/s 優先使用串流最後的
`usage.completion_tokens`，若服務未回傳 usage，報告會明確標示為 event-count
fallback。

```powershell
$env:OPENAI_API_KEY = "..." # 本機無驗證時可省略
python -m benchmarks `
  --base-url http://127.0.0.1:8000 `
  --model laguna `
  --context-window-tokens 262144 `
  --prompt-tokens 260000 `
  --output benchmarks/results/laguna.json
```

`--prompt-tokens` 是以 `characters / --chars-per-token` 推估，不冒充 tokenizer
精確計數；報告會同時記錄估算方法、實際字元數及 needle 深度。完整 prompt
不寫入 JSON，避免結果檔膨脹。

測試：

```powershell
python -m unittest discover -s benchmarks/tests -v
```
