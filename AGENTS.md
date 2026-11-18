# 專案代理規範

- 一律使用繁體中文（臺灣用語），回覆精簡並直接完成可完成事項。
- **本倉庫已封存（2026-08-21）**：3060 故障，GPU1 現為 3070 Ti 8GB。換上 ≥12GB 第二張卡前不要跑雙卡 10x、不要再掃 CLI。解封 SOP 在 `tasks/todo.md`。
- CLI 優先；修改後必須實測驗證，不能以設定檔存在取代 runtime 證據。
- 所有模型權重、Hugging Face cache、下載暫存、build 與 benchmark 產物只能放在本專案目錄。
- Laguna S 2.1 固定保留 `262144` context；不得用縮短 context 偽裝達成吞吐目標。
- 穩定吞吐門檻為單 request decode `>= 20 tok/s`；候選量化與 KV cache 的品質損失必須 `< 5%`。
- 本機 DDR4-2133 是已確認的穩定上限；禁止啟用 XMP 或記憶體超頻。
- 雙卡 20 tok/s 驗收必須是 RTX 5060 Ti 16GB + 12GB 級第二張卡；8GB 卡不得當作 3060 替代。
- 修改規劃與進度記錄於 `tasks/todo.md`，使用者修正與可重用教訓記錄於 `tasks/lessons.md`。
- 每次任務完成後同步精簡維護 `AGENTS.md`、`CLAUDE.md`，交接資訊更新 `tasks/todo.md`（≤100 行）。
- 不可硬編碼 secrets；服務預設只監聽 `127.0.0.1`。
