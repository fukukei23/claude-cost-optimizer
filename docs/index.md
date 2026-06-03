---
title: 概要
nav_order: 1
---

# Claude Cost Optimizer

> 📂 **[GitHub リポジトリ →](https://github.com/fukukei23/claude-cost-optimizer)**{: .btn .btn-blue } — 設定ファイル・フォールバックスクリプト詳細はこちらから

Claude Code CLIのバックエンドをサードパーティAI APIに切り替え、API利用料金を **95-98%削減** するフォールバックスクリプト。

## コスト比較

| モデル | 1Mトークン | Opus比 |
|---|---|---|
| Claude Opus 4 | $40.00 | 100%（基準） |
| プライマリAI | $2.08 | 約5% |
| セカンダリAI | $0.64 | 約2% |

## できること

- **自動フォールバック**: プライマリAI失敗時 → セカンダリAIに自動切替
- **エラー分類**: HTTPステータス・キーワードでリトライ可否判定
- **JSONLログ**: リクエスト・レスポンス・コストを時系列記録
- **設定テンプレート**: `settings.example.json` / `fallback-config.json`

## 技術スタック

| カテゴリ | 技術 |
|---|---|
| 言語 | Python 3 |
| 設定 | JSON |
| ログ | JSONL |
| 対象 | Claude Code CLI |

---

> 👉 詳細設定・解説記事: https://zenn.dev/fukukei23/articles/claude-code-cost-optimization
