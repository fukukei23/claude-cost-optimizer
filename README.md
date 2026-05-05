# Claude Cost Optimizer

> A Python script that reduces Claude Code CLI API costs by 95-98% by routing requests to third-party AI APIs. Features automatic failover on errors, intelligent error classification, and JSONL logging for monitoring.

## 解説記事

この構成の設計思想・コスト比較・実測結果についてはZenn記事で詳しく解説しています。

https://zenn.dev/fukukei23/articles/claude-code-cost-optimization

## プロジェクト概要

Claude Code CLIのバックエンドをサードパーティのAI APIに切り替え、API利用料金を大幅に削減するためのスクリプトと設定テンプレートを公開しています。プライマリプロバイダの呼び出しに失敗した際、自動的にセカンダリプロバイダへフェイルオーバーする堅牢な仕組みを備えています。

### できること

- Claude Code CLIの処理を低コストなサードパーティAI APIにルーティングする設定の雛形を提供
- プライマリプロバイダへのリクエスト失敗時、セカンダリプロバイダへ自動フォールバック
- テンプレートをもとに自身の環境に合わせた設定を構築

## コスト比較

2026年4月時点のAPI従量課金に基づく1Mトークンあたりの概算です。

| モデル | 1Mトークン合計 | Opus比 |
|---|---|---|
| Claude Opus 4 | $40.00 | 100%（基準） |
| プライマリAI | $2.08 | 約5%（95%削減） |
| セカンダリAI | $0.64 | 約2%（98%削減） |

## 技術スタック

- Python 3
- JSON（設定ファイル）
- JSONL（ログ出力）

## プロジェクト構造

```text
claude-cost-optimizer/
├── claude_fallback.py       # フォールバックスクリプト本体
├── settings.example.json    # Claude Code CLI用設定テンプレート（サニタイズ済み）
├── fallback-config.json     # フォールバック動作の条件・閾値設定
└── CLAUDE_FALLBACK.md       # 設計思想と動作の詳細説明
```

### 各ファイルの役割

| ファイル | 内容 |
|---|---|
| `claude_fallback.py` | プライマリAIへのリクエスト失敗時にセカンダリAIへ自動フォールバックするスクリプト |
| `settings.example.json` | Claude Code CLI向けの設定テンプレート。値はサニタイズ済み |
| `fallback-config.json` | フォールバック動作の条件や閾値を定義する設定ファイルテンプレート |
| `CLAUDE_FALLBACK.md` | フォールバック構成の設計思想と動作の詳細説明 |

## セットアップ

本リポジトリは、APIの認証情報管理やCLIの設定ファイル編集、Pythonスクリプトの実行環境構築に一定の知識を前提としています。

1. リポジトリをクローンする
   ```bash
   git clone <repository-url>
   cd claude-cost-optimizer
   ```
2. `settings.example.json` を参考に、実際のAPIキーやエンドポイントを設定したJSONファイルを作成する
3. `fallback-config.json` でフォールバックの条件やリトライの閾値を調整する

各テンプレートの `YOUR_*` プレースホルダーに入れる具体的なプロバイダ・モデル名・APIエンドポイントは、note記事の有料部分で詳しく解説しています。

## 使い方

```bash
python3 claude_fallback.py <claude args...>
```

### 動作の流れ

1. プライマリAIへClaudeコマンドを送信する（リトライ設定に基づき再試行）
2. HTTPステータスコードとエラーキーワードでエラーを分類する
3. 再試行可能なエラーの場合、セカンダリAIへフェイルオーバーする
4. 認証エラー等の再試行不可エラーはフォールバックせず終了する
5. 実行結果を `~/.claude/logs/` にJSONL形式で記録する

### 動作確認

```bash
python3 claude_fallback.py --simulate-primary-failure <args...>
```

`--simulate-primary-failure` フラグでプライマリの強制失敗をシミュレートし、フォールバック動作を確認できます。

## お問い合わせ

環境構築や設定でお困りの方は、GitHub Discussionsよりお気軽にご相談ください。
