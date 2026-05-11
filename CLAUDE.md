# Claude Code Cost Optimizer

Claude Code CLI の API コストを 95-98% 削減するフォールバックスクリプト。
プライマリAI（Claude）の失敗時にセカンダリAI へ自動フェイルオーバーする仕組みを提供する。

**Zenn 記事**: https://zenn.dev/fukukei23/articles/claude-code-cost-optimization

## ファイル構造

| ファイル | 役割 |
|---|---|
| `claude_fallback.py` | フォールバック本体。プライマリ失敗時にセカンダリAIへ自動切り替え |
| `settings.example.json` | Claude Code CLI 設定テンプレート（サニタイズ済み） |
| `fallback-config.json` | フォールバック条件・閾値設定テンプレート |
| `CLAUDE_FALLBACK.md` | 設計思想と動作詳細のドキュメント |

## 主要コマンド

```bash
# 通常実行（フォールバック付き）
python3 claude_fallback.py <claude args...>

# フォールバックのシミュレーション（テスト用）
python3 claude_fallback.py --simulate-primary-failure <args...>
```

## アーキテクチャ概要

```
ユーザー → claude_fallback.py → プライマリAI (Claude)
                                    │ 失敗/タイムアウト/エラー
                                    ↓
                               セカンダリAI（低コスト代替）
                                    ↓
                               レスポンス返却 + ~/.claude/logs/ にJSONL記録
```

## ⚠️ セキュリティ・運用上の注意

- **APIキー等の機密情報は絶対にコミットしない**（パブリックリポジトリ）
- `settings.example.json` はサニタイズ済みテンプレートのみ — `YOUR_*` プレースホルダーを使用
- **実際の設定**: `~/.claude/settings.json` に配置
- **ログ出力先**: `~/.claude/logs/` （JSONL形式）

## 設定の流れ

1. `settings.example.json` を `~/.claude/settings.json` にコピー
2. 実際のAPIキー・エンドポイントを設定
3. `fallback-config.json` でフォールバック条件（タイムアウト・リトライ回数等）を調整
4. `claude_fallback.py` を Claude Code CLI のコマンドフックとして登録
