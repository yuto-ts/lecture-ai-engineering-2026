# 第1回 解答

## 演習

- 講座に固有の質問（宿題の3レベル）へのモデルの回答（要約でよい）：
- 用語集の正しい意味と比べて気づいたこと：
- 表示できたトークン数（入力 / 出力）：

## 宿題

### 取り組んだレベル

発展（応答時間とトークン数を計測し、複数のモデルや質問で比較する。無料枠のレート制限との関係も考察する）

### 使ったモデル

| 提供元 | モデル | 料金 | 備考 |
|---|---|---|---|
| Google | `gemini-3.5-flash-lite` | 無料枠 | 講座の既定モデル |
| Google | `gemini-3.1-flash-lite` | 無料枠 | README で 503 のときの代わりとされているモデル |
| Google | `gemini-3.5-flash` | 無料枠 | 思考（thinking）するモデル |
| Anthropic | `claude-haiku-5-5` | 有料 | Claude の小型モデル |
| Anthropic | `claude-sonnet-5-5` | 有料 | Claude の中型モデル |
| Anthropic | `claude-opus-5-5` | 有料 | Claude の大型モデル |
| OpenAI | `gpt-5.4-nano` | 有料 | OpenAI の小型モデル |
| OpenAI | `gpt-5.4-mini` | 有料 | OpenAI の中型モデル |
| OpenAI | `gpt-5.5` | 有料 | OpenAI の大型モデル |

`gemini-2.5-flash` も試したが、12 回とも 404「This model models/gemini-2.5-flash is no longer available to new users」で使えず、比較から外した。設定はどのモデルも各社の既定のままにした（思考の量・effort・temperature は指定していない。Claude は `max_tokens` が必須なので 16000 にした）。

### やったこと

- 計測スクリプト `app/bench.py` を作った
  - モデル名の先頭（`gemini-` / `claude-` / `gpt-`）で呼び出し先を切り替え、どれもストリーミングで呼ぶ
  - 1 回ごとに、最初の本文が届くまでの秒数（TTFT）、全体が届くまでの秒数、入力・出力・思考トークン数を CSV に記録する
  - レート制限の情報も記録する。Claude と OpenAI はレスポンスヘッダーの上限値、Gemini は 429 のエラー本文にある `quotaId`・`quotaValue`・`retryDelay`
  - SDK の自動リトライは切り、429 や 503 はそのまま失敗として記録する
- 出力の長さが違う 4 つの質問を使った
  - short：「LoRAとは何ですか？一文で答えてください。」
  - normal：「LoRAとは何ですか？」
  - course：「AIE2026の講座の宿題にある「3レベル」とは何ですか？」
  - long：「Transformer の仕組みを、注意機構・位置エンコーディング・学習方法に分けて詳しく説明してください。」
- 次の 3 つの計測を、2026-10-26 にローカル PC から行った

  | 計測 | 内容 | 回数 | 生データ |
  |---|---|---|---|
  | ① Gemini の比較 | Gemini 3 モデル × 4 質問 × 3 回、間隔 5 秒（12:25 JST〜） | 36 回 | `results/bench.csv` |
  | ② 連続送信 | `gemini-3.5-flash-lite` に短い質問を待ちなしで送る | 30 回 | `results/burst.csv` |
  | ③ 3 社の比較 | 9 モデル × 4 質問 × 3 回、間隔 2 秒（13:06 JST〜） | 108 回 | `results/bench_multi.csv` |

- 再現方法（`.env` に各社の API キーを置く。`anthropic` と `openai` は dependency group `bench` に入れた）

  ```bash
  uv run --env-file .env python -m app.bench --out lectures/01/results/bench.csv
  uv run --env-file .env python -m app.bench --burst 30 --models gemini-3.5-flash-lite --out lectures/01/results/burst.csv
  uv run --group bench --env-file .env python -m app.bench --interval 2 --out lectures/01/results/bench_multi.csv \
    --models gemini-3.5-flash-lite,gemini-3.1-flash-lite,gemini-3.5-flash,claude-haiku-5-5,claude-sonnet-5-5,claude-opus-5-5,gpt-5.4-nano,gpt-5.4-mini,gpt-5.5
  ```

### 結果

#### 応答時間とトークン数（③ 3 社の比較）

成功した回の中央値。tok/s は（出力 + 思考トークン）÷ 全体の秒数。ok は成功回数 / 試行回数。

| モデル | 質問 | ok | TTFT (s) | 全体 (s) | 入力 | 出力 | 思考 | tok/s |
|---|---|---|---|---|---|---|---|---|
| gemini-3.5-flash-lite | short | 3/3 | 0.56 | 0.86 | 14 | 53 | 0 | 61 |
| gemini-3.5-flash-lite | normal | 3/3 | 0.57 | 2.51 | 7 | 543 | 0 | 216 |
| gemini-3.5-flash-lite | course | 3/3 | 0.61 | 2.78 | 21 | 547 | 0 | 197 |
| gemini-3.5-flash-lite | long | 3/3 | 0.62 | 5.14 | 22 | 1323 | 0 | 257 |
| gemini-3.1-flash-lite | short | 3/3 | 1.03 | 1.24 | 14 | 43 | 0 | 35 |
| gemini-3.1-flash-lite | normal | 2/3 | 3.19 | 5.66 | 7 | 699 | 0 | 124 |
| gemini-3.1-flash-lite | course | 3/3 | 1.26 | 3.42 | 21 | 671 | 0 | 196 |
| gemini-3.1-flash-lite | long | 1/3 | 0.63 | 3.64 | 22 | 898 | 0 | 247 |
| gemini-3.5-flash | short | 2/3 | 3.69 | 3.93 | 14 | 52 | 429 | 122 |
| gemini-3.5-flash | normal | 3/3 | 6.66 | 11.90 | 7 | 676 | 997 | 141 |
| gemini-3.5-flash | course | 3/3 | 8.76 | 12.71 | 21 | 585 | 1015 | 126 |
| gemini-3.5-flash | long | 1/3 | 8.06 | 20.50 | 22 | 2112 | 1456 | 174 |
| claude-haiku-5-5 | short | 3/3 | 0.77 | 1.27 | 27 | 90 | 0 | 71 |
| claude-haiku-5-5 | normal | 3/3 | 1.13 | 5.36 | 19 | 798 | 88 | 165 |
| claude-haiku-5-5 | course | 3/3 | 1.28 | 2.51 | 35 | 198 | 148 | 138 |
| claude-haiku-5-5 | long | 3/3 | 2.85 | 24.68 | 50 | 4738 | 532 | 214 |
| claude-sonnet-5-5 | short | 3/3 | 0.81 | 1.34 | 27 | 89 | 0 | 67 |
| claude-sonnet-5-5 | normal | 3/3 | 0.84 | 8.55 | 19 | 1015 | 0 | 119 |
| claude-sonnet-5-5 | course | 3/3 | 1.51 | 5.02 | 35 | 405 | 85 | 98 |
| claude-sonnet-5-5 | long | 3/3 | 7.05 | 41.36 | 50 | 5328 | 875 | 150 |
| claude-opus-5-5 | short | 3/3 | 1.56 | 2.32 | 27 | 98 | 76 | 75 |
| claude-opus-5-5 | normal | 3/3 | 1.61 | 10.11 | 19 | 921 | 75 | 99 |
| claude-opus-5-5 | course | 3/3 | 2.02 | 7.00 | 35 | 446 | 101 | 78 |
| claude-opus-5-5 | long | 3/3 | 5.31 | 61.57 | 50 | 6309 | 457 | 110 |
| gpt-5.4-nano | short | 3/3 | 0.95 | 1.52 | 20 | 55 | 0 | 36 |
| gpt-5.4-nano | normal | 3/3 | 0.81 | 3.47 | 13 | 403 | 0 | 116 |
| gpt-5.4-nano | course | 3/3 | 0.68 | 2.59 | 28 | 257 | 0 | 99 |
| gpt-5.4-nano | long | 3/3 | 0.64 | 17.14 | 37 | 2874 | 0 | 168 |
| gpt-5.4-mini | short | 3/3 | 0.65 | 0.95 | 20 | 35 | 0 | 37 |
| gpt-5.4-mini | normal | 3/3 | 0.78 | 3.44 | 13 | 484 | 0 | 141 |
| gpt-5.4-mini | course | 3/3 | 0.61 | 2.60 | 28 | 300 | 0 | 115 |
| gpt-5.4-mini | long | 3/3 | 0.90 | 10.76 | 37 | 2756 | 0 | 256 |
| gpt-5.5 | short | 3/3 | 1.28 | 2.19 | 20 | 53 | 25 | 36 |
| gpt-5.5 | normal | 3/3 | 2.44 | 8.71 | 13 | 514 | 111 | 72 |
| gpt-5.5 | course | 3/3 | 6.39 | 9.62 | 28 | 279 | 475 | 78 |
| gpt-5.5 | long | 3/3 | 2.03 | 48.67 | 37 | 4663 | 90 | 98 |

- 全体の時間は、ほぼ回答の長さで決まった。long への回答の長さはモデルで 7 倍違い（gemini-3.1-flash-lite 898 トークン、claude-opus-5-5 6309 トークン）、opus の long は 60〜67 秒かかった
- TTFT の中央値は 36 組中 28 組で 3 秒未満だった。3 秒を超えたのは、思考トークンが 400 以上ある組（gemini-3.5-flash の 4 質問すべて、gpt-5.5 の course）、claude-sonnet-5-5・claude-opus-5-5 の long（5.3〜7.1 秒）、gemini-3.1-flash-lite の normal（3.2 秒）だった
- 生成速度が最も高かったのは、gemini-3.5-flash-lite と gpt-5.4-mini の long（どちらも約 256 tok/s）だった
- 思考の量はモデルが質問ごとに変えていた。claude-sonnet-5-5 は short と normal で 0、claude-opus-5-5 は short・normal・course で 54〜112、gpt-5.5 は course だけ 398〜1013 と多かった。gemini-3.5-flash は「一文で」と頼んだ short でも 320〜537 あり、出力 52 トークンの約 8 倍だった
- 同じ日本語の質問でも、入力トークン数は社ごとに違った（short は Gemini 14、OpenAI 20、Claude 27）
- ① の Gemini だけの計測でも傾向は同じだった。ただし gemini-3.5-flash の long は 66.2 / 67.3 / 23.9 秒とばらついた。思考トークンは 1052 / 1381 / 1375 で大差がなく、差はほぼ TTFT（53.4 / 54.0 / 11.2 秒）に出ていた

#### レート制限

| 提供元 | 上限（このアカウントでの値） | 分かった方法 | 計測中に当たったか |
|---|---|---|---|
| Gemini 無料枠 | gemini-3.5-flash-lite：1 分 15 リクエスト | ② の 429 のエラー本文 | ② で当たった |
| Gemini 無料枠 | gemini-3.5-flash：1 日 20 リクエスト | ③ の 429 のエラー本文 | ③ で当たった |
| Claude | 1 分あたり リクエスト 10,000、入力 10,000,000 トークン、出力 2,000,000 トークン（3 モデル共通） | レスポンスヘッダー | 当たらなかった |
| OpenAI | 1 分あたり リクエスト 5,000、トークン 2,000,000（gpt-5.5 だけ 1,000,000） | レスポンスヘッダー | 当たらなかった |

- ② 連続送信：1〜17 回目は成功し、18 回目から 429 になった（`quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier`、`quotaValue: 15`）。429 のあいだにも 22・26 回目は成功した。`retryDelay` は 24s → 19s と減っていった
- ③ では gemini-3.5-flash が、この日 21 回成功したあとの 22 回目の試行で 429 になった（`quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier`、`quotaValue: 20`）。`retryDelay` は 9459 秒（約 2 時間 38 分）で、太平洋時間の 0 時（日本時間 16:00）から逆算すると 13:22 ごろの呼び出しになり、計測していた時間帯と合う
- ③ で失敗したのは Gemini の 6 回だけだった。503「This model is currently experiencing high demand」が 5 回（gemini-3.1-flash-lite 3 回、gemini-3.5-flash 2 回）と、上の 429 が 1 回。Claude と OpenAI の 72 回はすべて成功した

#### 費用

③ の CSV のトークン数に standard tier の単価を掛けた推定。思考トークンは出力として数えた。実際の請求額はコンソールで確認していない。

| モデル | 入力単価 / 出力単価（$/1M トークン） | 出力 + 思考トークン（12 回の合計） | 推定費用 |
|---|---|---|---|
| claude-opus-5-5 | 4 / 20 | 26,344 | $0.53 |
| gpt-5.5 | 5 / 30 | 17,632 | $0.53 |
| claude-sonnet-5-5 | 2 / 10 | 22,967 | $0.23 |
| gpt-5.4-mini | 0.75 / 4.50 | 10,737 | $0.05 |
| gpt-5.4-nano | 0.20 / 1.25 | 10,502 | $0.01 |
| claude-haiku-5-5 | 0.10 / 0.50 | 20,177 | $0.01 |
| 合計 | | | 約 $1.36 |

Gemini は無料枠の範囲で使ったので 0 円。

### 気づき・疑問

- 無料枠で先に当たった上限は、トークン数ではなくリクエスト数だった。入力は 1 回 7〜50 トークンと少なく、トークンの上限には遠い。Gemini の TPM の上限値はエラーにも公式ページにも出ず、確かめられていない
- 上限はリクエスト単位でかかるので、1 回が速いモデルほど早く上限に達する。gemini-3.5-flash-lite の short は 1 回 0.9 秒で、待ちなしだと 15 回を約 15 秒で使い切った
- 1 分 15 回の上限なのに 17 回目まで通り、429 のあとにも通る回があった。カウントの反映に遅れがあるのだと思う（推測）。上限ぎりぎりを狙うより、4 秒以上あけて送るほうが確実だと考える
- 第3回以降のエージェントは 1 つの質問で LLM を複数回呼ぶ。1 質問で 4 回呼ぶとすると、gemini-3.5-flash-lite でも 1 分に 3 質問強、gemini-3.5-flash は 1 日 5 質問しか処理できない計算になる。思考するモデルほど 1 日の上限が小さかったので、無料枠のうちは既定の gemini-3.5-flash-lite のまま使うのがよいと考える
- 有料 API の上限は、この計測のペース（1 分に数回）から 3 桁以上離れていて、一度も当たらなかった。503（混雑）も Gemini 無料枠でだけ起きた。無料枠は混雑時の優先度が低いのかもしれないが、有料の Gemini と比べていないので分からない
- 回答の長さはモデルの既定のふるまいで大きく変わり、応答時間と費用の差の大部分を占めた。速さを比べるなら TTFT と tok/s を分けて見る必要がある。全体の秒数だけだと、短く答えるモデルが速く見える
- 思考トークンが出力に含まれるという判断は、各 SDK の usage の構造（Gemini の `thoughts_token_count`、Claude と OpenAI の `output_tokens_details`）から行った。課金上の扱いは公式の料金ページで確かめていない
- モデル一覧（`models.list()`）に載っている `gemini-2.5-flash` が、新規ユーザーには 404 だった。一覧に出ていても、使えるとは限らない
