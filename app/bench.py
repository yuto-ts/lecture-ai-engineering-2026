"""第1回 宿題（発展）: 応答時間とトークン数を、モデル × 質問で計測して比較する。

使い方:
    uv run python -m app.bench                       # Gemini の既定モデル・質問を各3回
    uv run python -m app.bench --models gemini-3.5-flash-lite --reps 5
    uv run python -m app.bench --burst 20 --models gemini-3.5-flash-lite  # 連続送信でレート制限を観察

    # Claude / OpenAI も計測する（ANTHROPIC_API_KEY / OPENAI_API_KEY が必要。有料）
    uv run --group bench python -m app.bench --models gemini-3.5-flash-lite,claude-haiku-5-5,gpt-5.4-mini

モデル名の先頭（gemini- / claude- / gpt-）で呼び出し先を切り替える。
1回の呼び出しごとに次を記録し、CSV に保存してから集計表を表示する。
- ttft_s:  最初の本文（テキスト）が届くまでの秒数（ストリーミングで計測）
- total_s: 応答がすべて届くまでの秒数
- 入力 / 出力（本文）/ 思考（thinking・reasoning）トークン数
- limits:  レスポンスヘッダーにあるレート制限の上限値（Claude と OpenAI のみ。Gemini は返さない）
- 429 のときは、Gemini ならエラー本文の quotaId（どの上限か）と quotaValue（上限値）
"""

import argparse
import csv
import statistics
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from google import genai
from google.genai import errors, types

from app.config import get_api_key

DEFAULT_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

# 出力の長さが違う質問をそろえる（短い / ふつう / 講座に固有 / 長い）
QUESTIONS = {
    "short": "LoRAとは何ですか？一文で答えてください。",
    "normal": "LoRAとは何ですか？",
    "course": "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
    "long": "Transformer の仕組みを、注意機構・位置エンコーディング・学習方法に分けて詳しく説明してください。",
}

# --burst で送る質問。出力を短くして、トークンではなくリクエスト数の上限を見る
BURST_QUESTION = "1+1は？数字だけで答えて。"

# Claude は max_tokens が必須。思考と長い回答が収まる大きさにする
CLAUDE_MAX_TOKENS = 16000


@dataclass
class Record:
    model: str
    question: str
    rep: int
    status: str  # "ok"、"refusal"、または HTTPステータスコード
    ttft_s: float | None = None
    total_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None  # 本文のトークン数（思考は含めない）
    thoughts_tokens: int | None = None
    output_chars: int | None = None
    limits: str = ""
    quota_id: str = ""
    quota_value: str = ""
    retry_delay: str = ""
    error: str = ""


class Timer:
    """開始からの経過秒数と、最初の本文が届いた時刻を測る。"""

    def __init__(self):
        self.start = time.perf_counter()
        self.ttft = None

    def mark_text(self):
        if self.ttft is None:
            self.ttft = time.perf_counter() - self.start

    def elapsed(self) -> float:
        return time.perf_counter() - self.start


def quota_info(details) -> tuple[str, str, str]:
    """Gemini の 429 のエラー本文から (quotaId, quotaValue, retryDelay) を取り出す。"""
    quota_ids, quota_values, retry = [], [], ""
    error = details.get("error", {}) if isinstance(details, dict) else {}
    for d in error.get("details", []):
        kind = d.get("@type", "")
        if kind.endswith("QuotaFailure"):
            for v in d.get("violations", []):
                quota_ids.append(v.get("quotaId", ""))
                quota_values.append(str(v.get("quotaValue", "")))
        elif kind.endswith("RetryInfo"):
            retry = d.get("retryDelay", "")
    return ";".join(quota_ids), ";".join(quota_values), retry


def rate_limits(headers) -> str:
    """レスポンスヘッダーからレート制限の上限値（*-limit*）だけを抜き出す。"""
    keep = []
    for key, value in headers.items():
        key = key.lower()
        if "ratelimit" in key and "limit" in key.replace("ratelimit", ""):
            short = key.replace("anthropic-ratelimit-", "").replace("x-ratelimit-", "")
            keep.append(f"{short}={value}")
    return ";".join(sorted(keep))


def measure_gemini(client, model, question, config=None) -> Record:
    record = Record(model=model, question="", rep=0, status="ok")
    timer = Timer()
    text, usage = "", None
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=question, config=config):
            if chunk.text:
                timer.mark_text()
                text += chunk.text
            usage = chunk.usage_metadata or usage  # 最後のチャンクに合計が入る
    except errors.APIError as e:
        record.status = str(e.code)
        record.error = (e.message or "")[:200]
        record.quota_id, record.quota_value, record.retry_delay = quota_info(e.details)
        return record
    record.ttft_s, record.total_s, record.output_chars = timer.ttft, timer.elapsed(), len(text)
    if usage:
        record.input_tokens = usage.prompt_token_count
        record.output_tokens = usage.candidates_token_count
        record.thoughts_tokens = usage.thoughts_token_count
    return record


def measure_claude(client, model, question, max_tokens=CLAUDE_MAX_TOKENS) -> Record:
    import anthropic

    record = Record(model=model, question="", rep=0, status="ok")
    timer = Timer()
    text = ""
    try:
        with client.messages.stream(
            model=model, max_tokens=max_tokens, messages=[{"role": "user", "content": question}]
        ) as stream:
            for delta in stream.text_stream:
                timer.mark_text()
                text += delta
            message = stream.get_final_message()
            record.limits = rate_limits(stream.response.headers)
    except anthropic.APIStatusError as e:
        record.status = str(e.status_code)
        record.error = str(e.message)[:200]
        record.retry_delay = e.response.headers.get("retry-after", "")
        record.limits = rate_limits(e.response.headers)
        return record
    if message.stop_reason == "refusal":
        record.status = "refusal"
    record.ttft_s, record.total_s, record.output_chars = timer.ttft, timer.elapsed(), len(text)
    usage = message.usage
    thinking = usage.output_tokens_details.thinking_tokens if usage.output_tokens_details else 0
    record.input_tokens = usage.input_tokens
    record.output_tokens = usage.output_tokens - thinking  # output_tokens は思考を含む
    record.thoughts_tokens = thinking
    return record


def measure_openai(client, model, question, max_output_tokens=None) -> Record:
    import openai

    record = Record(model=model, question="", rep=0, status="ok")
    timer = Timer()
    text = ""
    options = {"max_output_tokens": max_output_tokens} if max_output_tokens else {}
    response = None
    try:
        # ヘッダー（レート制限）も読むため、with_raw_response でストリームを開く
        raw = client.responses.with_raw_response.create(model=model, input=question, stream=True, **options)
        record.limits = rate_limits(raw.headers)
        for event in raw.parse():
            if event.type == "response.output_text.delta":
                timer.mark_text()
                text += event.delta
            elif event.type == "response.completed":
                response = event.response
    except openai.APIStatusError as e:
        record.status = str(e.status_code)
        record.error = str(e.message)[:200]
        record.retry_delay = e.response.headers.get("retry-after", "")
        record.limits = rate_limits(e.response.headers)
        return record
    record.ttft_s, record.total_s, record.output_chars = timer.ttft, timer.elapsed(), len(text)
    if response is None:  # response.completed が来なかった（failed / incomplete）
        record.status = "incomplete"
        return record
    usage = response.usage
    reasoning = usage.output_tokens_details.reasoning_tokens if usage.output_tokens_details else 0
    record.input_tokens = usage.input_tokens
    record.output_tokens = usage.output_tokens - reasoning  # output_tokens は reasoning を含む
    record.thoughts_tokens = reasoning
    return record


class Clients:
    """プロバイダごとのクライアントを、使うときに1回だけ作る。"""

    def __init__(self):
        self._clients = {}

    def measure(self, model: str, question_id: str, question: str, rep: int, burst: bool = False) -> Record:
        if model.startswith("claude-"):
            r = measure_claude(self._get("claude"), model, question, **({"max_tokens": 16} if burst else {}))
        elif model.startswith(("gpt-", "o")):
            # OpenAI の max_output_tokens は 16 が下限
            r = measure_openai(self._get("openai"), model, question, **({"max_output_tokens": 16} if burst else {}))
        else:
            config = types.GenerateContentConfig(max_output_tokens=16) if burst else None
            r = measure_gemini(self._get("gemini"), model, question, config)
        r.question, r.rep = question_id, rep
        return r

    def _get(self, provider: str):
        if provider not in self._clients:
            # SDK の自動リトライは切る（待ち時間が計測に混ざらないように、429 もそのまま記録する）
            if provider == "claude":
                import anthropic

                self._clients[provider] = anthropic.Anthropic(max_retries=0)
            elif provider == "openai":
                import openai

                self._clients[provider] = openai.OpenAI(max_retries=0)
            else:
                self._clients[provider] = genai.Client(api_key=get_api_key())
        return self._clients[provider]


def summarize(records: list[Record]) -> list[dict]:
    """モデル × 質問ごとに、成功した呼び出しの中央値をまとめる。"""
    groups: dict[tuple[str, str], list[Record]] = {}
    for r in records:
        groups.setdefault((r.model, r.question), []).append(r)

    def median(values):
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    rows = []
    for (model, question), rs in groups.items():
        ok = [r for r in rs if r.status == "ok"]
        total = median(r.total_s for r in ok)
        output = median(r.output_tokens for r in ok)
        thoughts = median(r.thoughts_tokens or 0 for r in ok) if ok else None
        generated = (output or 0) + (thoughts or 0)
        rows.append({
            "model": model,
            "question": question,
            "ok": f"{len(ok)}/{len(rs)}",
            "ttft_s": median(r.ttft_s for r in ok),
            "total_s": total,
            "input": median(r.input_tokens for r in ok),
            "output": output,
            "thoughts": thoughts,
            # 思考トークンも生成に時間がかかるので、出力と合わせて速度を出す
            "tok_per_s": generated / total if total else None,
        })
    return rows


def print_table(rows: list[dict]) -> None:
    def fmt(v):
        if v is None:
            return "-"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    headers = list(rows[0])
    cells = [[fmt(row[h]) for h in headers] for row in rows]
    widths = [max(len(h), *(len(c[i]) for c in cells)) for i, h in enumerate(headers)]
    print(" | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("-|-".join("-" * w for w in widths))
    for c in cells:
        print(" | ".join(v.ljust(w) for v, w in zip(c, widths)))


def save_csv(records: list[Record], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[f.name for f in fields(Record)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)


def main() -> None:
    parser = argparse.ArgumentParser(description="応答時間とトークン数を計測する")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="カンマ区切り")
    parser.add_argument("--questions", default=",".join(QUESTIONS), help=f"カンマ区切り（{', '.join(QUESTIONS)}）")
    parser.add_argument("--reps", type=int, default=3, help="同じ質問を何回ずつ聞くか")
    parser.add_argument("--interval", type=float, default=5.0, help="呼び出しの間に待つ秒数（RPM 対策）")
    parser.add_argument("--burst", type=int, default=0, help="指定すると、待たずにこの回数だけ連続で送る")
    parser.add_argument("--out", type=Path, default=Path("bench_results.csv"))
    args = parser.parse_args()

    clients = Clients()
    models = args.models.split(",")
    records: list[Record] = []

    if args.burst:
        for model in models:
            for i in range(args.burst):
                r = clients.measure(model, "burst", BURST_QUESTION, i + 1, burst=True)
                print(f"[{model}] {i + 1:>3}: {r.status} {r.quota_id} {r.quota_value}", flush=True)
                records.append(r)
    else:
        jobs = [(m, q, rep) for rep in range(1, args.reps + 1) for m in models for q in args.questions.split(",")]
        for n, (model, question_id, rep) in enumerate(jobs, 1):
            r = clients.measure(model, question_id, QUESTIONS[question_id], rep)
            print(f"({n}/{len(jobs)}) {model} {question_id} #{rep}: {r.status} "
                  f"total={r.total_s and round(r.total_s, 2)} out={r.output_tokens} {r.quota_id}", flush=True)
            records.append(r)
            if n < len(jobs):
                time.sleep(args.interval)

    save_csv(records, args.out)
    print(f"\n[saved] {args.out}\n")
    print_table(summarize(records))


if __name__ == "__main__":
    main()
