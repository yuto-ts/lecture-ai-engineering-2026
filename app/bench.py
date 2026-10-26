"""第1回 宿題（発展）: 応答時間とトークン数を、モデル × 質問で計測して比較する。

使い方:
    uv run python -m app.bench                       # 既定のモデル・質問を各3回
    uv run python -m app.bench --models gemini-3.5-flash-lite --reps 5
    uv run python -m app.bench --burst 20 --models gemini-3.5-flash-lite  # 連続送信でレート制限を観察

1回の呼び出しごとに次を記録し、CSV に保存してから集計表を表示する。
- ttft_s:  最初のチャンクが届くまでの秒数（ストリーミングで計測）
- total_s: 応答がすべて届くまでの秒数
- 入力 / 出力 / 思考（thinking）トークン数
- 429 のときは、エラー本文に含まれる quotaId（どの上限か）と quotaValue（上限値）
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


@dataclass
class Record:
    model: str
    question: str
    rep: int
    status: str  # "ok" または HTTPステータスコード
    ttft_s: float | None = None
    total_s: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    thoughts_tokens: int | None = None
    output_chars: int | None = None
    quota_id: str = ""
    quota_value: str = ""
    retry_delay: str = ""
    error: str = ""


def quota_info(details) -> tuple[str, str, str]:
    """429 のエラー本文から (quotaId, quotaValue, retryDelay) を取り出す。"""
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


def measure(client: genai.Client, model: str, question_id: str, question: str, rep: int,
            config: types.GenerateContentConfig | None = None) -> Record:
    """1回呼び出して計測する。API エラーも記録として返す（例外にしない）。"""
    record = Record(model=model, question=question_id, rep=rep, status="ok")
    start = time.perf_counter()
    text, usage = "", None
    try:
        for chunk in client.models.generate_content_stream(model=model, contents=question, config=config):
            if record.ttft_s is None:
                record.ttft_s = time.perf_counter() - start
            text += chunk.text or ""
            usage = chunk.usage_metadata or usage  # 最後のチャンクに合計が入る
    except errors.APIError as e:
        record.status = str(e.code)
        record.error = (e.message or "")[:200]
        record.quota_id, record.quota_value, record.retry_delay = quota_info(e.details)
        return record
    record.total_s = time.perf_counter() - start
    record.output_chars = len(text)
    if usage:
        record.input_tokens = usage.prompt_token_count
        record.output_tokens = usage.candidates_token_count
        record.thoughts_tokens = usage.thoughts_token_count
    return record


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

    client = genai.Client(api_key=get_api_key())
    models = args.models.split(",")
    records: list[Record] = []

    if args.burst:
        config = types.GenerateContentConfig(max_output_tokens=16)
        for model in models:
            for i in range(args.burst):
                r = measure(client, model, "burst", BURST_QUESTION, i + 1, config)
                print(f"[{model}] {i + 1:>3}: {r.status} {r.quota_id} {r.quota_value}", flush=True)
                records.append(r)
    else:
        jobs = [(m, q, rep) for rep in range(1, args.reps + 1) for m in models for q in args.questions.split(",")]
        for n, (model, question_id, rep) in enumerate(jobs, 1):
            r = measure(client, model, question_id, QUESTIONS[question_id], rep)
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
