# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "torch==2.11.0",
#     "transformers==5.18.0",
#     "peft==0.21.1",
#     "trl==1.14.2",
#     "accelerate==1.15.0",
#     "datasets==5.1.0",
# ]
# ///
"""第2回 宿題（発展）: 用語集の Q&A で LoRA 学習し、学習前後と用語集検索を比べる。

ノートブック（notebooks/lora_format_tuning.ipynb）と同じモデル・revision・LoRA 設定・生成設定を使い、
学習データだけを scripts/make_sft_dataset.py の出力（messages 形式の JSONL）に替える。

使い方（リポジトリのルートで）:
    uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
    uv run lectures/02/homework/glossary_sft.py --data glossary_sft.jsonl --out lectures/02/homework/results/run_a

評価は 4 条件で行う。
- base: 学習前のモデルに質問だけを渡す
- base+search: 学習前のモデルに、app/tools.py の search_glossary(質問文) の結果を添えて渡す
- tuned: 学習後のモデルに質問だけを渡す
- tuned+search: 学習後のモデルに、検索結果を添えて渡す
"""

import argparse
import difflib
import json
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from app.tools import load_glossary, search_glossary  # noqa: E402

CONFIG = {
    "base_model": "Qwen/Qwen3.5-0.8B",
    "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    "seed": 42,
    "num_train_epochs": 5,
    "warmup_steps": 6,
    "lr": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "max_seq_len": 512,
    "max_new_tokens": 200,  # 定義文は 45〜212 文字。ノートブックの 96 では途中で切れる
    "gen": {"do_sample": False},
}

# 評価用の質問と正解。must_include は「すべて含む」。要素がリストのときはそのどれかを含めばよい
#   seen:       学習データと同じテンプレート（「{term}」とは何ですか？）で、用語集の見出し語を聞く
#   paraphrase: 学習データに無い言い回し・別名・逆引き（説明から用語名を答えさせる）
#   unknown:    用語集に無い語。一般知識が壊れていないか（must_include）と、知らないと言えるか（abstained）を見る
EVAL = [
    ("seen", "lora", "「LoRA」とは何ですか？", ["低ランク", ["アダプタ", "adapter"]]),
    ("seen", "hallucination", "「ハルシネーション」とは何ですか？", ["もっともらしく"]),
    ("seen", "default-model", "「既定モデル」とは何ですか？", ["gemini-3.5-flash-lite"]),
    ("seen", "homework-levels", "「宿題の3レベル」とは何ですか？", ["基礎", "実装", "発展"]),
    ("seen", "lec-start-tag", "「lecNN-start タグ」とは何ですか？", ["演習開始"]),
    ("seen", "pi", "「Pi」とは何ですか？", ["MIT", "コーディングエージェント"]),
    ("seen", "answer-md", "「answer.md」とは何ですか？", [["lectures/NN", "lectures/"], ["PR", "プルリクエスト"]]),
    ("seen", "tool-namespace", "「ツールの名前空間」とは何ですか？", [["接頭辞", "プレフィックス"]]),
    ("paraphrase", "lora", "Low-Rank Adaptation ってどういう仕組みですか？", ["低ランク"]),
    ("paraphrase", "default-model", "AIE2026 の演習で標準的に使うモデルはどれですか？", ["gemini-3.5-flash-lite"]),
    ("paraphrase", "homework-levels", "この講座の宿題はどんなレベル分けになっていますか？", ["基礎", "実装", "発展"]),
    ("paraphrase", "pi", "Pi coding agent はどんなソフトウェアですか？", ["MIT"]),
    ("paraphrase", "hallucination", "LLM がもっともらしい嘘をつく現象を何と呼びますか？", ["ハルシネーション"]),
    ("paraphrase", "tool-namespace", "複数の MCP サーバーでツール名がぶつかるときはどうしますか？", [["接頭辞", "プレフィックス"]]),
    ("paraphrase", "qlora", "QLoRA は LoRA と何が違いますか？", ["量子化"]),
    ("paraphrase", "answer-md", "この講座の宿題の結果はどのファイルに書いて提出しますか？", ["answer.md"]),
    ("unknown", None, "「RAG」とは何ですか？", ["検索"]),
    ("unknown", None, "「知識蒸留」とは何ですか？", [["教師", "大きなモデル", "大規模なモデル"]]),
    ("unknown", None, "「RLHF」とは何ですか？", [["人間", "人の"]]),
    ("unknown", None, "「ベクトルデータベース」とは何ですか？", [["類似", "近い"]]),
    ("unknown", None, "「Mixture of Experts」とは何ですか？", [["専門家", "エキスパート", "expert"]]),
    ("unknown", None, "「プロンプトインジェクション」とは何ですか？", [["指示", "命令"]]),
]
ABSTAIN_WORDS = ["記載がありません", "見つかりません", "分かりません", "わかりません", "確認が必要"]


def device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def user_content(question: str, with_search: bool) -> str:
    if not with_search:
        return question  # 学習データと同じく、質問だけを渡す
    return f"次の用語集の検索結果だけを根拠に答えてください。\n検索結果: {search_glossary(question)}\n\n質問: {question}"


def generate(model, tok, question: str, with_search: bool) -> str:
    messages = [{"role": "user", "content": user_content(question, with_search)}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=CONFIG["max_new_tokens"], pad_token_id=tok.pad_token_id, **CONFIG["gen"])
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def contains_all(text: str, must_include) -> bool:
    norm = text.casefold()
    return all(any(w.casefold() in norm for w in ([m] if isinstance(m, str) else m)) for m in must_include)


def evaluate(model, tok, label: str, with_search: bool, out_dir: Path) -> dict:
    definitions = {t["id"]: t["definition"] for t in load_glossary()}
    rows, t0 = [], time.time()
    for kind, term_id, question, must_include in EVAL:
        raw = generate(model, tok, question, with_search)
        row = {"kind": kind, "question": question, "raw": raw,
               "correct": contains_all(raw, must_include),
               "abstained": any(w in raw for w in ABSTAIN_WORDS)}
        if term_id:
            # 用語集の定義文との文字単位の一致度（0〜1）。丸暗記の度合いの目安
            row["similarity"] = round(difflib.SequenceMatcher(None, raw, definitions[term_id]).ratio(), 3)
        if with_search:
            row["search_hit"] = not search_glossary(question).startswith("用語集に「")
        rows.append(row)
    summary = {"label": label, "seconds": round(time.time() - t0, 1)}
    for kind in ["seen", "paraphrase", "unknown"]:
        rs = [r for r in rows if r["kind"] == kind]
        summary[f"{kind}_correct"] = f"{sum(r['correct'] for r in rs)}/{len(rs)}"
        if kind != "unknown":
            summary[f"{kind}_similarity"] = round(sum(r["similarity"] for r in rs) / len(rs), 3)
    summary["unknown_abstained"] = f"{sum(r['abstained'] for r in rows if r['kind'] == 'unknown')}/6"
    (out_dir / f"eval_{label}.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="make_sft_dataset.py が出力した JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=CONFIG["num_train_epochs"])
    parser.add_argument("--skip-base", action="store_true", help="学習前の評価を省く（学習データに依存しないので、別の run の結果を使う）")
    args = parser.parse_args()
    CONFIG["num_train_epochs"] = args.epochs
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(CONFIG["seed"])

    records = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    questions = {m["content"] for r in records for m in r["messages"] if m["role"] == "user"}
    leaked = [q for kind, _, q, _ in EVAL if kind != "seen" and q in questions]
    assert not leaked, f"評価用の質問が学習データに入っています: {leaked}"

    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(CONFIG["base_model"], revision=CONFIG["revision"], dtype=torch.float32).to(device())
    model.config.use_cache = True
    print("loaded:", CONFIG["base_model"], "| device:", device(), "| train records:", len(records), flush=True)

    summaries = [] if args.skip_base else [evaluate(model, tok, "base", False, args.out), evaluate(model, tok, "base+search", True, args.out)]

    model.config.use_cache = False
    lora = LoraConfig(r=CONFIG["lora_r"], lora_alpha=CONFIG["lora_alpha"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    sft = SFTConfig(output_dir=str(args.out / "ckpt"), num_train_epochs=CONFIG["num_train_epochs"], per_device_train_batch_size=4,
                    gradient_accumulation_steps=2, learning_rate=CONFIG["lr"], lr_scheduler_type="cosine", warmup_steps=CONFIG["warmup_steps"],
                    logging_steps=5, save_strategy="no", max_length=CONFIG["max_seq_len"], packing=False, report_to="none", seed=CONFIG["seed"])
    trainer = SFTTrainer(model=model, args=sft, train_dataset=Dataset.from_list(records), peft_config=lora, processing_class=tok)
    trainer.model.print_trainable_parameters()
    t0 = time.time()
    trainer.train()
    train_seconds = round(time.time() - t0, 1)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"学習時間: {train_seconds} 秒 | ステップ数: {trainer.state.global_step} | 最終 loss: {losses[-1]}", flush=True)
    trainer.model.save_pretrained(args.out / "adapter")
    tuned = trainer.model
    tuned.config.use_cache = True
    tuned.eval()

    summaries += [evaluate(tuned, tok, "tuned", False, args.out), evaluate(tuned, tok, "tuned+search", True, args.out)]

    result = {
        "config": CONFIG,
        "versions": {m.__name__: m.__version__ for m in [torch, __import__("transformers"), __import__("peft"), __import__("trl")]},
        "device": device(),
        "data": {"file": args.data.name, "records": len(records)},
        "train": {"seconds": train_seconds, "steps": trainer.state.global_step, "loss_history": losses},
        "summaries": summaries,
    }
    (args.out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
