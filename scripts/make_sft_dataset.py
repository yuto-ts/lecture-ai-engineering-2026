"""第2回 宿題（発展）: 用語集から SFT 用の Q&A データ（JSONL）を作る。

使い方:
    uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl
    uv run python scripts/make_sft_dataset.py --aliases --unknown > glossary_sft_plus.jsonl

できたファイルを Colab にアップロードし、LoRA 学習のデータとして使う。
質問の言い回しを増やす、別名でも質問を作る、検証用に一部を分ける、などを工夫してみよう。

--aliases: 別名（aliases）でも同じ定義を答える例を足す
--unknown: 用語集に無い語には「記載がありません」と答える例を足す
"""

import argparse
import json
import sys
from pathlib import Path

GLOSSARY = Path(__file__).resolve().parent.parent / "data" / "glossary.json"
TEMPLATES = ["「{term}」とは何ですか？", "{term}について簡単に説明してください。"]
# 用語集に無い語（評価に使う語とは重ねない）
UNKNOWN_TERMS = ["埋め込み", "ビームサーチ", "強化学習", "データ拡張", "勾配降下法", "ドロップアウト", "バッチサイズ", "トークナイザー"]
UNKNOWN_ANSWER = "「{term}」は講座の用語集に記載がありません。講義資料か運営に確認が必要です。"


def record(question: str, answer: str) -> dict:
    return {"messages": [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aliases", action="store_true", help="別名でも質問を作る")
    parser.add_argument("--unknown", action="store_true", help="用語集に無い語への応答例を足す")
    args = parser.parse_args()

    terms = json.loads(GLOSSARY.read_text(encoding="utf-8"))["terms"]
    records = []
    for t in terms:
        for template in TEMPLATES:
            records.append(record(template.format(term=t["term"]), t["definition"]))
        if args.aliases:
            # 「r」「env」のような短い別名は質問として不自然なので除く
            for alias in (a for a in t["aliases"] if len(a) >= 3):
                records.append(record(TEMPLATES[0].format(term=alias), t["definition"]))
    if args.unknown:
        for term in UNKNOWN_TERMS:
            for template in TEMPLATES:
                records.append(record(template.format(term=term), UNKNOWN_ANSWER.format(term=term)))
    for r in records:
        sys.stdout.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
