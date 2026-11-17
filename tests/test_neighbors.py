"""第5回 宿題（実装）: 前後ページ付き検索のテスト（APIキー不要）。"""

import json

from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.test import TestModel

from app import retrieval


def _hit(page_id: str, document_id: str, distance: float = 0.1) -> dict:
    return {"page_id": page_id, "document_id": document_id, "lecture": int(document_id[3:5]),
            "document_title": "t", "section": "s", "title": page_id, "page": 1,
            "content": "c", "distance": distance}


def _ids(results):
    return [(r["page_id"], r["role"]) for r in results]


def test_neighbors_added_in_pdf_order_without_duplicates(monkeypatch):
    # p052 と p053 が両方ヒット: p053 は neighbor として重複せず、hit として1回だけ出る
    monkeypatch.setattr(retrieval, "search", lambda q: [
        _hit("lec02-main-p052", "lec02-main"), _hit("lec02-main-p053", "lec02-main", 0.2)])
    results = retrieval.search_with_neighbors("q", window=1)
    assert _ids(results) == [
        ("lec02-main-p051", "neighbor"), ("lec02-main-p052", "hit"),
        ("lec02-main-p053", "hit"), ("lec02-main-p054", "neighbor"),
    ]
    assert len({r["page_id"] for r in results}) == len(results)
    assert results[0]["neighbor_of"] == "lec02-main-p052"
    assert results[0]["content"]  # 前後ページにも本文が入る


def test_neighbors_do_not_cross_documents(monkeypatch):
    pages = retrieval.load_pages()
    first_lec02 = next(p for p in pages if p["document_id"] == "lec02-main")
    last_lec01 = [p for p in pages if p["document_id"] == "lec01-main"][-1]
    monkeypatch.setattr(retrieval, "search", lambda q: [_hit(first_lec02["page_id"], "lec02-main")])
    results = retrieval.search_with_neighbors("q", window=1)
    assert all(r["document_id"] == "lec02-main" for r in results)
    assert last_lec01["page_id"] not in {r["page_id"] for r in results}
    assert len(results) == 2  # ヒット + 次ページだけ


def test_window_zero_equals_plain_search(monkeypatch):
    monkeypatch.setattr(retrieval, "search", lambda q: [_hit("lec03-main-p013", "lec03-main")])
    assert _ids(retrieval.search_with_neighbors("q", window=0)) == [("lec03-main-p013", "hit")]


def test_agent_neighbors_mode_keeps_tool_name(monkeypatch):
    from pydantic_ai import Tool
    from app import tools
    from app.agent import agent

    monkeypatch.setattr("app.retrieval.search", lambda q: [_hit("lec02-main-p052", "lec02-main")])
    tool = Tool(tools.search_materials_with_neighbors, name="search_materials")
    with agent.override(tools=[tool]):
        result = agent.run_sync("LoRA", model=TestModel(call_tools=["search_materials"]))
    returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
    roles = [r["role"] for r in json.loads(returns[0].content)["results"]]
    assert roles == ["neighbor", "hit", "neighbor"]
