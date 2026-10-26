"""計測スクリプトの集計部分のテスト（LLMもAPIキーも使わない）。"""

from app.bench import Record, quota_info, rate_limits, summarize


def test_rate_limits_keeps_only_limit_headers():
    claude = {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "49",
        "anthropic-ratelimit-requests-reset": "2026-10-26T04:04:43Z",
        "anthropic-ratelimit-output-tokens-limit": "8000",
        "content-type": "text/event-stream",
    }
    assert rate_limits(claude) == "output-tokens-limit=8000;requests-limit=50"
    openai = {"x-ratelimit-limit-requests": "500", "x-ratelimit-remaining-requests": "499"}
    assert rate_limits(openai) == "limit-requests=500"


def test_quota_info_reads_429_details():
    details = {
        "error": {
            "code": 429,
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [
                        {"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "quotaValue": "15"}
                    ],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "42s"},
            ],
        }
    }
    assert quota_info(details) == ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "15", "42s")


def test_quota_info_handles_missing_details():
    assert quota_info(None) == ("", "", "")
    assert quota_info({"error": {"code": 503}}) == ("", "", "")


def test_summarize_uses_only_successful_calls():
    records = [
        Record("m", "q", 1, "ok", ttft_s=0.5, total_s=2.0, input_tokens=10, output_tokens=100, thoughts_tokens=None),
        Record("m", "q", 2, "ok", ttft_s=0.7, total_s=4.0, input_tokens=10, output_tokens=300, thoughts_tokens=100),
        Record("m", "q", 3, "429"),
    ]
    [row] = summarize(records)
    assert row["ok"] == "2/3"
    assert row["total_s"] == 3.0
    assert row["output"] == 200
    assert row["thoughts"] == 50
    assert row["tok_per_s"] == (200 + 50) / 3.0
