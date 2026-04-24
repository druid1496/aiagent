"""Exercise B: Three direct Asta tool-call drills.

Drill 1 — search_papers_by_relevance: recent LLM agent papers
Drill 2 — get_citations: papers citing BERT since 2023
Drill 3 — get_paper references: ReAct's intellectual foundation

Run: python exerciseb.py
"""

import json
import os
import sys
import requests

ASTA_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}


def mcp_call(name: str, arguments: dict, request_id: int = 1) -> dict:
    """Invoke an MCP tool and return the parsed JSON content."""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(ASTA_ENDPOINT, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()

    # Asta returns SSE — pull the JSON off the first `data:` line.
    data = None
    for line in resp.text.splitlines():
        if line.startswith("data:"):
            data = json.loads(line[len("data:"):].strip())
            break
    if data is None:
        raise RuntimeError(f"No SSE data line in response: {resp.text[:300]}")
    if "error" in data:
        raise RuntimeError(f"MCP error: {data['error']}")

    text = data["result"]["content"][0]["text"]
    return json.loads(text)


def drill1_search_papers():
    print("=" * 60)
    print("Drill 1: search_papers_by_relevance — LLM agent papers")
    print("=" * 60)
    result = mcp_call(
        "search_papers_by_relevance",
        {
            "keyword": "large language model agents",
            "fields": "title,abstract,year,authors",
            "limit": 5,
        },
        request_id=2,
    )
    papers = result.get("data", result) if isinstance(result, dict) else result
    if isinstance(papers, dict) and "data" in papers:
        papers = papers["data"]
    for i, p in enumerate(papers[:5], 1):
        title = p.get("title", "<no title>")
        year = p.get("year", "?")
        print(f"{i}. ({year}) {title}")
    print()


def drill2_get_citations():
    print("=" * 60)
    print("Drill 2: get_citations — BERT citations since 2023")
    print("=" * 60)
    result = mcp_call(
        "get_citations",
        {
            "paper_id": "ARXIV:1810.04805",
            "fields": "title,year,authors",
            "limit": 10,
            "publication_date_range": "2023-01-01:",
        },
        request_id=3,
    )
    citations = result.get("data", result) if isinstance(result, dict) else result
    if isinstance(citations, dict) and "data" in citations:
        citations = citations["data"]
    print(f"Returned {len(citations)} citations (limit 10).")
    for i, c in enumerate(citations[:5], 1):
        # get_citations often wraps each entry as {"citingPaper": {...}}
        paper = c.get("citingPaper", c) if isinstance(c, dict) else c
        title = paper.get("title", "<no title>")
        year = paper.get("year", "?")
        print(f"{i}. ({year}) {title}")
    print()


def drill3_references():
    """Asta doesn't expose get_references directly — pull them via get_paper
    with the `references` field, which returns the paper's bibliography."""
    print("=" * 60)
    print("Drill 3: get_paper references — ReAct's lineage (sorted by year)")
    print("=" * 60)
    result = mcp_call(
        "get_paper",
        {
            "paper_id": "ARXIV:2210.03629",
            "fields": "title,references.title,references.year",
        },
        request_id=4,
    )
    refs = result.get("references", []) if isinstance(result, dict) else []
    # Sort ascending by year; push unknown years to the end.
    refs_sorted = sorted(
        refs, key=lambda r: (r.get("year") is None, r.get("year") or 0)
    )
    for r in refs_sorted:
        year = r.get("year", "?")
        title = r.get("title", "<no title>")
        print(f"  {year}  {title}")
    print()


if __name__ == "__main__":
    try:
        drill1_search_papers()
        drill2_get_citations()
        drill3_references()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
