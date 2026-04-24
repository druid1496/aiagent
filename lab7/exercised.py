"""Exercise D: Citation Network Explorer Agent.

Given a seed paper ID, builds a "citation neighborhood" by:
  1. fetching seed metadata
  2. pulling references -> abstracts for the 5 most-cited
  3. pulling recent citing papers (last 3 years)
  4. for each seed author, grabbing their most-cited other work
  5. handing the collected data to GPT-4o mini to compose a markdown report

The agent controls tool-call order; the LLM only generates the final report.

Run: python exercised.py ARXIV:2210.03629
"""

import json
import os
import sys
from datetime import date
import requests
from openai import OpenAI

ASTA_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"
MODEL = "gpt-4o-mini"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

client = OpenAI()


def _parse_sse(resp_text: str) -> dict:
    for line in resp_text.splitlines():
        if line.startswith("data:"):
            return json.loads(line[len("data:"):].strip())
    raise RuntimeError(f"No SSE data line: {resp_text[:300]}")


def mcp_call(name: str, arguments: dict) -> dict | list:
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    resp = requests.post(ASTA_ENDPOINT, headers=HEADERS, json=payload, timeout=30)
    resp.raise_for_status()
    data = _parse_sse(resp.text)
    if "error" in data:
        raise RuntimeError(f"MCP error on {name}: {data['error']}")
    return json.loads(data["result"]["content"][0]["text"])


def _unwrap(obj):
    """Asta sometimes wraps lists as {'data': [...]}. Normalize to a list."""
    if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        return obj["data"]
    return obj


def fetch_seed(paper_id: str) -> dict:
    return mcp_call("get_paper", {
        "paper_id": paper_id,
        "fields": "title,abstract,year,authors,fieldsOfStudy,citationCount,references",
    })


def fetch_top_references(seed: dict, k: int = 5) -> list[dict]:
    refs = seed.get("references") or []
    ref_ids = [r.get("paperId") for r in refs if r.get("paperId")]
    if not ref_ids:
        return []
    detailed = _unwrap(mcp_call("get_paper_batch", {
        "ids": ref_ids,
        "fields": "title,abstract,year,authors,citationCount",
    }))
    detailed = [p for p in detailed if isinstance(p, dict)]
    detailed.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return detailed[:k]


def fetch_recent_citations(paper_id: str, k: int = 5) -> list[dict]:
    three_years_ago = f"{date.today().year - 3}-01-01:"
    raw = _unwrap(mcp_call("get_citations", {
        "paper_id": paper_id,
        "fields": "title,year,authors,abstract,citationCount",
        "limit": 20,
        "publication_date_range": three_years_ago,
    }))
    papers = []
    for entry in raw:
        p = entry.get("citingPaper", entry) if isinstance(entry, dict) else entry
        if isinstance(p, dict):
            papers.append(p)
    papers.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return papers[:k]


def fetch_author_top_work(author_id: str, seed_paper_id: str) -> dict | None:
    papers = _unwrap(mcp_call("get_author_papers", {
        "author_id": author_id,
        "paper_fields": "title,year,citationCount",
        "limit": 25,
    }))
    candidates = [
        p for p in papers
        if isinstance(p, dict) and p.get("paperId") != seed_paper_id
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.get("citationCount") or 0, reverse=True)
    return candidates[0]


def compose_report(seed: dict, refs: list[dict], citations: list[dict],
                   author_profiles: list[dict]) -> str:
    """Hand all collected data to the LLM for report generation."""
    bundle = {
        "seed_paper": {
            "title": seed.get("title"),
            "year": seed.get("year"),
            "abstract": seed.get("abstract"),
            "authors": [a.get("name") for a in (seed.get("authors") or [])],
            "fieldsOfStudy": seed.get("fieldsOfStudy"),
            "citationCount": seed.get("citationCount"),
        },
        "foundational_works": [
            {"title": r.get("title"), "year": r.get("year"),
             "abstract": r.get("abstract"), "citationCount": r.get("citationCount")}
            for r in refs
        ],
        "recent_developments": [
            {"title": c.get("title"), "year": c.get("year"),
             "abstract": c.get("abstract"), "citationCount": c.get("citationCount")}
            for c in citations
        ],
        "author_profiles": author_profiles,
    }

    system = (
        "You are a scientific research writer. Given structured data about a seed "
        "paper, its key references, recent citing works, and author profiles, produce "
        "a clean markdown report with these sections in order: "
        "# Title (seed paper), ## Summary (one paragraph), ## Foundational Works "
        "(bulleted, title/year/citations + 1-sentence relevance), ## Recent Developments "
        "(bulleted, same shape), ## Author Profiles (per author, their most notable other "
        "work). Do not fabricate any paper that isn't in the input."
    )

    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(bundle, ensure_ascii=False)},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def main(paper_id: str):
    print(f"[1/4] Fetching seed paper: {paper_id}", file=sys.stderr)
    seed = fetch_seed(paper_id)

    print("[2/4] Fetching top 5 references...", file=sys.stderr)
    refs = fetch_top_references(seed, k=5)

    print("[3/4] Fetching recent citations...", file=sys.stderr)
    citations = fetch_recent_citations(paper_id, k=5)

    print("[4/4] Profiling authors...", file=sys.stderr)
    profiles = []
    for author in (seed.get("authors") or []):
        aid = author.get("authorId")
        if not aid:
            continue
        top_work = fetch_author_top_work(aid, seed.get("paperId"))
        if top_work:
            profiles.append({
                "name": author.get("name"),
                "most_notable_other_work": {
                    "title": top_work.get("title"),
                    "year": top_work.get("year"),
                    "citationCount": top_work.get("citationCount"),
                },
            })

    print("Generating report...\n", file=sys.stderr)
    print(compose_report(seed, refs, citations, profiles))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python exercised.py <PAPER_ID>", file=sys.stderr)
        print("  e.g. python exercised.py ARXIV:2210.03629", file=sys.stderr)
        sys.exit(1)
    try:
        main(sys.argv[1])
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
