# Q: Which tool would you use to find all papers about "transformer attention mechanisms"?
# A: search_papers — it accepts a keyword/semantic query and searches the corpus directly.
#
# Q: Which tool would you use to find who else published in the same area as a specific author?
# A: get_author_papers — retrieve that author's publications to identify their research area,
#    then feed those keywords back into search_papers to discover other researchers in the same space.
#    (Some deployments expose a get_author or search_authors tool that can do this in one step.)

import os
import sys
import requests

ASTA_ENDPOINT = "https://asta-tools.allen.ai/mcp/v1"

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "x-api-key": os.environ["ASTA_API_KEY"],
}

payload = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {},
}

resp = requests.post(ASTA_ENDPOINT, headers=headers, json=payload, timeout=15)
resp.raise_for_status()

# The endpoint returns text/event-stream (SSE). Extract the JSON from the data: line.
import json as _json
data = None
for line in resp.text.splitlines():
    if line.startswith("data:"):
        data = _json.loads(line[len("data:"):].strip())
        break

if data is None:
    print("No data: line found in SSE response.", file=sys.stderr)
    print(resp.text[:500], file=sys.stderr)
    sys.exit(1)

if "error" in data:
    print(f"JSON-RPC error {data['error']['code']}: {data['error']['message']}", file=sys.stderr)
    sys.exit(1)

tools = data["result"]["tools"]


def fmt_params(schema: dict) -> tuple[list[str], list[str]]:
    """Return (required_lines, optional_lines) from a JSON Schema object."""
    props = schema.get("properties", {})
    required_set = set(schema.get("required", []))
    required, optional = [], []

    for name, info in props.items():
        # type may be a string or a list (anyOf shorthand)
        raw_type = info.get("type", "any")
        if isinstance(raw_type, list):
            type_str = "|".join(raw_type)
        else:
            type_str = raw_type
        entry = f"{name} ({type_str})"
        (required if name in required_set else optional).append(entry)

    return required, optional


for tool in tools:
    name = tool.get("name", "<unnamed>")
    description = next(
        (ln.strip() for ln in (tool.get("description") or "").splitlines() if ln.strip()),
        "(no description)"
    )
    input_schema = tool.get("inputSchema") or tool.get("input_schema") or {}

    required, optional = fmt_params(input_schema)

    print(f"Tool: {name}")
    print(f"  Description: {description}")
    if required:
        print(f"  Required: {', '.join(required)}")
    if optional:
        print(f"  Optional: {', '.join(optional)}")
    print()
