"""Exercise C: Asta-powered research chatbot.

Fetches tool schemas from the MCP server at startup, converts them to
OpenAI function-calling format, and lets GPT-4o mini decide which
Asta tools to call.

Run: python exercisec.py
"""

import json
import os
import sys
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


def get_asta_tools() -> list[dict]:
    """Call tools/list and convert each MCP tool to OpenAI function format."""
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = requests.post(ASTA_ENDPOINT, headers=HEADERS, json=payload, timeout=15)
    resp.raise_for_status()
    data = _parse_sse(resp.text)
    if "error" in data:
        raise RuntimeError(f"tools/list error: {data['error']}")

    tools = []
    for t in data["result"]["tools"]:
        tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": (t.get("description") or "").strip(),
                "parameters": t.get("inputSchema") or {"type": "object", "properties": {}},
            },
        })
    return tools


def call_asta_tool(name: str, arguments: dict) -> str:
    """Execute a tools/call and return the raw text content."""
    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    try:
        resp = requests.post(ASTA_ENDPOINT, headers=HEADERS, json=payload, timeout=30)
        resp.raise_for_status()
        data = _parse_sse(resp.text)
        if "error" in data:
            return json.dumps({"error": data["error"]})
        return data["result"]["content"][0]["text"]
    except Exception as exc:
        return json.dumps({"error": str(exc)})


SYSTEM_PROMPT = (
    "You are a research assistant with access to the Semantic Scholar corpus "
    "(225M+ academic papers) via Asta MCP tools. Use the tools to find papers, "
    "trace citations, and profile authors. Cite titles and years in your answers. "
    "If a tool call fails, acknowledge it and try an alternative approach."
)


def chat_loop():
    tools = get_asta_tools()
    print(f"Loaded {len(tools)} Asta tools.\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_msg = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_msg:
            continue
        if user_msg in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_msg})

        # Inner loop: keep going until the model produces a message with no tool calls.
        while True:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                print(f"bot> {msg.content}\n")
                break

            for call in msg.tool_calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                print(f"  [tool] {name}({json.dumps(args)[:120]})")
                result = call_asta_tool(name, args)
                # Truncate huge results to keep context manageable.
                if len(result) > 8000:
                    result = result[:8000] + "...[truncated]"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })


if __name__ == "__main__":
    try:
        chat_loop()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
