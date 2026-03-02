# Lab 2: LangGraph Multi-Agent Chat System

This lab demonstrates progressively building a LangGraph-based multi-agent chat system with Llama and Qwen LLMs.

## Files Overview

| File | Description |
|------|-------------|
| `langgraph_simple_llama_agent_t3.py` | Parallel execution of both Llama and Qwen |
| `langgraph_simple_llama_agent_t4.py` | Conditional routing: "Hey Qwen" → Qwen, else → Llama |
| `langgraph_simple_llama_agent_t5.py` | Chat history with Message API (Llama only) |
| `langgraph_simple_llama_agent_t6.py` | Multi-participant chat with unified history |
| `langgraph_simple_llama_agent_t7.py` | **Checkpointing and crash recovery** |

---

## Task 3: Parallel LLM Execution (`t3.py`)

**Goal:** Run both Llama and Qwen in parallel, display both responses.

**Graph Structure:**
```
get_user_input → fan_out → call_llama ──┬→ combine_and_print → get_user_input
                        → call_qwen  ──┘
```

**Key Implementation:**
- `fan_out` node distributes input to both LLM nodes
- LangGraph's native parallelism runs both models simultaneously
- `combine_and_print` collects and displays both responses

---

## Task 4: Conditional Routing (`t4.py`)

**Goal:** Route to either Llama OR Qwen based on user input prefix.

**Routing Logic:**
- Input starts with "Hey Qwen" → routes to Qwen
- Otherwise → routes to Llama

**Graph Structure:**
```
get_user_input → [conditional] → call_llama → print_response → get_user_input
                      ↓
                  call_qwen  → print_response ───────────────────┘
```

---

## Task 5: Chat History with Message API (`t5.py`)

**Goal:** Maintain conversation context using LangChain's Message API.

**Message Types Used:**
- `SystemMessage` - Sets assistant behavior (role: "system")
- `HumanMessage` - User input (role: "user")
- `AIMessage` - Assistant response (role: "assistant")

**State Definition:**
```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
    should_exit: bool
```

**Key Feature:** Using `Annotated[..., operator.add]` to accumulate messages across nodes.

---

## Task 6: Multi-Participant Conversation (`t6.py`)

**Challenge:** Chat APIs only support user/assistant/system roles, but we have THREE participants (Human, Llama, Qwen).

**Solution:** When formatting messages for a specific LLM:
- That LLM's messages → `"assistant"` role
- All other participants → `"user"` role with name prefix

**Example for Qwen:**
```python
[
    {"role": "user", "content": "Human: What is the best ice cream?"},
    {"role": "user", "content": "Llama: Vanilla is most popular."}
]
```

**Example for Llama (after Qwen responds):**
```python
[
    {"role": "user", "content": "Human: What is the best ice cream?"},
    {"role": "assistant", "content": "Vanilla is most popular."},
    {"role": "user", "content": "Qwen: No way, chocolate is best!"},
    {"role": "user", "content": "Human: I agree."}
]
```

**System Prompts:** Each LLM gets a custom system prompt explaining:
- Who it is (Llama or Qwen)
- The three participants in the conversation
- That messages from others have name prefixes

---

## Task 7: Checkpointing and Crash Recovery (`t7.py`)

**Goal:** Persist conversation state so the program can be killed and restarted without losing history.

### Key Components:

1. **SQLite Checkpointer:**
```python
from langgraph.checkpoint.sqlite import SqliteSaver

with SqliteSaver.from_conn_string("conversation_checkpoints.db") as checkpointer:
    graph = graph_builder.compile(checkpointer=checkpointer)
```

2. **Thread ID for Session Management:**
```python
config = {"configurable": {"thread_id": "main_conversation"}}
graph.invoke(initial_state, config)
```

3. **Startup Options:**
   - Detects existing conversation on startup
   - Offers to resume or start fresh
   - `--new` flag to skip prompt and start fresh

### How to Test Crash Recovery:

```bash
# Start conversation
python langgraph_simple_llama_agent_t7.py --new

# Chat a bit, then press Ctrl+C to kill

# Restart - conversation is restored!
python langgraph_simple_llama_agent_t7.py
# Press Enter to resume

# Verify: Type "history" to see restored messages
```

### Commands:
| Command | Description |
|---------|-------------|
| `history` | Show message history |
| `verbose` | Enable verbose output |
| `quiet` | Disable verbose output |
| `quit` | Exit (conversation saved) |
| `--new` | Start fresh (command line flag) |

---

## Requirements

```
torch
transformers
langchain-huggingface
langgraph
langgraph-checkpoint-sqlite
grandalf
```

## Setup

```bash
# Create virtual environment
python3 -m venv myenvlab2
source myenvlab2/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run (example)
python langgraph_simple_llama_agent_t7.py
```

---

## Graph Visualization

The graph structure is saved to `lg_graph.png` on each run.

## Models Used

- **Llama:** `meta-llama/Llama-3.2-1B-Instruct`
- **Qwen:** `Qwen/Qwen2.5-0.5B`

Both models are loaded via HuggingFace Transformers and wrapped with `HuggingFacePipeline` for LangChain integration.
