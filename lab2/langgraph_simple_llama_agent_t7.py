# langgraph_simple_agent.py
# Program demonstrates CHECKPOINTING and CRASH RECOVERY with LangGraph.
#
# Key Feature: You can kill this program (Ctrl+C) in the middle of a conversation
# and restart it - the entire conversation history will be restored!
#
# How it works:
# - Uses SqliteSaver to persist graph state to a local SQLite database
# - Each conversation has a thread_id that identifies it
# - On startup, checks for existing conversations and offers to resume
#
# The challenge: Chat APIs only support user/assistant/system roles, but we have
# THREE participants: Human, Llama, and Qwen.
#
# Solution: When formatting messages for an LLM, that LLM's messages become "assistant"
# role, while messages from the other participants (human + other LLM) become "user"
# role with name prefixes.
#
# Graph structure:
#   get_user_input -> [conditional] -> call_llama -> print_response -> get_user_input
#                          |        -> call_qwen  -> print_response -+
#                          +-> END

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Literal
import operator
import sqlite3

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

# Import the SQLite checkpointer for persistence
from langgraph.checkpoint.sqlite import SqliteSaver

# =============================================================================
# CONFIGURATION
# =============================================================================

verbose_flag = True

# Database file for checkpointing (stores conversation state)
CHECKPOINT_DB = "conversation_checkpoints.db"

# Default thread ID for the conversation
DEFAULT_THREAD_ID = "main_conversation"

# System prompts that explain the multi-participant conversation
LLAMA_SYSTEM_PROMPT = """You are Llama, an AI assistant in a multi-participant conversation.
The participants are:
- Human: The user asking questions
- Llama (you): An AI assistant created by Meta
- Qwen: Another AI assistant created by Alibaba

You can see what others have said in the conversation. Messages from other participants 
are prefixed with their names (e.g., "Human: ..." or "Qwen: ...").
Be helpful, friendly, and feel free to agree or politely disagree with Qwen's opinions.
Keep your responses concise but informative."""

QWEN_SYSTEM_PROMPT = """You are Qwen, an AI assistant in a multi-participant conversation.
The participants are:
- Human: The user asking questions
- Llama: Another AI assistant created by Meta
- Qwen (you): An AI assistant created by Alibaba

You can see what others have said in the conversation. Messages from other participants 
are prefixed with their names (e.g., "Human: ..." or "Llama: ...").
Be helpful, friendly, and feel free to agree or politely disagree with Llama's opinions.
Keep your responses concise but informative."""


# =============================================================================
# STATE DEFINITION
# =============================================================================

class ChatMessage(TypedDict):
    """A message in the multi-participant conversation."""
    speaker: Literal["human", "llama", "qwen"]
    content: str


class AgentState(TypedDict):
    """
    State for the multi-participant conversation.
    This state is automatically persisted by the checkpointer after each node.
    """
    chat_history: Annotated[Sequence[ChatMessage], operator.add]
    user_input: str
    should_exit: bool
    last_speaker: str


# =============================================================================
# DEVICE DETECTION
# =============================================================================

def get_device():
    """Detect and return the best available compute device."""
    global verbose_flag
    if torch.cuda.is_available():
        if verbose_flag:
            print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        if verbose_flag:
            print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        if verbose_flag:
            print("Using CPU for inference")
        return "cpu"


# =============================================================================
# LLM CREATION
# =============================================================================

def create_llm(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    """Create and configure an LLM, returning both the LLM and tokenizer."""
    device = get_device()

    global verbose_flag
    if verbose_flag:
        print(f"Loading model: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    if device == "mps":
        model = model.to(device)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    if verbose_flag:
        print(f"Model {model_id} loaded successfully!")
    
    return llm, tokenizer


# =============================================================================
# MESSAGE FORMATTING
# =============================================================================

def format_history_for_llm(
    chat_history: Sequence[ChatMessage],
    target_llm: Literal["llama", "qwen"],
    system_prompt: str,
    tokenizer
) -> str:
    """
    Format the chat history for a specific LLM.
    
    Rules:
    - Messages from target_llm -> role: "assistant" (no name prefix in content)
    - Messages from others (human/other LLM) -> role: "user" with name prefix
    """
    chat_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history:
        speaker = msg["speaker"]
        content = msg["content"]
        
        if speaker == target_llm:
            chat_messages.append({
                "role": "assistant",
                "content": content
            })
        else:
            name = speaker.capitalize()
            chat_messages.append({
                "role": "user",
                "content": f"{name}: {content}"
            })
    
    try:
        formatted = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        print(f"Warning: Could not apply chat template: {e}")
        result = f"System: {system_prompt}\n\n"
        for msg in chat_messages[1:]:
            role = msg["role"].capitalize()
            result += f"{role}: {msg['content']}\n"
        result += "Assistant:"
        return result


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def check_existing_conversation(db_path: str, thread_id: str) -> bool:
    """Check if there's an existing conversation with the given thread_id."""
    if not os.path.exists(db_path):
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Check if the checkpoints table exists and has data for this thread
        cursor.execute("""
            SELECT COUNT(*) FROM checkpoints 
            WHERE thread_id = ?
        """, (thread_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def get_conversation_summary(db_path: str, thread_id: str) -> str:
    """Get a summary of the existing conversation."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT checkpoint FROM checkpoints 
            WHERE thread_id = ?
            ORDER BY checkpoint_id DESC
            LIMIT 1
        """, (thread_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            import json
            checkpoint = json.loads(row[0])
            # Try to extract chat history from checkpoint
            channel_values = checkpoint.get("channel_values", {})
            chat_history = channel_values.get("chat_history", [])
            if chat_history:
                msg_count = len(chat_history)
                human_count = len([m for m in chat_history if m.get("speaker") == "human"])
                return f"{msg_count} messages ({human_count} from human)"
        return "Unknown state"
    except Exception as e:
        return f"Could not read: {e}"


def clear_conversation(db_path: str, thread_id: str):
    """Clear the conversation for a given thread_id."""
    if not os.path.exists(db_path):
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
        conn.commit()
        conn.close()
        print(f"✅ Cleared conversation for thread '{thread_id}'")
    except Exception as e:
        print(f"Warning: Could not clear conversation: {e}")


# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_graph(llama_llm, llama_tokenizer, qwen_llm, qwen_tokenizer, checkpointer):
    """
    Create the LangGraph with checkpointing enabled.
    
    The checkpointer automatically saves state after each node execution,
    enabling crash recovery.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        global verbose_flag
        
        turn_count = len([m for m in state["chat_history"] if m["speaker"] == "human"]) + 1
        
        print("\n" + "=" * 60)
        print(f"Turn {turn_count} - Enter your text (or 'quit' to exit):")
        print("Start with 'Hey Qwen' to address Qwen, otherwise Llama responds")
        print("💾 State is automatically saved - you can Ctrl+C safely!")
        print("=" * 60)

        print("\n> ", end="")
        try:
            user_input = input()
        except EOFError:
            return {"user_input": "quit", "should_exit": True, "chat_history": []}
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Your conversation is saved.")
            print("   Run the program again to resume.\n")
            return {"user_input": "quit", "should_exit": True, "chat_history": []}

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Your conversation is saved.")
            return {
                "user_input": user_input,
                "should_exit": True,
                "chat_history": []
            }
        
        # Handle special commands
        if user_input == "verbose":
            verbose_flag = True
            print("Verbose mode ON")
            return {"user_input": "", "should_exit": False, "chat_history": []}
        elif user_input == "quiet":
            verbose_flag = False
            print("Verbose mode OFF")
            return {"user_input": "", "should_exit": False, "chat_history": []}
        elif user_input == "history":
            print("\n📜 Conversation History:")
            for i, msg in enumerate(state["chat_history"]):
                icon = {"human": "👤", "llama": "🦙", "qwen": "🤖"}.get(msg["speaker"], "?")
                content = msg["content"][:60] + "..." if len(msg["content"]) > 60 else msg["content"]
                print(f"  {i+1}. {icon} {msg['speaker'].capitalize()}: {content}")
            if not state["chat_history"]:
                print("  (No messages yet)")
            return {"user_input": "", "should_exit": False, "chat_history": []}
        elif user_input == "clear":
            print("⚠️  To clear the conversation, restart with --new flag")
            return {"user_input": "", "should_exit": False, "chat_history": []}

        # Add human message to history
        human_message: ChatMessage = {
            "speaker": "human",
            "content": user_input
        }
        
        return {
            "user_input": user_input,
            "should_exit": False,
            "chat_history": [human_message]
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        global verbose_flag
        if verbose_flag:
            print("\n🦙 Processing with Llama-3.2-1B-Instruct...")
            print(f"   (Context: {len(state['chat_history'])} messages in history)")

        prompt = format_history_for_llm(
            state["chat_history"],
            target_llm="llama",
            system_prompt=LLAMA_SYSTEM_PROMPT,
            tokenizer=llama_tokenizer
        )
        
        if verbose_flag:
            print(f"   Prompt length: {len(prompt)} characters")

        response = llama_llm.invoke(prompt)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        if verbose_flag:
            print("🦙 Llama: Done!")

        llama_message: ChatMessage = {
            "speaker": "llama",
            "content": response
        }
        
        return {
            "chat_history": [llama_message],
            "last_speaker": "llama"
        }

    # =========================================================================
    # NODE 3: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        global verbose_flag
        
        if verbose_flag:
            print("\n🤖 Processing with Qwen2.5-0.5B...")
            print(f"   (Context: {len(state['chat_history'])} messages in history)")

        prompt = format_history_for_llm(
            state["chat_history"],
            target_llm="qwen",
            system_prompt=QWEN_SYSTEM_PROMPT,
            tokenizer=qwen_tokenizer
        )
        
        if verbose_flag:
            print(f"   Prompt length: {len(prompt)} characters")

        response = qwen_llm.invoke(prompt)
        
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        if verbose_flag:
            print("🤖 Qwen: Done!")

        qwen_message: ChatMessage = {
            "speaker": "qwen",
            "content": response
        }
        
        return {
            "chat_history": [qwen_message],
            "last_speaker": "qwen"
        }

    # =========================================================================
    # NODE 4: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        last_speaker = state.get("last_speaker", "")
        
        speaker_msgs = [m for m in state["chat_history"] if m["speaker"] == last_speaker]
        if speaker_msgs:
            latest_response = speaker_msgs[-1]["content"]
        else:
            latest_response = "(No response)"
        
        print("\n" + "=" * 60)
        if last_speaker == "llama":
            print("🦙 Llama-3.2-1B-Instruct:")
        elif last_speaker == "qwen":
            print("🤖 Qwen2.5-0.5B:")
        else:
            print("Response:")
        print("=" * 60)
        print(latest_response)
        print("=" * 60)
        print("💾 State saved to checkpoint")

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if state["user_input"] == "":
            return "get_user_input"
        
        if state["user_input"].lower().startswith("hey qwen"):
            return "call_qwen"
        
        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION WITH CHECKPOINTER
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llama": "call_llama",
            "call_qwen": "call_qwen",
            "get_user_input": "get_user_input",
            END: END
        }
    )

    graph_builder.add_edge("call_llama", "print_response")
    graph_builder.add_edge("call_qwen", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    # COMPILE WITH CHECKPOINTER - This enables crash recovery!
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph


def save_graph_image(graph, filename="lg_graph.png"):
    """Generate and save a Mermaid diagram of the graph."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")


def main():
    """
    Main function with checkpointing for crash recovery.
    
    Features:
    - Automatically saves state after each node execution
    - On restart, offers to resume existing conversation or start fresh
    - Survives Ctrl+C interrupts without losing data
    """
    import sys
    
    print("=" * 60)
    print("Multi-Participant Chat with CRASH RECOVERY")
    print("=" * 60)
    print()
    print("✨ NEW: Checkpointing enabled!")
    print("   - Your conversation is saved after each message")
    print("   - Press Ctrl+C anytime - nothing will be lost")
    print("   - Run again to resume where you left off")
    print()
    
    thread_id = DEFAULT_THREAD_ID
    
    # Check for --new flag to start fresh
    start_fresh = "--new" in sys.argv
    
    # Check for existing conversation
    has_existing = check_existing_conversation(CHECKPOINT_DB, thread_id)
    
    if has_existing and not start_fresh:
        summary = get_conversation_summary(CHECKPOINT_DB, thread_id)
        print(f"📂 Found existing conversation: {summary}")
        print("   Options:")
        print("   - Press Enter to RESUME")
        print("   - Type 'new' to start FRESH")
        print("   - Run with --new flag to skip this prompt")
        
        choice = input("\n> ").strip().lower()
        if choice == "new":
            clear_conversation(CHECKPOINT_DB, thread_id)
            has_existing = False
            print("Starting fresh conversation...")
        else:
            print("Resuming conversation...")
    elif start_fresh and has_existing:
        clear_conversation(CHECKPOINT_DB, thread_id)
        has_existing = False
        print("Starting fresh conversation (--new flag)...")
    
    print()
    print("Instructions:")
    print("  - Type normally to talk to Llama")
    print("  - Start with 'Hey Qwen' to talk to Qwen")
    print("  - Type 'history' to see the message log")
    print("  - Type 'quit' to exit (conversation saved)")
    print("  - Press Ctrl+C anytime (conversation saved)")
    print()

    # Load both models
    print("Loading Llama-3.2-1B-Instruct...")
    llama_llm, llama_tokenizer = create_llm("meta-llama/Llama-3.2-1B-Instruct")
    
    print("\nLoading Qwen2.5-0.5B...")
    qwen_llm, qwen_tokenizer = create_llm("Qwen/Qwen2.5-0.5B")

    # Create SQLite checkpointer for persistence
    print(f"\n💾 Using checkpoint database: {CHECKPOINT_DB}")
    
    # Use context manager for the SqliteSaver
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        # Create graph with checkpointer
        print("Creating LangGraph with checkpointing...")
        graph = create_graph(llama_llm, llama_tokenizer, qwen_llm, qwen_tokenizer, checkpointer)
        print("Graph created successfully!")

        # Save visualization
        print("\nSaving graph visualization...")
        save_graph_image(graph)

        # Config with thread_id for conversation tracking
        config = {"configurable": {"thread_id": thread_id}}

        # Initial state (only used if no checkpoint exists)
        initial_state: AgentState = {
            "chat_history": [],
            "user_input": "",
            "should_exit": False,
            "last_speaker": ""
        }

        print("\n" + "-" * 60)
        if has_existing:
            print("Resuming multi-participant conversation...")
            print("(Your previous messages have been restored)")
        else:
            print("Starting new multi-participant conversation...")
        print("-" * 60)

        try:
            # The graph will automatically resume from checkpoint if one exists
            graph.invoke(initial_state, config)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Your conversation has been saved.")
            print(f"   Run the program again to resume from where you left off.")
            print(f"   Or run with --new to start fresh.\n")


if __name__ == "__main__":
    main()
