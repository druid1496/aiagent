# langgraph_simple_agent.py
# Program demonstrates a MULTI-PARTICIPANT conversation with Llama and Qwen.
#
# The challenge: Chat APIs only support user/assistant/system roles, but we have
# THREE participants: Human, Llama, and Qwen.
#
# Solution: When formatting messages for an LLM, that LLM's messages become "assistant"
# role, while messages from the other participants (human + other LLM) become "user"
# role with name prefixes.
#
# Example for Qwen (after Human asks, Llama answers, then "Hey Qwen"):
#   [{"role": "user", "content": "Human: What is the best ice cream flavor?"},
#    {"role": "user", "content": "Llama: There is no one best flavor, but vanilla is popular."}]
#
# Example for Llama (after Qwen responds and Human agrees):
#   [{"role": "user", "content": "Human: What is the best ice cream flavor?"},
#    {"role": "assistant", "content": "There is no one best flavor, but vanilla is popular."},
#    {"role": "user", "content": "Qwen: No way, chocolate is the best!"},
#    {"role": "user", "content": "Human: I agree."}]
#
# Graph structure:
#   get_user_input -> [conditional] -> call_llama -> print_response -> get_user_input
#                          |        -> call_qwen  -> print_response -+
#                          +-> END

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence, Literal
import operator

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

verbose_flag = True

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

# Custom message type to track speaker identity
class ChatMessage(TypedDict):
    """
    A message in the multi-participant conversation.
    
    Fields:
    - speaker: Who said this ("human", "llama", or "qwen")
    - content: The message content
    """
    speaker: Literal["human", "llama", "qwen"]
    content: str


class AgentState(TypedDict):
    """
    State for the multi-participant conversation.
    
    Fields:
    - chat_history: List of ChatMessage dicts with speaker identity
    - user_input: Current user input text
    - should_exit: Flag to exit the conversation
    - last_speaker: Who spoke last ("llama" or "qwen") for print_response
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
    
    Args:
        chat_history: List of ChatMessage dicts
        target_llm: "llama" or "qwen" - determines which messages become "assistant"
        system_prompt: System prompt for this LLM
        tokenizer: Tokenizer for applying chat template
    
    Returns:
        Formatted prompt string
    """
    chat_messages = [{"role": "system", "content": system_prompt}]
    
    for msg in chat_history:
        speaker = msg["speaker"]
        content = msg["content"]
        
        if speaker == target_llm:
            # This LLM's own messages -> assistant role
            chat_messages.append({
                "role": "assistant",
                "content": content  # No prefix for own messages
            })
        else:
            # Other participants -> user role with name prefix
            name = speaker.capitalize() if speaker == "human" else speaker.capitalize()
            chat_messages.append({
                "role": "user",
                "content": f"{name}: {content}"
            })
    
    # Apply chat template
    try:
        formatted = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        print(f"Warning: Could not apply chat template: {e}")
        # Fallback
        result = f"System: {system_prompt}\n\n"
        for msg in chat_messages[1:]:  # Skip system
            role = msg["role"].capitalize()
            result += f"{role}: {msg['content']}\n"
        result += "Assistant:"
        return result


# =============================================================================
# GRAPH CREATION
# =============================================================================

def create_graph(llama_llm, llama_tokenizer, qwen_llm, qwen_tokenizer):
    """
    Create the LangGraph with multi-participant chat history.
    
    Nodes:
    1. get_user_input: Reads input, adds to chat_history
    2. call_llama: Processes history from Llama's perspective
    3. call_qwen: Processes history from Qwen's perspective  
    4. print_response: Displays the response

    Graph:
        START -> get_user_input -> [conditional] -> call_llama -> print_response -+
                       ^                 |      -> call_qwen  -> print_response  |
                       |                 +-> END                                  |
                       +----------------------------------------------------------+
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
        print("=" * 60)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
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

        # Format history from Llama's perspective
        prompt = format_history_for_llm(
            state["chat_history"],
            target_llm="llama",
            system_prompt=LLAMA_SYSTEM_PROMPT,
            tokenizer=llama_tokenizer
        )
        
        if verbose_flag:
            print(f"   Prompt length: {len(prompt)} characters")

        response = llama_llm.invoke(prompt)
        
        # Clean up response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        if verbose_flag:
            print("🦙 Llama: Done!")

        # Add Llama's response to history
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
        
        # Get the user input and strip "Hey Qwen" prefix for cleaner context
        user_input = state["user_input"]
        if user_input.lower().startswith("hey qwen"):
            clean_input = user_input[8:].strip()
            if clean_input.startswith(","):
                clean_input = clean_input[1:].strip()
            # Update the last human message in history to use clean input
            # But we keep the original in history for context
        
        if verbose_flag:
            print("\n🤖 Processing with Qwen2.5-0.5B...")
            print(f"   (Context: {len(state['chat_history'])} messages in history)")

        # Format history from Qwen's perspective
        prompt = format_history_for_llm(
            state["chat_history"],
            target_llm="qwen",
            system_prompt=QWEN_SYSTEM_PROMPT,
            tokenizer=qwen_tokenizer
        )
        
        if verbose_flag:
            print(f"   Prompt length: {len(prompt)} characters")

        response = qwen_llm.invoke(prompt)
        
        # Clean up response
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        if verbose_flag:
            print("🤖 Qwen: Done!")

        # Add Qwen's response to history
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
        
        # Get the last message from the speaker
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

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if state["user_input"] == "":
            return "get_user_input"
        
        # Route based on "Hey Qwen" prefix
        if state["user_input"].lower().startswith("hey qwen"):
            return "call_qwen"
        
        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
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

    graph = graph_builder.compile()

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
    Main function for multi-participant chat with Llama and Qwen.
    
    Features:
    - Both LLMs share a unified conversation history
    - Each LLM sees messages formatted appropriately for its perspective
    - Human and other LLM messages appear as "user" role with name prefixes
    - Own previous messages appear as "assistant" role
    """
    print("=" * 60)
    print("Multi-Participant Chat: Human + Llama + Qwen")
    print("=" * 60)
    print()
    print("Instructions:")
    print("  - Type normally to talk to Llama")
    print("  - Start with 'Hey Qwen' to talk to Qwen")
    print("  - Both AIs can see the full conversation history")
    print("  - Type 'history' to see the message log")
    print("  - Type 'quit' to exit")
    print()

    # Load both models
    print("Loading Llama-3.2-1B-Instruct...")
    llama_llm, llama_tokenizer = create_llm("meta-llama/Llama-3.2-1B-Instruct")
    
    print("\nLoading Qwen2.5-0.5B...")
    qwen_llm, qwen_tokenizer = create_llm("Qwen/Qwen2.5-0.5B")

    # Create graph
    print("\nCreating LangGraph with multi-participant support...")
    graph = create_graph(llama_llm, llama_tokenizer, qwen_llm, qwen_tokenizer)
    print("Graph created successfully!")

    # Save visualization
    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Initialize state with empty history
    initial_state: AgentState = {
        "chat_history": [],
        "user_input": "",
        "should_exit": False,
        "last_speaker": ""
    }

    print("\n" + "-" * 60)
    print("Starting multi-participant conversation...")
    print("-" * 60)

    graph.invoke(initial_state)


if __name__ == "__main__":
    main()
