"""
Tool Calling with LangGraph
Shows how LangGraph orchestrates tool calling with nodes, edges, checkpointing, and recovery.
"""

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
import json
import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph.message import add_messages
import math
import ast
import json
import os
import signal

# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")


@tool
def calculator(geometric_type: str, params_list: list[float]) -> float:
    """Calculate the area of a geometric shape.
    The geometric type can be circle, rectangle, or triangle. 
    The params_list is a list of floats that represent the parameters of the geometric shape.
    If params_list is a string, it will be safely parsed using ast.literal_eval."""
    # Safely parse params_list if it's a string
    if isinstance(params_list, str):
        try:
            params_list = ast.literal_eval(params_list)
        except (ValueError, SyntaxError) as e:
            return f"Error: Could not parse params_list: {e}"
    
    # Ensure params_list is a list
    if not isinstance(params_list, list):
        return f"Error: params_list must be a list, got {type(params_list)}"
    
    # Convert to floats
    try:
        params_list = [float(p) for p in params_list]
    except (ValueError, TypeError) as e:
        return f"Error: Could not convert params to numbers: {e}"
    
    if geometric_type == "circle":
        return math.pi * params_list[0] ** 2
    elif geometric_type == "rectangle":
        return params_list[0] * params_list[1]
    elif geometric_type == "triangle":
        return 0.5 * params_list[0] * params_list[1]
    else:
        return f"Error: Unknown geometric type {geometric_type}"


@tool
def count_letter(text: str, letter: str) -> int:
    """Count the number of occurrences of a specific letter (case-insensitive) in a piece of text.
    For example, to count how many 's' letters are in 'Mississippi riverboats'."""
    if len(letter) != 1:
        return f"Error: letter must be a single character, got '{letter}'"
    
    # Case-insensitive count
    text_lower = text.lower()
    letter_lower = letter.lower()
    count = text_lower.count(letter_lower)
    return count


@tool
def word_statistics(text: str) -> str:
    """Analyze text and return statistics including word count, character count, average word length, and most common words.
    Useful for analyzing documents, articles, or any text content."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
    
    # Count word frequencies (case-insensitive)
    word_freq = {}
    for word in words:
        word_clean = word.lower().strip(".,!?;:\"()[]{}")
        if word_clean:
            word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
    
    # Get top 5 most common words
    most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    
    stats = {
        "word_count": word_count,
        "character_count": char_count,
        "character_count_no_spaces": char_count_no_spaces,
        "average_word_length": round(avg_word_length, 2),
        "most_common_words": most_common
    }
    
    return json.dumps(stats)


# Create tools list
tools = [get_weather, calculator, count_letter, word_statistics]
tool_map = {tool.name: tool for tool in tools}


# ============================================
# PART 2: Define State and Graph
# ============================================

class AgentState(TypedDict):
    messages: Annotated[Sequence, add_messages]


# Create LLM with tools
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)

# Create tool node
tool_node = ToolNode(tools)


# ============================================
# PART 3: Define Graph Nodes
# ============================================

def get_user_input(state: AgentState) -> AgentState:
    """Node to get user input from command line"""
    user_input = input("\nYou: ").strip()
    
    # Handle empty input - loop back to get_user_input
    if not user_input:
        print("Please enter a non-empty message.")
        return state
    
    # Handle special commands - add exit message to trigger end
    if user_input.lower() == "exit" or user_input.lower() == "quit":
        print("Goodbye!")
        # Add a special exit message that will be caught by router
        return {**state, "messages": [*state["messages"], HumanMessage(content="__EXIT__")]}
    
    # Add user message to state
    return {
        "messages": [*state["messages"], HumanMessage(content=user_input)]
    }


def call_llm(state: AgentState) -> AgentState:
    """Node to call the LLM with tools"""
    messages = state["messages"]
    
    # Check for exit - don't call LLM if user wants to exit
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, HumanMessage) and last_msg.content == "__EXIT__":
            return state  # Don't call LLM, just return state
    
    if not messages:
        return state
    
    print("\n[Agent] Processing...")
    
    # Add system message if not present
    if not any(isinstance(msg, SystemMessage) for msg in messages):
        messages = [SystemMessage(content="You are a helpful assistant. Use the provided tools when needed.")] + list(messages)
    
    # Call LLM
    response = llm_with_tools.invoke(messages)
    
    # Print tool calls if any
    if hasattr(response, 'tool_calls') and response.tool_calls:
        print(f"[Agent] Using {len(response.tool_calls)} tool(s): {[tc['name'] for tc in response.tool_calls]}")
    
    return {"messages": [*state["messages"], response]}


def should_continue(state: AgentState) -> str:
    """Router function to decide next step"""
    messages = state["messages"]
    if not messages:
        return "end"
    
    last_message = messages[-1]
    
    # Check for exit command
    if isinstance(last_message, HumanMessage) and last_message.content == "__EXIT__":
        return "end"
    
    # Check if last message has tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    else:
        return "end"


def print_response(state: AgentState) -> AgentState:
    """Node to print the final response"""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check for exit command
    if isinstance(last_message, HumanMessage) and last_message.content.lower() in ["exit", "quit"]:
        return state  # Don't print, just return to exit
    
    if hasattr(last_message, 'content') and last_message.content:
        print(f"\n[Assistant]: {last_message.content}")
    
    return state


# ============================================
# PART 4: Build the Graph
# ============================================

# Create a persistent checkpointer wrapper that saves to file
class PersistentMemorySaver(MemorySaver):
    """MemorySaver that also persists to a JSON file"""
    def __init__(self, checkpoint_file=".checkpoints.json"):
        super().__init__()
        self.checkpoint_file = checkpoint_file
        self._load_from_file()
    
    def _load_from_file(self):
        """Load checkpoints from file on startup"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r') as f:
                    data = json.load(f)
                    # Restore checkpoints to memory
                    for thread_id, checkpoints in data.items():
                        for checkpoint_id, checkpoint_data in checkpoints.items():
                            config = {"configurable": {"thread_id": thread_id}}
                            # Reconstruct checkpoint structure
                            # Note: This is a simplified version - full implementation would
                            # need to properly serialize/deserialize LangGraph checkpoint format
                            pass  # MemorySaver will handle in-memory storage
            except Exception as e:
                print(f"[Warning] Could not load checkpoints: {e}")
    
    def _save_to_file(self):
        """Save current checkpoints to file"""
        # This is a simplified version - in practice, we'd need to serialize
        # the full checkpoint structure. For now, we'll use a different approach.
        pass

# Use MemorySaver with manual file-based persistence for state
shared_checkpointer = MemorySaver()
checkpoint_file = ".conversation_state.json"

def save_conversation_state(thread_id, messages):
    """Save conversation state to file"""
    state_data = {
        "thread_id": thread_id,
        "messages": [
            {
                "type": type(msg).__name__,
                "content": msg.content if hasattr(msg, 'content') else str(msg),
                "tool_calls": getattr(msg, 'tool_calls', None)
            }
            for msg in messages
        ]
    }
    # Load existing states
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                all_states = json.load(f)
        except:
            all_states = {}
    else:
        all_states = {}
    
    all_states[thread_id] = state_data
    
    with open(checkpoint_file, 'w') as f:
        json.dump(all_states, f, indent=2, default=str)

def load_conversation_state(thread_id):
    """Load conversation state from file"""
    if not os.path.exists(checkpoint_file):
        return None
    
    try:
        with open(checkpoint_file, 'r') as f:
            all_states = json.load(f)
        return all_states.get(thread_id)
    except:
        return None

def build_graph():
    """Build and compile the LangGraph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("get_user_input", get_user_input)
    workflow.add_node("call_llm", call_llm)
    workflow.add_node("tools", tool_node)
    workflow.add_node("print_response", print_response)
    
    # Set entry point
    workflow.set_entry_point("get_user_input")
    
    # Add edges
    workflow.add_edge("get_user_input", "call_llm")
    workflow.add_conditional_edges(
        "call_llm",
        should_continue,
        {
            "tools": "tools",
            "end": "print_response"
        }
    )
    workflow.add_edge("tools", "call_llm")  # After tools, call LLM again
    
    # Conditional edge from print_response - check if we should exit
    def should_continue_conversation(state: AgentState) -> str:
        messages = state["messages"]
        if messages:
            last_msg = messages[-1]
            if isinstance(last_msg, HumanMessage) and last_msg.content == "__EXIT__":
                return "end"
        return "continue"
    
    workflow.add_conditional_edges(
        "print_response",
        should_continue_conversation,
        {
            "continue": "get_user_input",
            "end": END
        }
    )
    
    return workflow.compile(checkpointer=shared_checkpointer)


# ============================================
# PART 5: Main Conversation Loop
# ============================================

def run_conversation(config: dict = None):
    """Run a persistent conversation with checkpointing"""
    graph = build_graph()
    
    # Create config for checkpointing
    if config is None:
        config = {"configurable": {"thread_id": "conversation-1"}}
    
    # Check for saved state in file first
    thread_id = config.get("configurable", {}).get("thread_id", "conversation-1")
    saved_state = load_conversation_state(thread_id)
    
    # Also check LangGraph's checkpointer
    existing_state = graph.get_state(config)
    
    # Determine initial state - prefer file-based, then LangGraph checkpoint
    messages = []
    if saved_state and saved_state.get("messages"):
        # Reconstruct messages from file
        for msg_data in saved_state["messages"]:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "HumanMessage" and content != "__EXIT__":
                messages.append(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                messages.append(AIMessage(content=content))
    elif existing_state.values and existing_state.values.get("messages"):
        # Use LangGraph checkpoint
        messages = [msg for msg in existing_state.values["messages"] 
                   if not (isinstance(msg, HumanMessage) and msg.content == "__EXIT__")]
    
    if messages:
        initial_state = {"messages": messages}
        print("="*60)
        print("Continuing existing conversation")
        print("="*60)
    else:
        # Start new conversation
        initial_state = {"messages": []}
        print("="*60)
        print("LangGraph Tool-Enabled Agent")
        print("="*60)
    
    print("Type 'exit' or 'quit' to end the conversation")
    print("The conversation state is checkpointed and can be recovered.")
    print("="*60)
    
    # Handle interrupt signal for recovery demo
    def signal_handler(sig, frame):
        print("\n\n[Interrupted] Conversation state saved. Restart the program to recover.")
        raise KeyboardInterrupt()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Run the graph - always use initial_state (either new or resumed)
        for event in graph.stream(initial_state, config, stream_mode="updates"):
            # Print node execution info
            for node_name, node_state in event.items():
                if node_name == "get_user_input":
                    # User input already printed
                    pass
                elif node_name == "call_llm":
                    # LLM processing message already printed
                    pass
                elif node_name == "tools":
                    # Tool execution info
                    pass
            
            # Save state after each update for persistence
            current_state = graph.get_state(config)
            if current_state.values and current_state.values.get("messages"):
                thread_id = config.get("configurable", {}).get("thread_id", "conversation-1")
                save_conversation_state(thread_id, current_state.values["messages"])
        
        # Check if we should exit
        final_state = graph.get_state(config)
        if final_state.values and final_state.values.get("messages"):
            last_msg = final_state.values["messages"][-1]
            if isinstance(last_msg, HumanMessage) and last_msg.content == "__EXIT__":
                return
    
    except KeyboardInterrupt:
        print("\n\n[Interrupted] Conversation state saved.")
        print("To recover, run: python langgraph-tool-handling_q5.py recover conversation-1")
    
    except Exception as e:
        print(f"\n[Error]: {e}")
        print("Conversation state has been saved and can be recovered.")


def recover_conversation(thread_id: str = "conversation-1"):
    """Recover a conversation from checkpoint"""
    # First try to load from file-based persistence
    saved_state = load_conversation_state(thread_id)
    
    if saved_state and saved_state.get("messages"):
        print("="*60)
        print("Recovered Conversation")
        print("="*60)
        print(f"Thread ID: {thread_id}")
        print(f"Messages in history: {len(saved_state['messages'])}")
        print("\nRecent conversation history:")
        for msg_data in saved_state["messages"][-5:]:  # Show last 5 messages
            msg_type = msg_data.get("type", "Unknown")
            content = msg_data.get("content", "")
            if msg_type == "HumanMessage":
                if content and content != "__EXIT__":
                    print(f"  [User]: {content[:100]}")
            elif msg_type == "AIMessage":
                if content:
                    print(f"  [Assistant]: {content[:100]}")
                else:
                    print(f"  [Assistant]: [Tool calls made]")
            elif msg_type == "ToolMessage":
                print(f"  [Tool]: {content[:50]}...")
        print("="*60)
        print("\nContinuing conversation...\n")
        
        # Reconstruct messages from saved state
        messages = []
        for msg_data in saved_state["messages"]:
            msg_type = msg_data.get("type")
            content = msg_data.get("content", "")
            if msg_type == "HumanMessage" and content != "__EXIT__":
                messages.append(HumanMessage(content=content))
            elif msg_type == "AIMessage":
                # For AIMessage, we'd need to reconstruct tool_calls if present
                # For simplicity, we'll just create a basic message
                msg = AIMessage(content=content)
                if msg_data.get("tool_calls"):
                    # Note: Full reconstruction would need proper tool_calls format
                    pass
                messages.append(msg)
        
        # Continue with reconstructed messages
        config = {"configurable": {"thread_id": thread_id}}
        run_conversation(config)
    else:
        # Fall back to checking LangGraph's checkpointer
        graph = build_graph()
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        
        if state.values and state.values.get("messages"):
            print("="*60)
            print("Recovered Conversation (from LangGraph checkpoint)")
            print("="*60)
            print(f"Thread ID: {thread_id}")
            print(f"Messages in history: {len(state.values['messages'])}")
            print("\nRecent conversation history:")
            for msg in state.values["messages"][-5:]:
                if isinstance(msg, HumanMessage):
                    content = msg.content[:100] if msg.content else "[Empty]"
                    if content != "__EXIT__":
                        print(f"  [User]: {content}")
                elif isinstance(msg, AIMessage):
                    content = msg.content[:100] if msg.content else "[Tool calls made]"
                    print(f"  [Assistant]: {content}")
                elif isinstance(msg, ToolMessage):
                    print(f"  [Tool]: {msg.content[:50]}...")
            print("="*60)
            print("\nContinuing conversation...\n")
            run_conversation(config)
        else:
            print("No saved conversation found. Starting new conversation.")
            run_conversation(config)


# ============================================
# PART 6: Main Entry Point
# ============================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "recover":
        thread_id = sys.argv[2] if len(sys.argv) > 2 else "conversation-1"
        recover_conversation(thread_id)
    else:
        run_conversation()
