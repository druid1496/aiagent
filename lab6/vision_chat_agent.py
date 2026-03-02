# vision_chat_agent.py
# A LangGraph-based Vision-Language Chat Agent
#
# This agent allows multi-turn conversations about an uploaded image.
# It uses Ollama with LLaVA (or similar vision model) and maintains
# full conversation history with checkpointing for crash recovery.
#
# Features:
# - Upload an image and discuss it across multiple turns
# - Full conversation history maintained
# - Checkpointing for crash recovery
# - Image resolution reduction option for performance
#
# Graph structure:
#   START -> get_user_input -> [conditional] -> call_vision_llm -> print_response -+
#                  ^                 |                                              |
#                  |                 +-> END (if user wants to quit)                |
#                  +----------------------------------------------------------------+

import os
import sys
import base64
from typing import TypedDict, Annotated, Sequence, Literal, Optional
import operator

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Ollama for vision model
import ollama

# PIL for image processing (resolution reduction)
from PIL import Image
import io

# =============================================================================
# CONFIGURATION
# =============================================================================

verbose_flag = True

# Checkpoint database
CHECKPOINT_DB = "vision_chat_checkpoints.db"
DEFAULT_THREAD_ID = "vision_conversation"

# Default vision model
VISION_MODEL = "llava"  # Can also use "llava:13b", "bakllava", etc.

# System prompt for the vision model
SYSTEM_PROMPT = """You are a helpful AI assistant with vision capabilities.
You can see and discuss the image that was uploaded at the start of the conversation.
You remember all previous messages in our conversation about this image.
Be helpful, observant, and provide detailed answers about what you see in the image."""


# =============================================================================
# STATE DEFINITION
# =============================================================================

class ChatMessage(TypedDict):
    """A message in the conversation."""
    role: Literal["user", "assistant", "system"]
    content: str
    # Images are only included in the first message
    images: Optional[list]


class VisionAgentState(TypedDict):
    """
    State for the vision chat agent.
    
    Fields:
    - messages: Conversation history (accumulates via operator.add)
    - image_path: Path to the uploaded image
    - image_base64: Base64-encoded image data (for persistence)
    - user_input: Current user input
    - should_exit: Flag to exit the conversation
    - image_loaded: Whether an image has been loaded
    """
    messages: Annotated[Sequence[ChatMessage], operator.add]
    image_path: str
    image_base64: str
    user_input: str
    should_exit: bool
    image_loaded: bool


# =============================================================================
# IMAGE UTILITIES
# =============================================================================

def load_and_resize_image(image_path: str, max_size: int = 1024) -> tuple[str, str]:
    """
    Load an image and optionally resize it for better performance.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) in pixels
    
    Returns:
        Tuple of (resized_path, base64_data)
    """
    global verbose_flag
    
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Open and check image size
    img = Image.open(image_path)
    original_size = img.size
    
    if verbose_flag:
        print(f"📷 Original image size: {original_size[0]}x{original_size[1]}")
    
    # Resize if needed
    if max(original_size) > max_size:
        # Calculate new size maintaining aspect ratio
        ratio = max_size / max(original_size)
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        if verbose_flag:
            print(f"📐 Resized to: {new_size[0]}x{new_size[1]} for better performance")
        
        # Save resized image to temp file
        resized_path = image_path.rsplit('.', 1)[0] + '_resized.jpg'
        img.save(resized_path, 'JPEG', quality=85)
        working_path = resized_path
    else:
        working_path = image_path
    
    # Convert to base64 for persistence
    with open(working_path, 'rb') as f:
        image_data = f.read()
    base64_data = base64.b64encode(image_data).decode('utf-8')
    
    return working_path, base64_data


def get_image_for_ollama(state: VisionAgentState) -> list:
    """Get the image in the format Ollama expects."""
    # Return the image path for Ollama
    if state.get("image_path") and os.path.exists(state["image_path"]):
        return [state["image_path"]]
    return []


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================

def check_existing_conversation(db_path: str, thread_id: str) -> bool:
    """Check if there's an existing conversation."""
    if not os.path.exists(db_path):
        return False
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM checkpoints 
            WHERE thread_id = ?
        """, (thread_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    except Exception:
        return False


def clear_conversation(db_path: str, thread_id: str):
    """Clear the conversation for a given thread_id."""
    if not os.path.exists(db_path):
        return
    
    try:
        import sqlite3
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

def create_graph(checkpointer):
    """
    Create the LangGraph for vision chat.
    
    Nodes:
    1. get_user_input: Read user's question about the image
    2. call_vision_llm: Send image + history to vision model
    3. print_response: Display the response
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: VisionAgentState) -> dict:
        global verbose_flag
        
        # Count turns
        user_msgs = [m for m in state["messages"] if m["role"] == "user"]
        turn_count = len(user_msgs) + 1
        
        # Show image info on first turn
        if turn_count == 1 and state.get("image_loaded"):
            print(f"\n🖼️  Image loaded: {state.get('image_path', 'Unknown')}")
        
        print("\n" + "=" * 60)
        print(f"Turn {turn_count} - Ask about the image (or 'quit' to exit):")
        print("💾 Conversation is saved automatically")
        print("=" * 60)
        
        print("\n> ", end="")
        try:
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            print("\n\n⚠️  Interrupted! Conversation saved.")
            return {"user_input": "quit", "should_exit": True, "messages": []}
        
        # Handle commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Conversation saved.")
            return {"user_input": user_input, "should_exit": True, "messages": []}
        
        if user_input == "verbose":
            verbose_flag = True
            print("Verbose mode ON")
            return {"user_input": "", "should_exit": False, "messages": []}
        elif user_input == "quiet":
            verbose_flag = False
            print("Verbose mode OFF")
            return {"user_input": "", "should_exit": False, "messages": []}
        elif user_input == "history":
            print("\n📜 Conversation History:")
            for i, msg in enumerate(state["messages"]):
                role_icon = {"user": "👤", "assistant": "🤖", "system": "⚙️"}.get(msg["role"], "?")
                content = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                has_image = "🖼️" if msg.get("images") else ""
                print(f"  {i+1}. {role_icon} {msg['role']}: {content} {has_image}")
            return {"user_input": "", "should_exit": False, "messages": []}
        
        if not user_input.strip():
            return {"user_input": "", "should_exit": False, "messages": []}
        
        # Create user message
        user_message: ChatMessage = {
            "role": "user",
            "content": user_input,
            "images": None
        }
        
        return {
            "user_input": user_input,
            "should_exit": False,
            "messages": [user_message]
        }

    # =========================================================================
    # NODE 2: call_vision_llm
    # =========================================================================
    def call_vision_llm(state: VisionAgentState) -> dict:
        global verbose_flag
        
        if verbose_flag:
            print(f"\n🔍 Processing with {VISION_MODEL}...")
            print(f"   (Context: {len(state['messages'])} messages)")
        
        # Build messages for Ollama
        # The image should be included with the first user message
        ollama_messages = []
        image_included = False
        images = get_image_for_ollama(state)
        
        for msg in state["messages"]:
            ollama_msg = {
                "role": msg["role"],
                "content": msg["content"]
            }
            
            # Include image with first user message
            if msg["role"] == "user" and not image_included and images:
                ollama_msg["images"] = images
                image_included = True
            
            ollama_messages.append(ollama_msg)
        
        if verbose_flag:
            print(f"   Image included: {'Yes' if image_included else 'No'}")
        
        try:
            # Call Ollama vision model
            response = ollama.chat(
                model=VISION_MODEL,
                messages=ollama_messages
            )
            
            assistant_content = response['message']['content']
            
        except Exception as e:
            print(f"\n❌ Error calling vision model: {e}")
            print("   Make sure Ollama is running and llava is installed:")
            print("   $ ollama pull llava")
            assistant_content = f"Error: Could not get response from vision model. {str(e)}"
        
        if verbose_flag:
            print("🤖 Vision model: Done!")
        
        # Create assistant message
        assistant_message: ChatMessage = {
            "role": "assistant",
            "content": assistant_content,
            "images": None
        }
        
        return {
            "messages": [assistant_message]
        }

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    def print_response(state: VisionAgentState) -> dict:
        # Get latest assistant message
        assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
        if assistant_msgs:
            latest = assistant_msgs[-1]["content"]
        else:
            latest = "(No response)"
        
        print("\n" + "=" * 60)
        print(f"🤖 {VISION_MODEL} Response:")
        print("=" * 60)
        print(latest)
        print("=" * 60)
        print("💾 State saved")
        
        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: VisionAgentState) -> str:
        if state.get("should_exit", False):
            return END
        if not state.get("user_input"):
            return "get_user_input"
        return "call_vision_llm"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(VisionAgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_vision_llm", call_vision_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_vision_llm": "call_vision_llm",
            "get_user_input": "get_user_input",
            END: END
        }
    )

    graph_builder.add_edge("call_vision_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    # Compile with checkpointer
    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph


def save_graph_image(graph, filename="vision_graph.png"):
    """Save graph visualization."""
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph saved to {filename}")
    except Exception as e:
        print(f"Could not save graph: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main function for the Vision-Language Chat Agent.
    """
    print("=" * 60)
    print("🖼️  Vision-Language Chat Agent")
    print("=" * 60)
    print()
    print("Features:")
    print("  - Multi-turn conversation about an image")
    print("  - Conversation history maintained")
    print("  - Checkpointing for crash recovery")
    print("  - Image resolution reduction for performance")
    print()
    
    # Parse arguments
    image_path = None
    start_fresh = "--new" in sys.argv
    max_resolution = 1024  # Default max resolution
    
    # Find image path argument
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--"):
            if arg.startswith("--resolution="):
                max_resolution = int(arg.split("=")[1])
            continue
        if not image_path and os.path.exists(arg):
            image_path = arg
    
    # Default image
    if not image_path:
        # Try default photo.jpg in same directory
        default_path = os.path.join(os.path.dirname(__file__), "photo.jpg")
        if os.path.exists(default_path):
            image_path = default_path
        else:
            print("❌ No image specified!")
            print("Usage: python vision_chat_agent.py <image_path> [--new] [--resolution=1024]")
            print("Example: python vision_chat_agent.py photo.jpg")
            sys.exit(1)
    
    thread_id = DEFAULT_THREAD_ID
    
    # Check for existing conversation
    has_existing = check_existing_conversation(CHECKPOINT_DB, thread_id)
    
    if has_existing and not start_fresh:
        print(f"📂 Found existing conversation")
        print("   Options:")
        print("   - Press Enter to RESUME")
        print("   - Type 'new' to start FRESH with new image")
        
        choice = input("\n> ").strip().lower()
        if choice == "new":
            clear_conversation(CHECKPOINT_DB, thread_id)
            has_existing = False
    elif start_fresh and has_existing:
        clear_conversation(CHECKPOINT_DB, thread_id)
        has_existing = False
    
    # Load and optionally resize image
    print(f"\n📷 Loading image: {image_path}")
    try:
        working_path, image_base64 = load_and_resize_image(image_path, max_resolution)
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        sys.exit(1)
    
    # Check Ollama is running
    print(f"\n🔌 Checking connection to Ollama ({VISION_MODEL})...")
    try:
        ollama.list()
        print("✅ Ollama is running")
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Please start Ollama: ollama serve")
        print(f"   And pull the model: ollama pull {VISION_MODEL}")
        sys.exit(1)
    
    print()
    print("Commands:")
    print("  'history' - Show conversation history")
    print("  'quit'    - Exit (conversation saved)")
    print("  Ctrl+C    - Exit (conversation saved)")
    
    # Create checkpointer and graph
    print(f"\n💾 Using checkpoint database: {CHECKPOINT_DB}")
    
    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        graph = create_graph(checkpointer)
        print("✅ Graph created")
        
        # Save visualization
        save_graph_image(graph)
        
        # Config with thread ID
        config = {"configurable": {"thread_id": thread_id}}
        
        # Initial state
        initial_state: VisionAgentState = {
            "messages": [],
            "image_path": working_path,
            "image_base64": image_base64,
            "user_input": "",
            "should_exit": False,
            "image_loaded": True
        }
        
        print("\n" + "-" * 60)
        if has_existing:
            print("Resuming conversation about the image...")
        else:
            print("Starting new conversation about the image...")
        print("-" * 60)
        
        try:
            graph.invoke(initial_state, config)
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted! Conversation saved.")
            print("   Run again to resume.")


if __name__ == "__main__":
    main()
