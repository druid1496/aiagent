# langgraph_simple_agent.py
# Program demonstrates use of LangGraph with CHAT HISTORY using the Message API.
# It maintains conversation context across multiple turns using:
#   - SystemMessage: Sets the assistant's behavior
#   - HumanMessage: User's input
#   - AIMessage: Assistant's response
#
# The chat history is stored in the state and passed to the LLM each turn,
# allowing the model to reference previous messages in the conversation.
#
# Graph structure:
#   get_user_input -> [conditional] -> call_llm -> print_response -> get_user_input
#                          +-> END (if user wants to quit)
#
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
import operator

# Import LangChain Message types for chat history
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

# Determine the best available device for inference
# Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon) > CPU
def get_device():
    """
    Detect and return the best available compute device.
    Returns 'cuda' for NVIDIA GPUs, 'mps' for Apple Silicon, or 'cpu' as fallback.
    """
    global verbose_flag
    if verbose_flag:
        if torch.cuda.is_available():
            print("Using CUDA (NVIDIA GPU) for inference")
            return "cuda"
        elif torch.backends.mps.is_available():
            print("Using MPS (Apple Silicon) for inference")
            return "mps"
        else:
            print("Using CPU for inference")
            return "cpu"

# =============================================================================
# STATE DEFINITION
# =============================================================================
# The state uses the Message API to maintain chat history.
# The 'messages' field uses Annotated with operator.add to accumulate messages.

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - messages: List of chat messages (SystemMessage, HumanMessage, AIMessage)
                Uses Annotated[list, operator.add] to accumulate messages across nodes
    - user_input: The current text entered by the user
    - should_exit: Boolean flag indicating if user wants to quit

    Message Types (from langchain_core.messages):
    - SystemMessage: Sets the assistant's behavior/personality (role: "system")
    - HumanMessage: User's input (role: "human" or "user")  
    - AIMessage: Assistant's response (role: "ai" or "assistant")
    - ToolMessage: Tool/function results (role: "tool" or "function")

    State Flow with Chat History:
    1. Initial state: messages contains SystemMessage, other fields empty
    2. After get_user_input: HumanMessage appended to messages
    3. After call_llm: AIMessage appended to messages
    4. After print_response: response displayed
    5. Loop back - full message history is maintained for context
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_input: str
    should_exit: bool


# System prompt that sets the assistant's behavior
SYSTEM_PROMPT = """You are a helpful AI assistant. You are friendly, concise, and informative.
You remember the conversation history and can reference previous messages.
If the user asks about something mentioned earlier in the conversation, you should recall it."""


def create_llm(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    """
    Create and configure the LLM using HuggingFace's transformers library.
    Returns the tokenizer as well for formatting chat messages.
    """
    device = get_device()

    global verbose_flag
    if verbose_flag:
        print(f"Loading model: {model_id}")
        print("This may take a moment on first run as the model is downloaded...")

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
        print("Model loaded successfully!")
    
    return llm, tokenizer


verbose_flag = True


def format_messages_for_llm(messages: Sequence[BaseMessage], tokenizer) -> str:
    """
    Format the message history into a prompt string for the LLM.
    
    Uses the tokenizer's chat template if available, otherwise falls back
    to a simple text format.
    
    Args:
        messages: List of BaseMessage objects (SystemMessage, HumanMessage, AIMessage)
        tokenizer: The tokenizer with apply_chat_template method
    
    Returns:
        Formatted prompt string
    """
    # Convert LangChain messages to the format expected by apply_chat_template
    chat_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            chat_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            chat_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            chat_messages.append({"role": "assistant", "content": msg.content})
    
    # Use the tokenizer's chat template to format messages
    try:
        formatted = tokenizer.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    except Exception as e:
        # Fallback to simple format if chat template fails
        print(f"Warning: Could not apply chat template: {e}")
        result = ""
        for msg in chat_messages:
            role = msg["role"].capitalize()
            result += f"{role}: {msg['content']}\n"
        result += "Assistant:"
        return result


def create_graph(llm, tokenizer):
    """
    Create the LangGraph state graph with chat history support.
    
    Nodes:
    1. get_user_input: Reads input, creates HumanMessage
    2. call_llm: Processes full message history, creates AIMessage
    3. print_response: Displays the response

    Graph structure:
        START -> get_user_input -> [conditional] -> call_llm -> print_response -+
                       ^                 |                                       |
                       |                 +-> END (if user wants to quit)         |
                       +---------------------------------------------------------+
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        global verbose_flag
        """
        Node that prompts the user for input and creates a HumanMessage.
        """
        # Show conversation turn count
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        turn_count = len(human_msgs) + 1
        
        print("\n" + "=" * 60)
        print(f"Turn {turn_count} - Enter your text (or 'quit' to exit):")
        print("(Chat history is maintained - the model remembers context)")
        print("=" * 60)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True,
                "messages": []  # Empty list - nothing to add
            }
        
        if user_input == "verbose":
            verbose_flag = True
            return {"user_input": "", "should_exit": False, "messages": []}
        elif user_input == "quiet":
            verbose_flag = False
            return {"user_input": "", "should_exit": False, "messages": []}
        elif user_input == "history":
            # Debug command to show message history
            print("\n📜 Message History:")
            for i, msg in enumerate(state["messages"]):
                role = type(msg).__name__
                content = msg.content[:50] + "..." if len(msg.content) > 50 else msg.content
                print(f"  {i+1}. [{role}] {content}")
            return {"user_input": "", "should_exit": False, "messages": []}

        # Create a HumanMessage and add it to the message history
        human_message = HumanMessage(content=user_input)
        
        return {
            "user_input": user_input,
            "should_exit": False,
            "messages": [human_message]  # This will be appended to existing messages
        }

    # =========================================================================
    # NODE 2: call_llm
    # =========================================================================
    def call_llm(state: AgentState) -> dict:
        """
        Node that invokes the LLM with the full message history.
        Creates an AIMessage with the response.
        """
        global verbose_flag
        if verbose_flag:
            print("\n🦙 Processing with Llama-3.2-1B-Instruct...")
            print(f"   (Context: {len(state['messages'])} messages in history)")

        # Format all messages into a prompt
        prompt = format_messages_for_llm(state["messages"], tokenizer)
        
        if verbose_flag:
            print(f"   Prompt length: {len(prompt)} characters")

        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Clean up the response (remove the prompt echo if present)
        # The HuggingFace pipeline sometimes returns the full text including prompt
        if prompt in response:
            response = response.replace(prompt, "").strip()
        
        if verbose_flag:
            print("🦙 Llama: Done!")

        # Create AIMessage and add to history
        ai_message = AIMessage(content=response)
        
        return {
            "messages": [ai_message]  # This will be appended to existing messages
        }

    # =========================================================================
    # NODE 3: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """
        Node that prints the latest AI response.
        """
        # Get the last AI message
        ai_messages = [m for m in state["messages"] if isinstance(m, AIMessage)]
        if ai_messages:
            latest_response = ai_messages[-1].content
        else:
            latest_response = "(No response)"
        
        print("\n" + "=" * 60)
        print("🦙 Llama-3.2-1B-Instruct Response:")
        print("=" * 60)
        print(latest_response)
        print("=" * 60)

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines the next node.
        """
        if state.get("should_exit", False):
            return END
        if state["user_input"] == "":
            return "get_user_input"
        
        return "call_llm"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    # Add nodes
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    # Define edges
    graph_builder.add_edge(START, "get_user_input")

    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llm": "call_llm",
            "get_user_input": "get_user_input",
            END: END
        }
    )

    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    graph = graph_builder.compile()

    return graph


def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    """
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")


def main():
    """
    Main function that orchestrates the chat agent with message history.
    
    Features:
    - Maintains full conversation history using Message API
    - SystemMessage sets assistant behavior
    - HumanMessage for user input
    - AIMessage for assistant responses
    - Context is preserved across turns
    """
    print("=" * 60)
    print("LangGraph Chat Agent with Message History")
    print("=" * 60)
    print()
    print("Features:")
    print("  - Chat history is maintained across turns")
    print("  - The model can reference previous messages")
    print("  - Type 'history' to see the message log")
    print("  - Type 'quit' to exit")
    print()

    # Step 1: Create and configure the LLM (Llama only, Qwen disabled)
    print("Loading Llama-3.2-1B-Instruct...")
    llm, tokenizer = create_llm()

    # Step 2: Build the LangGraph
    print("\nCreating LangGraph with chat history support...")
    graph = create_graph(llm, tokenizer)
    print("Graph created successfully!")

    # Step 3: Save graph visualization
    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Step 4: Initialize state with SystemMessage
    # The SystemMessage sets the assistant's behavior for the entire conversation
    initial_state: AgentState = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "user_input": "",
        "should_exit": False,
    }

    print("\n" + "-" * 60)
    print("System prompt loaded. Starting conversation...")
    print("-" * 60)

    # Run the graph - it loops internally until user quits
    graph.invoke(initial_state)


# Entry point
if __name__ == "__main__":
    main()
