# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a simple agent with CONDITIONAL LLM routing.
# It writes to stdout and asks the user to enter a line of text through stdin.
# Based on the user's input, it routes to EITHER Llama OR Qwen (not both):
#   - If input starts with "Hey Qwen" -> routes to Qwen2.5-0.5B
#   - Otherwise -> routes to Llama-3.2-1B-Instruct
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
#
# Graph structure with conditional LLM routing:
#   get_user_input -> [conditional] -> call_llama -> print_response -> get_user_input
#                          |        -> call_qwen  -> print_response -+
#                          +-> END (if user wants to quit)
#
# The code is commented in detail so a reader can understand each step.

# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

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
# The state is a TypedDict that flows through all nodes in the graph.
# Each node can read from and write to specific fields in the state.
# LangGraph automatically merges the returned dict from each node into the state.

class AgentState(TypedDict):
    """
    State object that flows through the LangGraph nodes.

    Fields:
    - user_input: The text entered by the user (set by get_user_input node)
    - should_exit: Boolean flag indicating if user wants to quit (set by get_user_input node)
    - llm_response: The response from the selected LLM (set by call_llama OR call_qwen)
    - model_used: Which model was used ("llama" or "qwen")

    State Flow with Conditional Routing:
    1. Initial state: all fields empty/default
    2. After get_user_input: user_input and should_exit are populated
    3. Conditional routing: if input starts with "Hey Qwen" -> call_qwen, else -> call_llama
    4. After call_llama OR call_qwen: llm_response and model_used populated
    5. After print_response: response printed

    The graph loops continuously with conditional LLM routing:
        get_user_input -> [conditional] -> call_llama -> print_response -> get_user_input
                              |         -> call_qwen  -> print_response -+
                              +-> END (if user wants to quit)
    """
    user_input: str
    should_exit: bool
    llm_response: str
    model_used: str

def create_llm(model_id="meta-llama/Llama-3.2-1B-Instruct"):
    """
    Create and configure the LLM using HuggingFace's transformers library.
    Downloads llama-3.2-1B-Instruct from HuggingFace Hub and wraps it
    for use with LangChain via HuggingFacePipeline.
    """
    # Get the optimal device for inference
    device = get_device()

    # Model identifier on HuggingFace Hub

    global verbose_flag
    if verbose_flag:
        print(f"Loading model: {model_id}")
        print("This may take a moment on first run as the model is downloaded...")

    # Load the tokenizer - converts text to tokens the model understands
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load the model itself with appropriate settings for the device
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device if device == "cuda" else None,
    )

    # Move model to MPS device if using Apple Silicon
    if device == "mps":
        model = model.to(device)

    # Create a text generation pipeline that combines model and tokenizer
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,  # Maximum tokens to generate in response
        do_sample=True,      # Enable sampling for varied responses
        temperature=0.7,     # Controls randomness (lower = more deterministic)
        top_p=0.95,          # Nucleus sampling parameter
        pad_token_id=tokenizer.eos_token_id,  # Suppress pad_token_id warning
    )

    # Wrap the HuggingFace pipeline for use with LangChain
    llm = HuggingFacePipeline(pipeline=pipe)

    
    if verbose_flag:
        print("Model loaded successfully!")
    return llm


verbose_flag = True
def create_graph(llm, llm2=None):
    """
    Create the LangGraph state graph with CONDITIONAL LLM routing:
    1. get_user_input: Reads input from stdin
    2. call_llama: Calls Llama model (if input does NOT start with "Hey Qwen")
    3. call_qwen: Calls Qwen model (if input DOES start with "Hey Qwen")
    4. print_response: Prints the response from whichever LLM was called

    Graph structure with CONDITIONAL LLM routing:
        START -> get_user_input -> [conditional] -> call_llama -> print_response -+
                       ^                 |      -> call_qwen  -> print_response  |
                       |                 +-> END (if user wants to quit)         |
                       |                                                         |
                       +---------------------------------------------------------+

    The graph runs continuously until the user types 'quit', 'exit', or 'q'.
    Routing is based on whether user input starts with "Hey Qwen".
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        global verbose_flag
        """
        Node that prompts the user for input via stdin.
        """
        print("\n" + "=" * 60)
        print("Enter your text (or 'quit' to exit):")
        print("Start with 'Hey Qwen' to use Qwen, otherwise Llama is used")
        print("=" * 60)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {
                "user_input": user_input,
                "should_exit": True
            }
        
        if user_input == "verbose":
            verbose_flag = True
        elif user_input == "quiet":
            verbose_flag = False

        return {
            "user_input": user_input,
            "should_exit": False
        }

    # =========================================================================
    # NODE 2: call_llama
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        """
        Node that invokes Llama model with the user's input.
        Called when input does NOT start with "Hey Qwen".
        """
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"
        
        global verbose_flag
        if verbose_flag:
            print("\n🦙 Routing to Llama-3.2-1B-Instruct...")

        response = llm.invoke(prompt)

        if verbose_flag:
            print("🦙 Llama: Done!")
        
        return {"llm_response": response, "model_used": "llama"}

    # =========================================================================
    # NODE 3: call_qwen
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        """
        Node that invokes Qwen model with the user's input.
        Called when input DOES start with "Hey Qwen".
        """
        user_input = state["user_input"]
        # Remove the "Hey Qwen" prefix for cleaner prompting
        if user_input.lower().startswith("hey qwen"):
            user_input = user_input[8:].strip()  # Remove "Hey Qwen" prefix
            if user_input.startswith(","):
                user_input = user_input[1:].strip()  # Remove comma if present
        
        prompt = f"User: {user_input}\nAssistant:"
        
        global verbose_flag
        if verbose_flag:
            print("\n🤖 Routing to Qwen2.5-0.5B...")

        response = llm2.invoke(prompt)

        if verbose_flag:
            print("🤖 Qwen: Done!")
        
        return {"llm_response": response, "model_used": "qwen"}

    # =========================================================================
    # NODE 4: print_response
    # =========================================================================
    def print_response(state: AgentState) -> dict:
        """
        Node that prints the LLM response.
        Works for both Llama and Qwen responses.
        """
        model_used = state.get("model_used", "unknown")
        
        print("\n" + "=" * 60)
        if model_used == "llama":
            print("🦙 Llama-3.2-1B-Instruct Response:")
        elif model_used == "qwen":
            print("🤖 Qwen2.5-0.5B Response:")
        else:
            print("LLM Response:")
        print("=" * 60)
        print(state.get("llm_response", "(No response)"))
        print("=" * 60)

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines which LLM to use based on user input.
        
        - If input starts with "Hey Qwen" -> route to Qwen
        - Otherwise -> route to Llama
        - Empty input -> loop back to get_user_input
        - Quit commands -> END
        """
        if state.get("should_exit", False):
            return END
        if state["user_input"] == "":
            return "get_user_input"
        
        # Check if input starts with "Hey Qwen" (case insensitive)
        if state["user_input"].lower().startswith("hey qwen"):
            return "call_qwen"
        
        # Default: route to Llama
        return "call_llama"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    # Add all nodes to the graph
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("print_response", print_response)

    # Define edges:
    # 1. START -> get_user_input
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> call_llama OR call_qwen OR END
    #    Routes based on whether input starts with "Hey Qwen"
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

    # 3. call_llama -> print_response
    graph_builder.add_edge("call_llama", "print_response")

    # 4. call_qwen -> print_response
    graph_builder.add_edge("call_qwen", "print_response")

    # 5. print_response -> get_user_input (loop back for next input)
    graph_builder.add_edge("print_response", "get_user_input")

    # Compile the graph into an executable form
    graph = graph_builder.compile()

    return graph

def save_graph_image(graph, filename="lg_graph.png"):
    """
    Generate a Mermaid diagram of the graph and save it as a PNG image.
    Uses the graph's built-in Mermaid export functionality.
    """
    try:
        # Get the Mermaid PNG representation of the graph
        # This requires the 'grandalf' package for rendering
        png_data = graph.get_graph(xray=True).draw_mermaid_png()

        # Write the PNG data to file
        with open(filename, "wb") as f:
            f.write(png_data)

        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")
        print("You may need to install additional dependencies: pip install grandalf")

def main():
    """
    Main function that orchestrates the conditional LLM agent workflow:
    1. Initialize BOTH LLMs (Llama and Qwen)
    2. Create the LangGraph with conditional routing
    3. Save the graph visualization
    4. Run the graph (loops internally until user quits)

    The graph handles all looping internally through its edge structure:
    - get_user_input: Prompts and reads from stdin
    - Conditional routing: "Hey Qwen" -> Qwen, otherwise -> Llama
    - call_llama OR call_qwen: Only ONE is called based on input
    - print_response: Prints result, loops back

    The graph only terminates when the user types 'quit', 'exit', or 'q'.
    """
    print("=" * 60)
    print("LangGraph CONDITIONAL Agent: Llama or Qwen")
    print("=" * 60)
    print()
    print("Routing rules:")
    print("  - Start with 'Hey Qwen' -> routes to Qwen2.5-0.5B")
    print("  - Otherwise -> routes to Llama-3.2-1B-Instruct")
    print()

    # Step 1: Create and configure BOTH LLMs
    print("Loading Llama-3.2-1B-Instruct...")
    llm = create_llm()
    
    print("\nLoading Qwen2.5-0.5B...")
    llm2 = create_llm(model_id="Qwen/Qwen2.5-0.5B")

    # Step 2: Build the LangGraph with conditional routing
    print("\nCreating LangGraph with CONDITIONAL routing...")
    graph = create_graph(llm, llm2)
    print("Graph created successfully!")

    # Step 3: Save a visual representation of the graph before execution
    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Step 4: Run the graph - it will loop internally until user quits
    # Only ONE LLM is called based on the user's input
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llm_response": "",
        "model_used": ""
    }

    # Single invocation - the graph loops internally
    # Conditional routing happens based on whether input starts with "Hey Qwen"
    graph.invoke(initial_state)

# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()
