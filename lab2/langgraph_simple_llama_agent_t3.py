# langgraph_simple_agent.py
# Program demonstrates use of LangGraph for a very simple agent with PARALLEL LLM execution.
# It writes to stdout and asks the user to enter a line of text through stdin.
# It passes the line to BOTH Llama-3.2-1B-Instruct AND Qwen2.5-0.5B in PARALLEL,
# then prints what both LLMs return.
# The LLM should use Cuda if available, if not then if mps is available then use that,
# otherwise use cpu.
# After the LangGraph graph is created but before it executes, the program
# uses the Mermaid library to write a image of the graph to the file lg_graph.png
#
# Graph structure with parallel LLM nodes:
#   get_user_input -> fan_out -> call_llama ---+-> combine_and_print -> get_user_input
#                             -> call_qwen  ---+
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
    - llama_response: The response from Llama model (set by call_llama node)
    - qwen_response: The response from Qwen model (set by call_qwen node)

    State Flow with Parallel Execution:
    1. Initial state: all fields empty/default
    2. After get_user_input: user_input and should_exit are populated
    3. After fan_out: passes input to both LLM nodes (no state change)
    4. After call_llama & call_qwen (PARALLEL): llama_response and qwen_response populated
    5. After combine_and_print: both responses printed

    The graph loops continuously with parallel LLM execution:
        get_user_input -> [conditional] -> fan_out -> call_llama ---+-> combine_and_print -> get_user_input
                              |                    -> call_qwen  ---+
                              +-> END (if user wants to quit)
    """
    user_input: str
    should_exit: bool
    llama_response: str
    qwen_response: str

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
    Create the LangGraph state graph with PARALLEL LLM nodes:
    1. get_user_input: Reads input from stdin
    2. fan_out: Passes input to both LLM nodes (triggers parallel execution)
    3. call_llama: Calls Llama model (runs in parallel with call_qwen)
    4. call_qwen: Calls Qwen model (runs in parallel with call_llama)
    5. combine_and_print: Waits for both LLMs, prints both results

    Graph structure with PARALLEL LLM execution:
        START -> get_user_input -> [conditional] -> fan_out -> call_llama ---+-> combine_and_print -+
                       ^                 |                  -> call_qwen  ---+                      |
                       |                 +-> END (if user wants to quit)                            |
                       |                                                                            |
                       +----------------------------------------------------------------------------+

    The graph runs continuously until the user types 'quit', 'exit', or 'q'.
    LangGraph automatically executes call_llama and call_qwen in PARALLEL
    because they both receive edges from fan_out and both send edges to combine_and_print.
    """

    # =========================================================================
    # NODE 1: get_user_input
    # =========================================================================
    def get_user_input(state: AgentState) -> dict:
        global verbose_flag
        """
        Node that prompts the user for input via stdin.
        """
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

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
    # NODE 2: fan_out
    # =========================================================================
    # This node simply passes the input forward to trigger parallel execution.
    # LangGraph will execute all nodes that receive edges from this node in parallel.
    def fan_out(state: AgentState) -> dict:
        """
        Fan-out node that triggers parallel execution of both LLM nodes.
        
        This node doesn't modify state - it simply passes through to trigger
        the parallel execution of call_llama and call_qwen nodes.
        """
        global verbose_flag
        if verbose_flag:
            print("\n🔀 Fan-out: Sending input to BOTH LLMs in parallel...")
        
        return {}

    # =========================================================================
    # NODE 3: call_llama (PARALLEL BRANCH 1)
    # =========================================================================
    def call_llama(state: AgentState) -> dict:
        """
        Node that invokes Llama model with the user's input.
        Runs in PARALLEL with call_qwen node.
        """
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"
        
        global verbose_flag
        if verbose_flag:
            print("\n🦙 Llama: Processing...")

        response = llm.invoke(prompt)

        if verbose_flag:
            print("🦙 Llama: Done!")
        
        return {"llama_response": response}

    # =========================================================================
    # NODE 4: call_qwen (PARALLEL BRANCH 2)
    # =========================================================================
    def call_qwen(state: AgentState) -> dict:
        """
        Node that invokes Qwen model with the user's input.
        Runs in PARALLEL with call_llama node.
        """
        user_input = state["user_input"]
        prompt = f"User: {user_input}\nAssistant:"
        
        global verbose_flag
        if verbose_flag:
            print("\n🤖 Qwen: Processing...")

        response = llm2.invoke(prompt)

        if verbose_flag:
            print("🤖 Qwen: Done!")
        
        return {"qwen_response": response}

    # =========================================================================
    # NODE 5: combine_and_print
    # =========================================================================
    # This node waits for BOTH parallel LLM calls to complete, then prints both results.
    def combine_and_print(state: AgentState) -> dict:
        """
        Node that receives both LLM responses and prints them.
        LangGraph automatically waits for ALL incoming edges (both LLMs) before executing.
        """
        print("\n" + "=" * 60)
        print("📊 PARALLEL LLM RESULTS")
        print("=" * 60)
        
        print("\n" + "-" * 60)
        print("🦙 Llama-3.2-1B-Instruct Response:")
        print("-" * 60)
        print(state.get("llama_response", "(No response)"))
        
        print("\n" + "-" * 60)
        print("🤖 Qwen2.5-0.5B Response:")
        print("-" * 60)
        print(state.get("qwen_response", "(No response)"))
        
        print("\n" + "=" * 60)

        return {}

    # =========================================================================
    # ROUTING FUNCTION
    # =========================================================================
    def route_after_input(state: AgentState) -> str:
        """
        Routing function that determines the next node based on state.
        """
        if state.get("should_exit", False):
            return END
        if state["user_input"] == "":
            return "get_user_input"

        return "fan_out"

    # =========================================================================
    # GRAPH CONSTRUCTION
    # =========================================================================
    graph_builder = StateGraph(AgentState)

    # Add all nodes to the graph
    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("fan_out", fan_out)
    graph_builder.add_node("call_llama", call_llama)
    graph_builder.add_node("call_qwen", call_qwen)
    graph_builder.add_node("combine_and_print", combine_and_print)

    # Define edges:
    # 1. START -> get_user_input
    graph_builder.add_edge(START, "get_user_input")

    # 2. get_user_input -> [conditional] -> fan_out OR END OR loop back
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "fan_out": "fan_out",
            "get_user_input": "get_user_input",
            END: END
        }
    )

    # 3. fan_out -> call_llama AND call_qwen (PARALLEL EXECUTION)
    #    Both edges from fan_out trigger parallel execution
    graph_builder.add_edge("fan_out", "call_llama")
    graph_builder.add_edge("fan_out", "call_qwen")

    # 4. call_llama -> combine_and_print
    # 5. call_qwen -> combine_and_print
    #    LangGraph waits for BOTH incoming edges before executing combine_and_print
    graph_builder.add_edge("call_llama", "combine_and_print")
    graph_builder.add_edge("call_qwen", "combine_and_print")

    # 6. combine_and_print -> get_user_input (loop back for next input)
    graph_builder.add_edge("combine_and_print", "get_user_input")

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
    Main function that orchestrates the parallel LLM agent workflow:
    1. Initialize BOTH LLMs (Llama and Qwen)
    2. Create the LangGraph with parallel execution nodes
    3. Save the graph visualization
    4. Run the graph (loops internally until user quits)

    The graph handles all looping internally through its edge structure:
    - get_user_input: Prompts and reads from stdin
    - fan_out: Triggers parallel LLM execution
    - call_llama & call_qwen: Run in PARALLEL
    - combine_and_print: Waits for both, prints results, loops back

    The graph only terminates when the user types 'quit', 'exit', or 'q'.
    """
    print("=" * 60)
    print("LangGraph PARALLEL Agent: Llama + Qwen")
    print("=" * 60)
    print()

    # Step 1: Create and configure BOTH LLMs
    print("Loading Llama-3.2-1B-Instruct...")
    llm = create_llm()
    
    print("\nLoading Qwen2.5-0.5B...")
    llm2 = create_llm(model_id="Qwen/Qwen2.5-0.5B")

    # Step 2: Build the LangGraph with parallel LLM nodes
    print("\nCreating LangGraph with PARALLEL LLM nodes...")
    graph = create_graph(llm, llm2)
    print("Graph created successfully!")

    # Step 3: Save a visual representation of the graph before execution
    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Step 4: Run the graph - it will loop internally until user quits
    # The graph executes call_llama and call_qwen in PARALLEL
    initial_state: AgentState = {
        "user_input": "",
        "should_exit": False,
        "llama_response": "",
        "qwen_response": ""
    }

    # Single invocation - the graph loops internally
    # Parallel execution happens automatically when both LLM nodes receive input from fan_out
    graph.invoke(initial_state)

# Entry point - only run main() if this script is executed directly
if __name__ == "__main__":
    main()
