"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
import math
import ast
import json
import sys
from io import StringIO

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


# Create tools list and mapping
tools = [get_weather, calculator, count_letter, word_statistics]
tool_map = {tool.name: tool for tool in tools}


# ============================================
# PART 2: Create LLM with Tools
# ============================================

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Start conversation with user query
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed."),
        HumanMessage(content=user_query)
    ]
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = llm_with_tools.invoke(messages)
        
        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(response)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                # Execute the tool
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
            
            print()
            # Loop continues - LLM will see the tool results
            
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            return response.content
    
    return "Max iterations reached"


def run_agent_with_capture(user_query: str):
    """
    Run agent and capture all output to a string.
    Returns the output string and the result.
    """
    old_stdout = sys.stdout
    sys.stdout = captured_output = StringIO()
    
    try:
        result = run_agent(user_query)
        output = captured_output.getvalue()
        return output, result
    finally:
        sys.stdout = old_stdout


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires tool use
    print("="*60)
    print("TEST 1: Query requiring tool")
    print("="*60)
    run_agent("What's the weather like in San Francisco?")
    
    print("\n" + "="*60)
    print("TEST 2: Query not requiring tool")
    print("="*60)
    run_agent("Say hello!")
    
    print("\n" + "="*60)
    print("TEST 3: Multiple tool calls (weather)")
    print("="*60)
    run_agent("What's the weather in New York and London?")
    
    print("\n" + "="*60)
    print("TEST 4: Multiple letter counts (comparison)")
    print("="*60)
    run_agent("Are there more i's than s's in Mississippi riverboats?")
    
    print("\n" + "="*60)
    print("TEST 5: Multiple tool calls with calculation")
    print("="*60)
    run_agent("What is the difference between the number of i's and the number of s's in Mississippi riverboats?")
    
    print("\n" + "="*60)
    print("TEST 6: Complex multi-tool query (count + calculate)")
    print("="*60)
    run_agent("Calculate the area of a rectangle where the width is the number of 's' letters in 'Mississippi riverboats' and the height is the number of 'i' letters in the same text.")
    
    print("\n" + "="*60)
    print("TEST 7: All tools in one query")
    print("="*60)
    run_agent("Get the weather for San Francisco, then analyze that weather description to get word statistics, count how many 'y' letters are in the weather description, and calculate the area of a circle with radius equal to the word count from the statistics.")
    
    print("\n" + "="*60)
    print("TEST 8: Sequential chaining hitting 5-iteration limit")
    print("="*60)
    # This query is designed to force sequential tool calls that will hit the limit
    query = """Get the weather for Tokyo, then analyze that weather text to get word statistics. 
    Based on the word count from the statistics, count how many 'e' letters are in the weather description. 
    Then calculate the area of a circle with radius equal to that letter count. 
    Finally, calculate the area of a rectangle where the width is the word count and height is the letter count. 
    Then get weather for another city and repeat the same analysis."""
    
    # Run and capture output
    output, result = run_agent_with_capture(query)
    
    # Print to console
    print(output)
    print(f"Final Result: {result}\n")
    
    # Save output to portfolio file
    with open("sequential_chaining_trace.txt", "w") as f:
        f.write("="*60 + "\n")
        f.write("Sequential Chaining Test - 5 Iteration Limit\n")
        f.write("="*60 + "\n\n")
        f.write(f"Query: {query}\n\n")
        f.write(output)
        f.write(f"\nFinal Result: {result}\n")