time { python3 llama_mmlu_eval.py ; python3 llama_mmlu_eval2.py }
time { python3 llama_mmlu_eval.py & python3 llama_mmlu_eval2.py }



client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Say: Working!"}],
 max_tokens=5)

the first line will create a client to interact with the server in the cloud.
The second line will submit message to the selected model. And the maximum output token is 5. 

