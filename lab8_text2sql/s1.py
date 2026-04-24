"""Text-to-SQL fine-tuning with Tinker on Llama-3.2-1B.

Pipeline:
  1. Load b-mc2/sql-create-context from local JSON, split 200 test / rest train.
  2. Evaluate base model (execution-based).
  3. Tokenize training examples with prompt-masked loss.
  4. LoRA fine-tune for 1 epoch.
  5. Evaluate fine-tuned model on the same 200 test examples.
  6. Sample on 5 novel-schema questions (out-of-distribution).
"""

import json
import random
import numpy as np
import tinker
from tinker import types

from sql_matches import sql_matches

random.seed(0)

# =============================================================================
# Step 1: Load and split data
# =============================================================================

with open("sql_create_context_v4.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")
print("\nSample example:")
ex = data[0]
print(f"  Question: {ex['question']}")
print(f"  Context:  {ex['context'][:120]}...")
print(f"  Answer:   {ex['answer']}")

NUM_TEST_EXAMPLES = 200
random.shuffle(data)
test_data = data[:NUM_TEST_EXAMPLES]
train_data = data[NUM_TEST_EXAMPLES:]
print(f"\nTraining examples: {len(train_data)}")
print(f"Test examples:     {len(test_data)}")

# =============================================================================
# Step 2: Tinker client & tokenizer
# =============================================================================

service_client = tinker.ServiceClient()
BASE_MODEL = "meta-llama/Llama-3.2-1B"
training_client = service_client.create_lora_training_client(base_model=BASE_MODEL)
tokenizer = training_client.get_tokenizer()

# =============================================================================
# Step 3: Sampling + evaluation helpers
# =============================================================================

PROMPT_TEMPLATE = """Table schema:
{context}
Question: {question}
SQL: """


def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = types.ModelInput.from_ints(tokens=prompt_tokens)
    params = types.SamplingParams(
        max_tokens=150, temperature=0.0, stop=["\n\n", "Question:"]
    )
    result = sampling_client.sample(
        prompt=model_input, sampling_params=params, num_samples=1
    ).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""


def eval_one(sampling_client, tokenizer, ex: dict) -> bool:
    sql = sample_from_model(sampling_client, tokenizer, ex["context"], ex["question"])
    return sql_matches(sql, ex["answer"], schema=ex["context"])


def evaluate_test_set(sampling_client, tokenizer, test_data: list, label: str = "") -> float:
    correct = 0
    for i, ex in enumerate(test_data, 1):
        ok = eval_one(sampling_client, tokenizer, ex)
        correct += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  [{label}] {i}/{len(test_data)} {mark}  running acc {correct / i:.2%}",
              flush=True)
    return correct / len(test_data)


# =============================================================================
# Step 4: Evaluate base model
# =============================================================================

print("\n--- Evaluating Base Model on 200 Test Questions ---")
base_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="base-model"
)
base_accuracy = evaluate_test_set(
    base_sampling_client, tokenizer, test_data, label="base"
)
print(f"\nBase model accuracy: {base_accuracy:.2%} "
      f"({int(base_accuracy * NUM_TEST_EXAMPLES)}/{NUM_TEST_EXAMPLES})")

# =============================================================================
# Step 5: Tokenize training data (prompt-masked loss)
# =============================================================================

def format_prompt(example: dict) -> tuple[str, str]:
    prompt = PROMPT_TEMPLATE.format(
        context=example["context"], question=example["question"]
    )
    return prompt, example["answer"]


def process_example(example: dict, tokenizer) -> types.Datum:
    prompt, completion = format_prompt(example)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    completion_str = f" {completion}\n\n"
    completion_tokens = tokenizer.encode(completion_str, add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "weights": np.array(weights, dtype=np.float32),
        },
    )


print("\n--- Tokenizing training data ---")
processed_train = [process_example(ex, tokenizer) for ex in train_data]
random.shuffle(processed_train)
print(f"Tokenized {len(processed_train)} training examples.")

# =============================================================================
# Step 6: Fine-tune
# =============================================================================

NUM_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 5e-4


def to_arr(x):
    return x.to_numpy() if hasattr(x, "to_numpy") else np.array(x.tolist())


print(f"\n--- Training: {NUM_EPOCHS} epoch(s), batch {BATCH_SIZE}, lr {LEARNING_RATE} ---")
step = 0
for epoch in range(NUM_EPOCHS):
    random.shuffle(processed_train)
    for batch_idx in range(0, len(processed_train), BATCH_SIZE):
        batch = processed_train[batch_idx : batch_idx + BATCH_SIZE]
        if len(batch) == 0:
            break

        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_future.result()

        logprobs = np.concatenate(
            [to_arr(o["logprobs"]) for o in fwdbwd_result.loss_fn_outputs]
        )
        weights = np.concatenate(
            [to_arr(d.loss_fn_inputs["weights"]) for d in batch]
        )
        loss = float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))

        step += 1
        if step % 25 == 0 or batch_idx + BATCH_SIZE >= len(processed_train):
            print(f"  epoch {epoch + 1}/{NUM_EPOCHS}  step {step}  loss {loss:.4f}")

# =============================================================================
# Step 7: Evaluate fine-tuned model on the held-out 200
# =============================================================================

print("\n--- Evaluating Fine-Tuned Model on 200 Test Questions ---")
ft_sampling_client = training_client.save_weights_and_get_sampling_client(
    name="text2sql-finetuned"
)
ft_accuracy = evaluate_test_set(
    ft_sampling_client, tokenizer, test_data, label="finetuned"
)

print("\n=============================================")
print(f"  Base model accuracy:       {base_accuracy:.2%} "
      f"({int(base_accuracy * NUM_TEST_EXAMPLES)}/{NUM_TEST_EXAMPLES})")
print(f"  Fine-tuned model accuracy: {ft_accuracy:.2%} "
      f"({int(ft_accuracy * NUM_TEST_EXAMPLES)}/{NUM_TEST_EXAMPLES})")
print(f"  Improvement:               +{(ft_accuracy - base_accuracy) * 100:.1f} pp")
print("=============================================")

# =============================================================================
# Step 8: Novel-schema questions (out-of-distribution, manual inspection)
# =============================================================================

NOVEL_QUESTIONS = [
    # Easy
    {
        "difficulty": "easy",
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What are the names of employees in the engineering department?",
    },
    {
        "difficulty": "easy",
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "How many products cost more than 50 dollars?",
    },
    # Medium
    {
        "difficulty": "medium",
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the highest score in the science class?",
    },
    {
        "difficulty": "medium",
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "List the top 3 customers by total order amount.",
    },
    # Hard
    {
        "difficulty": "hard",
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "How many students are enrolled in each department?",
    },
]

print("\n--- Step 7: Novel-Schema Questions (manual inspection) ---")
for q in NOVEL_QUESTIONS:
    print(f"\n[{q['difficulty']}] {q['question']}")
    print(f"  schema: {q['context']}")
    base_sql = sample_from_model(
        base_sampling_client, tokenizer, q["context"], q["question"]
    )
    ft_sql = sample_from_model(
        ft_sampling_client, tokenizer, q["context"], q["question"]
    )
    print(f"  base       : {base_sql}")
    print(f"  fine-tuned : {ft_sql}")

print("\nDone.")
