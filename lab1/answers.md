# Lab 1: Running an LLM - Answers & Summary

## Task-to-File Mapping

| Task | Description | Files/Evidence |
|------|-------------|----------------|
| **1-2** | Environment setup & HF auth | `.venv/`, timelog.txt shows HF auth working |
| **3** | Verify setup with llama_mmlu_eval.py | `llama_mmlu_eval.py` - original evaluation script |
| **4** | Timing comparisons | `timelog.txt` (lines 1-68) |
| **5** | Modified evaluation script | `llama_mmlu_eval_modified.py` |
| **6** | Graphs & mistake analysis | `compare_models.py`, `analyze_mistakes.py`, `model_comparison.png/pdf` |
| **7** | Google Colab experiments | `colab_version/llama_mmlu_eval.ipynb` |
| **8** | Chat agent | `simple_chat_agent.py` |

---

## Q4: Timing Comparisons (Laptop - Apple Silicon)

| Configuration | Accuracy | Duration | Notes |
|--------------|----------|----------|-------|
| GPU (MPS) + No Quant | 48.02% | 0.5 min | Best speed on Mac |
| CPU + No Quant | 47.62% | 1.3 min | 2.6x slower than GPU |
| CPU + 4-bit Quant | 44.44% | 2.8 min | Slower due to CPU quantization overhead |

**Key Finding:** On Apple Silicon, GPU (MPS) without quantization is fastest. 4-bit/8-bit quantization with bitsandbytes is NOT supported on MPS (Metal), only on CUDA.

---

## Q5: Modified Evaluation Script Features

The `llama_mmlu_eval_modified.py` adds:
1. **10 MMLU subjects** tested across multiple models
2. **TimingStats class** tracking:
   - Real time (wall clock)
   - CPU time
   - GPU time (CUDA kernel timing where available)
3. **VERBOSE_OUTPUT flag** - prints each question, model answer, correct answer, and ✓/✗ status

### Models Tested (Small - Laptop):
- `meta-llama/Llama-3.2-1B-Instruct` (baseline)
- `allenai/OLMo-2-0425-1B`
- `Qwen/Qwen2.5-0.5B`

---

## Q6: Mistake Pattern Analysis

### Accuracy Comparison (10 MMLU Subjects)

| Model | Overall Accuracy | Best Subject | Worst Subject |
|-------|-----------------|--------------|---------------|
| Qwen2.5-0.5B | **42.80%** | clinical_knowledge (52.08%) | college_mathematics (25.00%) |
| OLMo-2-0425-1B | 34.77% | anatomy (44.44%) | college_computer_science (28.00%) |

### Major Patterns Discovered

1. **OLMo has extreme 'C' bias** - When wrong, OLMo overwhelmingly chooses answer 'C' (~68% of wrong answers), suggesting positional/middle-option preference bias.

2. **Significant overlap in mistakes** - 32.8% of questions both models got wrong with the **exact same wrong answer**. This indicates:
   - Questions have "attractive distractors" that fool both models
   - Systematic reasoning failures, NOT random guessing
   - Both struggle with detailed metabolic pathway questions
   - "False statement" questions (asking what is NOT true) are particularly hard

3. **Complementary knowledge** - Qwen outperforms OLMo on 25 questions where only one was correct, suggesting model-specific knowledge strengths.

4. **Topic weaknesses** - Both models struggle most with:
   - Biochemistry (metabolic pathways, enzyme reactions)
   - Physiology (muscle fiber types, lactate metabolism)
   - Abstract mathematics

---

## Q7: Google Colab Results (Medium Models)

| Model | Size | Overall Accuracy | Best Subject | Throughput |
|-------|------|-----------------|--------------|------------|
| **Qwen2.5-7B** | 7B | **69.39%** | college_biology (83.33%) | 1.41 q/s |
| OLMo-3-1025-7B | 7B | 57.49% | college_biology (70.83%) | 2.00 q/s |
| OLMo-2-1124-7B | 7B | 53.83% | college_biology (71.53%) | 2.00 q/s |
| Qwen2.5-0.5B | 0.5B | 43.10% | clinical_knowledge (52.45%) | 20.71 q/s |
| OLMo-2-0425-1B | 1B | 34.92% | anatomy (44.44%) | 30.14 q/s |

### Colab Timing Comparisons (Tesla T4 GPU)
| Configuration | Duration | Notes |
|--------------|----------|-------|
| Llama-3.2-1B + GPU + Full | 0.2 min | Fastest |
| Llama-3.2-1B + GPU + 4-bit | 0.3 min | Slight overhead |
| Llama-3.2-1B + GPU + 8-bit | 0.5 min | More overhead |
| Llama-3.2-1B + CPU + Full | 13.0 min | 65x slower! |

**Key Finding:** GPU acceleration provides ~65x speedup over CPU for inference.

---

## Q8: Chat Agent Features

The `simple_chat_agent.py` implements:

### Context Management
- **Fixed Window Strategy**: Keeps system prompt + last N messages (configurable, default 10)
- Prevents context length overflow on long conversations

### History Toggle
```python
add_history = True  # Set to False to disable conversation memory
```
- When `True`: Full conversation context maintained (multi-turn aware)
- When `False`: Each turn is independent (only system + current message)

### Multi-turn Comparison
- **With history**: Model remembers previous exchanges, can reference earlier topics
- **Without history**: Model treats each message as new conversation, loses context

---

## Summary of Key Findings

1. **Model size matters**: 7B models (Qwen2.5-7B: 69.39%) vastly outperform 1B models (OLMo-2-0425-1B: 34.77%)

2. **Qwen models consistently outperform OLMo** at similar sizes

3. **Quantization tradeoffs**:
   - 4-bit reduces accuracy by ~3-4%
   - 8-bit has minimal accuracy impact
   - Only works on CUDA, not Apple MPS

4. **Mistake patterns are systematic**, not random - models share similar failure modes on "attractive distractor" questions

5. **Chat context management is essential** - fixed window approach balances memory usage with conversation coherence
