# Readme
## Exercise 1
- Does the model hallucinate specific values without RAG?

Yes, the model does.
- Does RAG ground the answers in the actual manual?

Yes. RAG will find the reference data for the answer. And if it doesn't find, it will say it can not answer.
- Are there questions where the model's general knowledge is actually correct?

Yes. Like How do I fix a slipping transmission band?. It is generally correct. For general questions, the LLM without RAG has the general knowledge to answer it.

## Exercise 2

**Does GPT 4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?**
- **General Knowledge (Model T):** Yes. GPT-4o Mini is much more capable than the small Qwen model (without RAG) at recalling specific facts about the Model T. The small model often hallucinates numbers or procedures, while GPT-4o Mini provides coherent and generally accurate standard maintenance procedures.
- **Specific/Future Data (Congressional Record):** No. For the queries about 2026, GPT-4o Mini either hallucinates (making up plausible-sounding political speeches) or refuses to answer. It cannot avoid hallucination/error here because it lacks the data. Qwen 1.5B **with RAG** avoids this completely by retrieving the actual text.

**Which questions does GPT 4o Mini answer correctly?**
- **Model T Questions:** GPT-4o Mini answers these correctly. The manuals and maintenance info for a 1920s car are widely available in its training data.
- **Congressional Record Questions:** GPT-4o Mini answers these **incorrectly**. It does not know about "Mr. Flood's" comments or the "Main Street Parity Act" of 2026.

**Compare the cut-off date of GPT 4o Mini pre-training and the age of the Model T Ford and Congressional Record corpora:**
- **GPT-4o Mini Cut-off:** October 2023.
- **Model T Ford Corpus:** ~1920s (Public Domain).
  - *Result:* Deeply embedded in the model's training weights. No RAG needed for general questions.
- **Congressional Record Corpus:** January 2026 (Synthetically generated "Future" Data).
  - *Result:* Outside the model's knowledge horizon. RAG is **mandatory** for this use case.

  ## Exercise 3

**Where does the frontier model’s general knowledge succeed?**
- It gives a plausible **step-by-step structure** for “how-to” questions (even if the details are not Model-T specific).
- It applies **general mechanical troubleshooting** patterns (inspect/replace/maintain), which can be useful at a high level.

**When did the frontier model appear to be using live web search to help answer your questions?**
- It **never** shows signs of live web search in these outputs.
- It makes **confident, date-specific Congressional claims** (Jan 2026) with **no citations**, which looks like hallucination rather than web lookup.

**Where does your RAG system provide more accurate, specific answers?**
- For **Model T questions**, RAG supplies **manual-like, part-level procedure details** (e.g., adjusting rod, needle valve, gaskets, figure/paragraph references).
- For **Congressional Record questions**, RAG correctly **says the context doesn’t contain the answer** instead of inventing details.

**What does this tell you about when RAG adds value vs. when a powerful model suffices?**
- RAG adds value when the task needs **grounded, verifiable specifics** (exact procedures, exact numbers/specs, exact who/what/when facts).
- A powerful model suffices when the task is **general explanation or generic advice** and doesn’t depend on precise external facts.

## Exercise 4
**k = 1 (TOP_K = 1): What you observe**
- **Answer quality:** Often **procedural and specific** when the single retrieved chunk is on-topic (e.g., carburetor adjust, brakes).
- **Failure mode:** If the top chunk is missing the key fact, the model **can’t recover** (e.g., spark plug gap → “context does not provide…”).
- **Example of “not enough context”:**
  - Spark plug gap: **no value found** (model abstains).
  - Engine cylinder specs: **no specs found** (model abstains).
- **Latency:** mean **3.74s** (min **1.49s**, max **8.38s**) — high variance.

**k = 3 (TOP_K = 3): What you observe**
- **Answer quality:** Usually **more complete** than k=1 for multi-step procedures (carburetor adjust, cleaning).
- **Failure mode:** Can introduce **generic/hallucinated repair advice** if extra chunks are only loosely related.
- **Example of “context drift”:**
  - Slipping transmission band: shifts toward **generic “replace the band”** instead of a grounded adjustment procedure.
- **Latency:** mean **2.81s** (min **1.48s**, max **4.79s**) — more stable than k=1.

**k = 5 (TOP_K = 5): What you observe**
- **Answer quality:** Often the **best tradeoff** here: enough context to locate the relevant procedure/spec, but not so much that the model mixes topics.
- **Example of “sweet spot”:**
  - Spark plug gap: now produces a **specific range** (0.006"–0.010") instead of abstaining.
  - Slipping band: returns to a **more context-grounded adjustment** style answer.
- **Failure mode:** Still can be incomplete if the corpus itself lacks the missing detail (oil question gives quantity, not type/viscosity).
- **Latency:** mean **2.42s** (min **0.75s**, max **4.65s**) — lowest average among your runs.

**k = 10 (TOP_K = 10): What you observe**
- **Answer quality:** Can improve **step-by-step completeness** for big procedures (disassembly becomes more detailed).
- **Failure mode:** **Too much context → conflicting or off-topic chunks get merged.**
- **Examples of “too much context hurts”:**
  - Spark plug gap: now changes to a **different number** (“about .025 inches”), conflicting with k=5.
  - Slipping transmission band: shifts toward **clutch finger screws** (may be a different subsystem/topic than “band slipping” fix).
  - Brakes: includes **clutch lever / low-speed pedal** adjustments mixed into brake adjustment steps.
- **Latency:** mean **3.26s** (min **1.64s**, max **4.39s**) — slower than k=5.

**k = 20 (TOP_K = 20): What you observe**
- **Answer quality:** Sometimes better grounding for long procedures (disassembly steps look most “manual-like”).
- **Failure mode:** **High confusion risk**: the model may summarize a broad set of chunks and either
  - (a) **hedge/abstain** (“no specific mention”), or
  - (b) **mix unrelated specs** (e.g., spark plug gap answer starts talking about coil box terminals / multiple gap sizes).
- **Examples of “overload”:**
  - Spark plug gap: becomes **messy/less direct**, pulling in other “gap” concepts.
  - Engine cylinder specs: returns to **no specific mention**, despite k=5 having at least some “spec 8/spec 35” references.
- **Latency:** mean **3.08s** (min **1.51s**, max **4.94s**) — not dramatically worse than k=10, but quality is less consistent.

**At what point does adding more context stop helping?**
- In your runs, **k ≈ 5** is where returns often **start diminishing** for narrow questions (spark plug gap, band slip).
- For very broad “manual chapter” questions (e.g., **disassembling the car**), increasing to **k=10 or k=20** can still help completeness.

**When does too much context hurt (irrelevant info, confusion)?**
- When the query keyword matches multiple topics in the corpus (e.g., “gap,” “band,” “adjust”):
  - Spark plug gap becomes **inconsistent** across k=5, k=10, k=20.
  - Band slipping gets **pulled toward clutch-related paragraphs** at higher k.
  - Brake adjustment gets **contaminated** by clutch/low-speed pedal steps.

**How does k interact with chunk size? (conceptual takeaway)**
- **Large chunk size + high k:** lots of redundant text + more unrelated nearby procedures ⇒ **higher risk of topic mixing**.
- **Small chunk size + low k:** you may miss the one sentence containing the critical spec ⇒ **more abstentions** (like k=1 spark plug gap).
- Practical implication:
  - If chunks are **small**, you often need a **moderate k (≈ 5–10)** to capture a complete procedure.
  - If chunks are **large**, you should keep **k smaller (≈ 3–5)** to avoid drowning the model in adjacent but irrelevant sections.

## Exercise 5
**What happens on fully off-topic questions (no real overlap with the corpus)?**
- The system is **inconsistent**: sometimes it refuses correctly, sometimes it hallucinates.
- Hallucinations happen when **loose keyword matches** exist (even in titles/metadata), which nudges the model to “force” an answer.
- Takeaway: off-topic queries expose a **retrieval + prompt weakness**—weak matches can still trigger confident guessing.

**What happens on related-but-missing questions (same domain, but the exact fact isn’t in the retrieved text)?**
- The model often **tries to fill in missing numeric facts** using general knowledge (e.g., inventing a plausible top speed).
- It may hedge (“estimate,” “not in context”), but it still violates strict grounding.
- When it behaves well, it **refuses** and/or **explains why a similar-looking number is not the answer** (good evidence checking).

**What happens on false-premise questions (the question assumes something that doesn’t apply)?**
- If retrieval finds a **neighbor topic** (e.g., “battery”), the model may produce a **grounded-but-misaligned** answer: accurate to the manual, but not addressing the false premise directly.
- The worst failures occur when it **misreads unrelated numbers** as the requested spec (e.g., treating spring-tension “24–28 lbs” as tire pressure).
- When the premise is clearly absent (e.g., “infotainment system”), the model more often **refuses correctly**.

**What does this imply about the current `PROMPT_TEMPLATE` and grounding?**
- “Use/quote the context” can **backfire** by encouraging the model to search for *any* connection and answer anyway.
- RAG helps only when retrieved chunks are **highly relevant**; otherwise it can amplify **spurious matches**.
- Overall: the pipeline needs stronger **relevance filtering** and **abstention rules** to prevent answering from weak evidence.

## Exercise 6: Query Phrasing Sensitivity (Model T brakes adjustment)

**Underlying question (same intent across all queries)**
- Adjusting the **Model T brake system** (primarily the hand brake lever + pull rods/clevises/brake setting).

**Phrasings tested (5+)**
- Query 1 (Formal): “What is the procedure for adjusting the braking system of a Model T Ford?”
- Query 2 (Casual): “How do I go about adjusting the Model T’s brakes?”
- Query 3 (Keywords-only): “Model T brakes adjustment procedure”
- Query 4 (Instruction/imperative): “Adjust the brakes on the Model T, please.”
- Query 5 (Indirect/polite request): “Could you provide instructions on how to adjust Model T brakes?”

**Per-phrasing retrieval + answer behavior (Top-5 chunk themes + impact)**
- Query 1 (Formal)
  - Retrieved chunks (top-5 themes)
    - Brake adjustment via **hand brake lever**
    - **Threaded clevises** + pull rods adjustment
    - **Brake shoes / brake setting evenly** across wheels
    - Tool usage (e.g., **drift** to align/fit pins)
    - “New Design **Transmission Brake Band**” (related but component-specific)
  - Similarity scores
    - Not recorded in the write-up (**add scores here**)
  - Answer quality
    - Mostly **on-target step-by-step**, transmission brake band mentioned but not dominant.

- Query 2 (Casual)
  - Retrieved chunks (top-5 themes)
    - Same core brake-adjustment chunks: **clevises / pull rods / drift**
    - Hand brake lever positioning
    - Even brake setting across wheels
    - A chunk about **brake drum shaft gear clearance** appears (more internal/assembly detail)
    - General brake linkage guidance
  - Similarity scores
    - Not recorded (**add scores here**)
  - Answer quality
    - Still **consistent** with Query 1; the gear-clearance chunk is mostly ignored (good filtering).

- Query 3 (Keywords-only)
  - Retrieved chunks (top-5 themes)
    - Core brake-adjustment chunks remain dominant (clevises/pull rods/hand brake)
    - Brake setting evenly
    - Drift/pin alignment
    - A broader match appears: **Rear Axle Overhaul** (keyword spillover)
    - Additional brake linkage context
  - Similarity scores
    - Not recorded (**add scores here**)
  - Answer quality
    - **Concise and correct**, Rear Axle Overhaul does not noticeably derail the answer.

- Query 4 (Instruction/imperative)
  - Retrieved chunks (top-5 themes)
    - Brake drum shaft + **gear clearance (~0.006")**
    - Checking for distorted bushings / internal mechanical checks
    - Still includes clevis/pull rod adjustment chunks
    - Hand brake lever positioning chunks
    - Mixed “assembly-level” brake details
  - Similarity scores
    - Not recorded (**add scores here**)
  - Answer quality
    - **Noticeably different**: starts with internal gear/clearance checks, then returns to linkage adjustment.
    - Interpretation: this phrasing pulled the system toward **technical sub-assembly** details.

- Query 5 (Indirect/polite request)
  - Retrieved chunks (top-5 themes)
    - Core brake-adjustment chunks (clevis/pull rod/drift) are present
    - BUT also retrieves **clutch lever screw adjustment**
    - Clutch components: cap screws, clutch springs, clutch finger screws, etc.
    - Additional clutch procedure chunks (context drift)
    - Less brake-specific content in the top ranks than other queries
  - Similarity scores
    - Not recorded (**add scores here**)
  - Answer quality
    - **Strong context drift**: answer begins with clutch adjustment and stays mostly clutch-focused.
    - This is the clearest case where retrieval noise directly caused a wrong/off-topic generation.

**Overlap between result sets (qualitative comparison)**
- High overlap:
  - Query 1, Query 2, Query 3 share the same “core brake adjustment” cluster:
    - hand brake lever + clevises + pull rods + drift + even brake setting
- Partial shift:
  - Query 4 overlaps with the core cluster but adds more **internal brake assembly** chunks (gear clearance, bushings).
- Low overlap / drift:
  - Query 5 overlaps weakly with the core cluster and introduces a competing cluster (**clutch adjustment**).

**Which phrasings retrieved the best chunks?**
- Best / most stable:
  - Query 1 (Formal) and Query 2 (Casual): consistently retrieve **direct brake adjustment** procedure chunks.
  - Query 3 (Keywords-only): still strong, but slightly more prone to broad matches (e.g., Rear Axle Overhaul).
- Most risky:
  - Query 5 (Indirect/polite): highest risk of **semantic spillover** due to generic term “adjust,” pulling in clutch.
  - Query 4 (Imperative): tends to pull more **fine-grained assembly** checks, changing the answer emphasis.

**Do keyword-style queries work better or worse than natural questions?**
- In this run, keyword-style (Query 3) works **about as well** as natural questions for the main procedure.
- However, keyword-style queries appear **more likely to broaden retrieval** into nearby sections (e.g., overhaul chapters).

**What this suggests about query rewriting strategies**
- Add **entity + subsystem anchoring** in rewrites:
  - Include “brakes / brake pull rods / clevis / hand brake lever” explicitly.
- Reduce drift with **negative constraints**:
  - If your retriever supports it, downweight or exclude “clutch” when the user intent is brakes.
- Use a two-stage approach:
  - (1) rewrite user query → canonical query (“Model T brake adjustment clevis pull rods hand brake lever”)
  - (2) retrieve + rerank with a cross-encoder/reranker to penalize clutch-heavy chunks.
- Use intent-aware expansion:
  - Expand “adjust brakes” to include expected terms (clevis, pull rod, brake shoe setting) so the retriever locks onto the right section.

**What you still need to record to fully satisfy the exercise requirements**
- For each phrasing:
  - The **top 5 retrieved chunks** (IDs/titles/first line)
  - Their **similarity scores**
  - A simple overlap metric (e.g., “3/5 chunks overlap with Query 1”)

## Exercise 7:
### Q&A

**Question:** How did increasing the chunk overlap value affect the number of vectors stored in the index?

**Answer:** For lower overlap values (0, 64, and 128), the document was split into 3 chunks (vectors). However, when the overlap was increased to 256, the number of chunks increased to 5 vectors. This demonstrates that significantly higher overlap results in more segments being generated for the same source text.

### Data Analysis Key Findings

*   **Experiment Configuration:** The analysis successfully established a testing framework for chunk overlap using values of **0, 64, 128, and 256**, while keeping the chunk size constant at 512.
*   **Index Size Variation:** The experiment revealed a non-linear relationship between overlap and index size:
    *   Overlap values of **0, 64, and 128** produced a consistent index size of **3 vectors**.
    *   The highest overlap value of **256** resulted in a larger index size of **5 vectors**.
*   **Retrieval Consistency:** across all overlap settings, the RAG system functioned correctly. It consistently provided answers for procedural questions (e.g., disassembling the car) and correctly identified when information was missing from the context (e.g., regarding the spark plug gap).

### Insights or Next Steps

*   **Trade-off Analysis:** Higher overlap values (like 256) increase the number of vectors, which implies higher storage requirements and slightly more computational cost during search, but they ensure that boundary-spanning information is preserved.
*   **Quality Evaluation:** While the system produced answers for all settings, the next logical step is to qualitatively compare the answers for the specific "boundary" questions to determine if the higher overlap actually resulted in more coherent or complete responses compared to zero overlap.

## Exercise 8
**Exercise 8: Chunk Size Experiment (Model T corpus)**

**Setup (three chunking configurations)**
- Chunk size = **128 chars** (overlap 32) → **10 chunks** indexed
- Chunk size = **512 chars** (overlap 128) → **3 chunks** indexed
- Chunk size = **2048 chars** (overlap 512) → **3 chunks** indexed

**Queries tested (same across configs)**
- “How to adjust the brakes?”
- “What is the function of the commutator?”
- “Describe the procedure for disassembling the car.”

---

**What happens with very small chunks (128 chars)?**
- **Retrieval precision (relevant vs. irrelevant):**
  - Chunks are so small that key procedures get **split across many fragments**.
  - Even if retrieval finds the right area, it often returns **incomplete fragments** that don’t contain the critical steps.
- **Answer completeness:**
  - The model **falls back to generic advice** and hedging (“refer to the manual… general guideline…”).
  - This suggests the retrieved evidence is **too thin** to support grounded step-by-step answers.
- **Overall quality pattern:**
  - High chance of **missing essential context** → more “manual says…” style answers rather than quoting real procedure.

---

**What happens with medium chunks (512 chars)?**
- **Retrieval precision (relevant vs. irrelevant):**
  - Chunks contain more contiguous context, so retrieval is **less fragmentary** than 128.
  - However, because only **3 chunks** exist in the index, retrieval becomes **coarse** (each chunk covers a lot).
- **Answer completeness:**
  - Answers are still largely **generic** and modern-car flavored (fluids, master cylinder, warning lights), which suggests:
    - The retrieved chunk may not be Model-T-specific, or
    - The model is still not being strongly anchored to manual-style text.
- **Overall quality pattern:**
  - More context per chunk helps continuity, but the **index is too low-resolution**, so retrieval can’t reliably target the right passage.

---

**What happens with very large chunks (2048 chars)?**
- **Retrieval precision (relevant vs. irrelevant):**
  - Very large chunks increase the chance that retrieved text includes the right information somewhere, but also include a lot of **irrelevant surrounding material**.
  - With only **3 chunks**, retrieval is even more **blunt**: you essentially retrieve a large section of the corpus.
- **Answer completeness:**
  - Answers remain **generic** and sometimes even more “textbook” (master cylinder/calipers/rotors), indicating the model is not extracting Model T–specific procedures from the retrieved text.
- **Overall quality pattern:**
  - Large chunks tend to **increase noise**, encouraging summarization and generalization instead of precise instruction.

---

**How does chunk size affect retrieval precision?**
- **128 chars:** high lexical match potential, but **low semantic completeness** (procedures get broken).
- **512 chars:** better completeness, but in this run the index collapsed to **only 3 vectors**, reducing precision.
- **2048 chars:** lowest precision because each chunk contains **too many topics**, making retrieval less discriminative.

---

**How does chunk size affect answer completeness?**
- **128 chars:** lowest completeness (model lacks enough evidence → generic fallback).
- **512 chars:** medium completeness, but still generic because retrieval is coarse.
- **2048 chars:** can be “complete-sounding,” but often **not grounded** to the correct subsystem details (more generic automotive content).

---

**Is there a sweet spot for this corpus (based on these results)?**
- In this run, none of the three settings produced clearly grounded, Model-T-specific answers.
- The most likely sweet spot for a manual-style corpus is typically **medium chunks (≈ 300–800 chars)** *with enough total chunks* to preserve retrieval resolution.
- Here, **512 and 2048 produced only 3 chunks**, which is a warning sign: chunking likely needs to be applied to a **larger corpus slice** (or different splitting rules) so the index contains enough vectors to discriminate sections.

---

**Does the optimal size depend on the type of question?**
- **Fact lookup (e.g., a specific spec/number):**
  - Prefer **smaller-to-medium chunks** so the exact sentence is retrievable without extra noise.
- **Multi-step procedures (e.g., brake adjustment, disassembly steps):**
  - Prefer **medium chunks** so one chunk contains a coherent sequence of steps.
- **Broad conceptual questions (e.g., “function of commutator”):**
  - Larger chunks can work because the answer is definitional, but it may also encourage generic textbook responses if not grounded to the corpus.

---

**Key conclusion**
- Chunk size trades off **precision vs. completeness**:
  - Too small → fragmented evidence → generic answers.
  - Too large → too much irrelevant context → drift/generalization.
- For a repair manual corpus, a practical strategy is **medium chunking** plus ensuring the index has **enough chunks** (not just 3) to retrieve the correct section reliably.

## Exercise 9

### Q&A
**Q: How effective is the 0.25 similarity score threshold?**
**A:** The 0.25 threshold is effective at filtering out clearly irrelevant queries (e.g., "Capital of France"), achieving high precision. However, it is somewhat too strict for specific technical queries (e.g., "Spark plug gap"), leading to false negatives where relevant questions are refused because they lack exact keyword matches in the index.

**Q: What is the average "confidence gap" between the top two results?**
**A:** The average gap is approximately 0.069, indicating that for many queries, the system does not strongly distinguish the top result from the second-best result.

### Data Analysis Key Findings
*   **Retrieval Statistics:**
    *   Average Rank 1 Score: `0.205` (indicating generally low confidence).
    *   Average Rank 2 Score: `0.136`.
    *   Average Confidence Gap: `0.069`.
*   **Threshold Performance (0.25 cut-off):**
    *   **Answered:** 3 queries passed the threshold (e.g., "Disassembling the car" with score ~0.482).
    *   **Refused:** 7 queries fell below the threshold.
    *   **False Negatives:** Valid technical queries like "What is the correct spark plug gap?" were refused, highlighting a recall issue.
*   **Confidence Extremes:**
    *   Only 1 query achieved a "High Confidence" score (> 0.4).
    *   Both off-topic queries (e.g., baking a cake) and specific technical queries without exact phrasing fell into the "Low Confidence" bucket (< 0.25).

## Exercise 10


### Q&A
**Q: How did the "Strict Grounding" template behave across different query types?**
**A:** The "Strict Grounding" template successfully triggered refusals (e.g., "I do not know") for technical queries where information was missing from the context. However, for queries involving strong general knowledge (like "Capital of France"), the model failed to refuse and overrode the negative constraints to answer using its internal training data.

**Q: What specific risk was identified with the "Encouraging Citation" template?**
**A:** This template often led to "hallucinated citations," where the model would provide a correct answer based on general knowledge but falsely attribute it to a dummy document that did not actually contain that information.

### Data Analysis Key Findings
- **Hallucination Control:** The "Strict Grounding" template effectively curbed hallucination for domain-specific technical queries but struggled to suppress the model's internal knowledge on highly familiar topics.
- **Citation Reliability:** Asking the model to cite sources ("Encouraging Citation") introduced a reliability risk where the model fabricated connections between its internal knowledge and the provided context documents.
- **Trade-offs in Template Styles:**
    - The **Permissive** template maximized helpfulness and detail but sacrificed verifiability.
    - The **Minimal** template achieved conciseness but remained prone to hallucinating details absent from the sparse context.


## Exercise 11
**Can the model successfully combine information from multiple chunks?**
- **Sometimes, but only when the retrieved chunks are clearly aligned.**
- When the top-k set contains chunks that all describe the same procedure/topic (e.g., several brake-adjustment passages), the model can **merge them into a single step-by-step answer** without obvious seams.
- When the retrieved chunks come from different but related subsystems (e.g., brakes + transmission band + clutch), the model often **fails to separate them**, and synthesis turns into **topic mixing** rather than true combination.

**Does it miss information that wasn’t retrieved?**
- **Yes—consistently. Retrieval is the bottleneck.**
- If the key sentence/spec/procedure step is not in the retrieved set, the model typically:
  - **Hedges/abstains** (“context does not provide…”) when it follows grounding rules, or
  - **Backfills with generic knowledge** (hallucination) if the prompt/retrieval weakly suggests an answer.
- In practice, this means the system can look “smart” but still **omit critical steps or exact numbers** simply because they were not retrieved.

**Does contradictory information in different chunks cause problems?**
- **Yes—contradictions often lead to inconsistency or cherry-picking.**
- Common failure patterns when chunks disagree (or “look like they disagree”):
  - The model **chooses one value** without explaining why (e.g., spark plug gap shifting across retrieved contexts).
  - The model **averages/hedges** into a range or vague wording (“typically around… depends on…”), even if the corpus contains a specific instruction.
  - The model **mixes procedures** (e.g., brake adjustment answer contaminated by clutch adjustment steps) when both contain the same action words like “adjust,” “turn screw,” “clearance,” etc.
- Net effect: contradictory or competing chunks can reduce reliability unless you add reranking, filtering, or a “conflict check” step.

**Overall takeaway**
- Cross-chunk synthesis works best when retrieval returns **multiple chunks that are mutually consistent and tightly on-topic**.
- The model misses important information primarily due to **retrieval gaps**.
- Contradictions and near-contradictions in retrieved chunks can cause **value switching, hedging, or subsystem drift**, especially for numeric specs and “adjustment” procedures.
