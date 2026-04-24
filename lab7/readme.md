## 2.1 How REST APIs Work
curl -s "https://api.open-meteo.com/v1/forecast?latitude=40.7128&longitude=-74.0060&current_weather=true"
{"latitude":40.710335,"longitude":-73.99308,"generationtime_ms":0.16009807586669922,"utc_offset_seconds":0,"timezone":"GMT","timezone_abbreviation":"GMT","elevation":32.0,"current_weather_units":{"time":"iso8601","interval":"seconds","temperature":"°C","windspeed":"km/h","winddirection":"°","is_day":"","weathercode":"wmo code"},"current_weather":{"time":"2026-03-10T18:45","interval":900,"temperature":25.7,"windspeed":16.1,"winddirection":190,"is_day":1,"weathercode":0}}%   

Parallel tool calls are best when:

the calls are independent

you want lower latency

you are gathering evidence from multiple sources before reasoning

The message back to the LLM should usually include:

tool name

whether failure is retryable

error type/category

short human-readable message

any partial result if available

suggested next actions

1. Return only the fields the model needs
Instead of the full API payload, send a compact schema.
Example: for email search, return sender, subject, date, snippet — not every header.

2. Top-k filtering
Keep only the most relevant results.
Example: top 5 search hits, not 100 hits.

3. Extractive summarization
Pull out the most relevant passages, rows, or facts rather than the full document.

4. Map-then-reduce
For very large outputs:

summarize chunks separately

then combine into one final summary

5. Hierarchical return format
Send:

short summary

key facts

optional references/ids for drill-down later


## Exercise a 
Tool: get_paper
  Description: Get details about a paper by its id.
  Required: paper_id (string)
  Optional: fields (string)

Tool: get_paper_batch
  Description: Get details about a list of papers by their ids.
  Required: ids (array)
  Optional: fields (string)

Tool: get_citations
  Description: Get details about the papers that cite this paper (i.e. papers in whose bibliography this paper appears)
  Required: paper_id (string)
  Optional: fields (string), limit (integer), publication_date_range (string)

Tool: search_authors_by_name
  Description: Search for authors by name.
  Required: name (string)
  Optional: fields (string), limit (integer)

Tool: get_author_papers
  Description: Get papers written by this author.
  Required: author_id (string)
  Optional: paper_fields (string), limit (integer), publication_date_range (string)

Tool: search_papers_by_relevance
  Description: Search for papers by keyword relevance.
  Required: keyword (string)
  Optional: fields (string), limit (integer), publication_date_range (string), venues (string)

Tool: search_paper_by_title
  Description: Search for papers by title.
  Required: title (string)
  Optional: fields (string), publication_date_range (string), venues (string)

Tool: snippet_search
  Description: Search for text snippets that most closely match the query.
  Required: query (string)
  Optional: limit (integer), venues (string), paper_ids (string), inserted_before (string)

## Discussion questions

### MCP vs A2A: How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?

An MCP tool is a deterministic, stateless function: you send arguments that match a fixed JSON schema, the server runs a known routine, and you get a structured result. The caller holds all the intelligence — the tool just executes.

An A2A peer is itself an agent. When I POST a question to another agent, I don't specify *how* it should be answered. The receiving agent reasons about the task, may decide to call its own LLM, invoke its own MCP tools, delegate further to yet another agent, ask for clarification, or refuse. In this lab my history agent received raw questions and chose the framing ("committed history nonsense" vs "real answer") on its own.

Things an agent can do that a tool cannot:
- **Reason about the task** — interpret ambiguous requests, recognize its own limits, decide whether to answer at all.
- **Hold state across turns** — maintain a conversation, remember prior context.
- **Compose capabilities** — call tools, call other agents, do multi-step planning.
- **Refuse or negotiate** — say "I'd rather answer this subset" or "I need more info."
- **Produce novel output** — generate, not just retrieve.

A tool is a verb; an agent is a participant.

### Discovery: We used a central registry. What are the alternatives? What are the tradeoffs of centralized vs decentralized discovery?

**Alternatives to a central registry:**
- **DNS-like well-known URLs** — publish agent cards at fixed domains (`https://alice.example.com/.well-known/agent.json`). Clients discover via search engines or direct sharing.
- **Gossip / mesh discovery** — each agent tells its neighbors about peers it knows; the graph propagates. Used in P2P systems like BitTorrent DHT.
- **Blockchain / distributed ledger** — agent registrations recorded on-chain, anyone can query without a trusted operator.
- **Out-of-band sharing** — URLs shared in Slack, email, or pasted directly. "I know Bob's URL because Bob told me."
- **Capability-based discovery** — marketplaces or search indexes where agents are indexed by what they do, not who runs them.

**Tradeoffs:**

| | Centralized (our registry) | Decentralized |
|---|---|---|
| Setup | Trivial — one server, everyone points at it | Complex protocols, bootstrapping problem |
| Latency | Fast single lookup | Multi-hop gossip can be slow |
| Censorship | Operator can delist agents | Very hard to block |
| Trust | Single point to trust, single point to compromise | No single actor to trust, but harder to verify |
| Freshness | Always current | Gossip lag, stale entries |
| Fairness | Operator sets the rules | Emergent — could be captured by whoever has more compute |

In a classroom we took the easy win. In a production multi-agent ecosystem, a federated model (multiple registries that trust each other) is the realistic middle ground.

### System prompts as strategy: How much did the system prompt matter for scoring? Could you craft a prompt that is good at all categories while still being funny on off-topic questions?

The prompt mattered enormously. The default template prompt ("don't answer correctly, make up a funny answer") wasn't strong enough — GPT-4o-mini kept leaking the real answer anyway. Adding an explicit A/B classification rule, four few-shot examples, and hard "never break character" rules made the off-topic answers reliable and funnier.

**Could you be good at every category AND funny off-topic?** Not really, with the scoring rules as given:
- You score +1 per correct answer — so breadth helps.
- You get a +1 funniness bonus only for *wrong* answers in a round.

A generalist prompt that answers everything correctly gives up all funniness bonuses. A specialist prompt trades 20 correct answers (5 categories × 4 questions) for potentially 6 funniness bonuses (one per round). The specialist is playing a concentrated bet: dominate your category + win funny bonuses on the 20 wrong ones.

You could try a hybrid — answer correctly if confident above some threshold, be funny otherwise — but GPT-4o-mini is calibrated poorly enough that it'll be "confident" on bad answers. The specialist strategy is more predictable.

### Smart routing: TF-IDF matched questions to agents based on text overlap. What would happen with semantic embeddings instead? What if agents could self-report confidence?

**Semantic embeddings** (e.g., embedding the question and each agent card, then cosine similarity) would catch cases where TF-IDF fails because the vocabulary doesn't overlap. Example: a question about "Napoleon's generals" would match my description containing "monarchs, empires, battles" via embedding similarity but might miss on TF-IDF if none of those exact words appear. Downside: embeddings hide *why* a match happened (less debuggable), and a well-crafted generic agent card ("answers all questions about anything historical, scientific, cultural...") could game the embedding space.

**Self-reported confidence** would let the router ask each agent "how well can you answer this?" before dispatching. Advantages: the agent has more context than its static card — it can look at the question and know whether it's on-topic. Disadvantages: agents have every incentive to overstate confidence (more traffic = more points in this game). You'd need either a reputation system that penalizes overconfident wrong answers, or a verification step that grades the claim.

Best real-world design: use embeddings as a coarse filter (top-K candidates), then ask those K agents to bid with calibrated confidence, then weight by historical accuracy from a reputation ledger.

### Trust and reliability: In a real multi-agent system, how would you handle an agent that returns bad data? What if an agent is slow or goes offline mid-task?

**Bad data:**
- **Cross-verification** — ask multiple agents the same question, treat disagreement as a red flag.
- **Reputation scores** — track per-agent accuracy over time; weight or deprioritize consistently-wrong agents.
- **Provenance** — require agents to cite sources, and verify citations out-of-band.
- **Signed responses** — sign answers cryptographically so you know *which* version of the agent produced bad output (agents get updated; bad output might be from a regression).
- **Human-in-the-loop for high-stakes calls** — never auto-act on a single agent's answer without review.

**Slow / offline agents:**
- **Timeouts** — every A2A call needs a deadline; no call hangs forever.
- **Circuit breakers** — after N consecutive failures, stop calling that agent for a cooldown period.
- **Async patterns** — for long-running tasks, use a task ID + polling or webhook callback instead of blocking RPC.
- **Redundancy** — route to a second agent if the first exceeds SLA.
- **Health checks** — the registry should actively ping agents and mark dead ones as unavailable.
- **Idempotency** — design tasks so retrying is safe (duplicate submissions produce the same result).

The trivia script in this lab used timeouts and parallel broadcast, which covers the basics. Production needs reputation + circuit breakers on top.

### Scaling: What would break if there were 1,000 agents instead of 20? What architectural changes would you need?

Things that break:

- **Broadcast-to-all** — with 1,000 agents per question, even at low latency you're racing the slowest agent; the tail latency kills the tournament. Need top-K routing via embeddings instead of broadcast.
- **Central registry as bottleneck** — 1,000 registrations, 1,000 health checks, all queries hitting one process. Need sharding, replication, or a federated registry.
- **LLM cost to judge** — judging 1,000 answers per question × 24 questions = 24,000 GPT calls for scoring alone. Need cheaper first-pass scoring (exact-match, rubric-based) with LLM only as tiebreaker.
- **Message ordering and state** — with 20 agents you can keep the full conversation in memory. With 1,000 you need a proper message bus (Kafka, Redis streams) and event-sourced state.
- **TF-IDF scoring over 1,000 cards per question** — still fast, but the signal-to-noise gets worse. You need hierarchical routing: coarse category classifier → specialist sub-router.
- **Spam / Sybil attacks** — nothing stops me from registering 500 fake agents biased toward my answers. Need registration limits per identity, proof-of-work, or a payment gate.
- **Observability** — debugging a bad answer in a 1,000-agent system requires distributed tracing (OpenTelemetry-style), not print statements.

Architectural changes:
- Replace broadcast with **router → top-K selection** (embeddings + reputation).
- Replace the single registry with a **sharded / federated directory** (agents register with their regional shard; shards gossip).
- Add a **message bus** (Kafka/NATS) for async task dispatch instead of synchronous HTTP.
- Add **reputation & identity layer** (signed agent cards, per-agent accuracy history).
- Add **distributed tracing** for debugging.
- Add **rate limits & quotas** per agent to contain misbehavior.

In short: what works for 20 is synchronous, centralized, and hand-debuggable. What works for 1,000 is asynchronous, federated, and observable.
