# Agentic AI Systems 🤖

> Complete agentic AI pipeline — ReAct agent, Deep Research 
> workflow, RAG with FAISS, and hallucination evaluation framework.
> Direct replica of CareerGPT production system.

## Results   

| Component | Metric | Result |
|-----------|--------|--------|
| ReAct Agent | Tool selection accuracy | ✅ Automatic |       
| Deep Research | Steps per query | 4 (decompose→research→validate→synthesize) |      
| RAG Retrieval | Avg similarity score | 0.680 top match |
| Hallucination | Faithfulness score | **1.00/1.0** |          
| Evaluation | Correct answers | **4/5 tests** |
| Response time | Avg per query | 3.21s |

---

## What We Built

### Part 1 — ReAct Agent (Reason + Act)
```
Chatbot:  User asks → LLM answers → Done
Agent:    User asks → LLM thinks → Selects tool
          → Executes tool → Observes result
          → Thinks again → Final answer
```
Implemented from scratch with 5 tools:
calculator, web_search, get_datetime,
summarize_text, fact_check

### Part 2 — Tool Orchestration
5 tools with automatic selection:
- **Calculator** — safe math evaluation
- **Web Search** — Wikipedia API with retry logic
- **DateTime** — current time retrieval
- **Summarizer** — LLM-powered text compression
- **Fact Checker** — claim verification with search

### Part 3 — Deep Research Agent
Direct replica of CareerGPT Deep Research workflow:
```
Step 1: Query Decomposition
        Complex question → N focused sub-questions

Step 2: Parallel Sub-Research
        Each sub-question researched independently
        Wikipedia API for real information

Step 3: Consistency Validation
        Cross-reference findings
        Flag contradictions and uncertainties

Step 4: Synthesis
        All findings → comprehensive final answer
        Honest about gaps in information
```

### Part 4 — RAG Pipeline with FAISS
```
8 document knowledge base (ML systems domain)
     ↓
SentenceTransformer embeddings (all-MiniLM-L6-v2)
     ↓
FAISS IndexFlatIP (cosine similarity)
     ↓
Top-K retrieval → context injection → LLM answer
```

Retrieval accuracy:
- Flash Attention query → cuda_optimization_guide (0.680)
- NCCL query → nccl_tuning_guide (0.606)
- TensorRT query → tensorrt_deployment_guide (0.716)

### Part 5 — Evaluation Framework
Automated hallucination detection:

| Test | Faithfulness | Correct |
|------|-------------|---------|
| Flash Attention memory | 1.00 ✅ | ❌* |
| NCCL parameters | 1.00 ✅ | ✅ |
| TensorRT speedup | 1.00 ✅ | ✅ |
| Gradient accumulation | 1.00 ✅ | ✅ |
| Capital of Mars (OOD) | 1.00 ✅ | ✅ |

*Flash Attention: answer was faithful but 
missed exact keywords O(N)/tiling/SRAM

**Average faithfulness: 1.00/1.0 — zero hallucinations**

---

## Resume Claims Proven

| Claim | Implementation |
|-------|---------------|
| "Engineered Deep Research agentic workflow" | 4-step pipeline built and tested |
| "Orchestrating complex tool-use patterns" | 5 tools, automatic selection |
| "Automated multistep information retrieval" | Wikipedia + FAISS + LLM synthesis |
| "Developed rigorous evaluation pipelines" | Faithfulness + consistency scoring |
| "Proactively identifying hallucination issues" | 1.00 faithfulness, OOD detection |
| "Using synthetic data to stress-test" | 5 test cases including adversarial |

---

## Key Concepts

### ReAct Pattern
```python
while not done:
    thought = llm.think(context)
    action, input = parse(thought)
    observation = execute_tool(action, input)
    context += observation
    if "Final Answer" in thought:
        done = True
```

### RAG Pipeline
```python
# Retrieval
query_embedding = embedder.encode(query)
scores, indices = faiss_index.search(query_embedding, k=3)

# Augmented Generation
context = get_docs(indices)
answer  = llm(f"Answer using context:\n{context}\n\nQuestion: {query}")
```

### Faithfulness Evaluation
```python
# For each answer:
# 1. Get retrieved context
# 2. Ask LLM: "Is answer faithful to context?"
# 3. Score 0.0 (hallucinated) to 1.0 (faithful)
# 4. Flag answers below threshold
```

---

## Tech Stack
```
Google Colab AI · FAISS-CPU · SentenceTransformers
Wikipedia API · BeautifulSoup · ReAct Pattern
RAG · Hallucination Detection · Synthetic Evaluation
Python 3.12 · NumPy · Regex
```
   
---

## Part of ML Systems Optimization Suite
- ✅ Module 1 — Inference Optimization (ONNX + Quantization)
- ✅ Module 2 — CUDA Kernel Optimization (Triton + Flash Attention)
- ✅ Module 3 — Distributed Training (FSDP + NCCL)
- ✅ Module 4 — TensorRT Optimization (12.82x speedup)
- ✅ Module 5 — Agentic AI Systems (this repo)

---  

## Author
**Piyush Kunjilwar**
MS Information Systems — Northeastern University (May 2026)
[LinkedIn](https://linkedin.com/in/piyush-kunjilwar) ·
[GitHub](https://github.com/piyush12kunjilwar) ·
[Portfolio](https://piyush12kunjilwar.github.io)# Agentic-AI-Systems
