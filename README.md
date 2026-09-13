# AmazonHelp AI Support Agent

Hiver SDE Intern take-home: an AI customer-support agent for **@AmazonHelp** on Twitter.

> 📄 **Note to Reviewer:** Please see `Report.pdf` (included in this repository) for the full architectural breakdown, failure analysis, and design decisions!

For every inbound customer message the agent returns:
1. **Intent** — one of a small fixed set
2. **Draft reply** — grounded in historical AmazonHelp resolutions (RAG)
3. **Escalation decision** — `auto-handle` or `escalate`, with a stated reason

**LLM:** Groq free tier — `openai/gpt-oss-120b`  
**Embeddings:** local ChromaDB default (`all-MiniLM-L6-v2`) — no paid embedding API

---

## Reproduce headline results (< 15 minutes)

### Reviewer Fast-Track (Run Evaluation Only)
Since the database and golden set are 
already included in the repository, you can evaluate the agent immediately: 
1. Set your own GROQ_API_KEY in your .env file. I have included an .env example file 
in Repo. 
2. Activate a virtual environment venv and install dependencies: pip install -r 
requirements.txt 
3. Run the evaluation: python evaluation.py 
4. View metrics in the terminal or run mlflow ui to view the side-by-side comparison.

### B. Full Pipeline Reproduction (Optional) 
If you wish to rebuild the vector database and evaluation sets from scratch (typically this can take more time): 
1. Set your own GROQ_API_KEY and KAGGLE_TOKEN in your .env file. I have 
included an .env example file in Repo. 
2. Activate a virtual environment venv and install dependencies: pip install -r 
requirements.txt 
3. Download data from kaggle: python download_data.py 
4. Process data and build ChromaDB: python data_processing.py && python 
rag_pipeline.py 
5. Generate new silver labels: python golden_set_generator.py ,it will generate 
golden set csv file in data folder. 
6. Verify golden_set_generator.csv manually and add human intent/actions wherever 
needed the AI agent will prioritize those. 
7. Run the evaluation: python evaluation.py ,it will generate 
llm_judge_scores.csv file in data folder.  
8. Copy the customer_text column from llm_judge_scores.csv to a new csv file 
human_reply_ratings.csv and provide Human ratings . This is necessary to evaluate 
llm on human basis and compare AI response . 
9. Again run the evaluation ,python evaluation.py this time it will log the scores and 
metrics on console and in MLflow ui. 
10. (optional) Run Mlflow ui for comparison of different Results and baselines. 
11. Generate this report: python generate_report_pdf.py   
### 1. Environment
```bash
python3 -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set:
#   GROQ_API_KEY=...
#   KAGGLE_API_TOKEN=KGAT_...   # from https://www.kaggle.com/settings/api
```
### 2. Data → RAG → golden set → eval
```bash
python download_data.py      # uses KAGGLE_API_TOKEN; or place twcs.csv under data/
python data_processing.py    # AmazonHelp pairs + heuristic intents
python rag_pipeline.py       # local Chroma index (~3k samples)
python golden_set_generator.py
# Optional but recommended: manually review data/golden_evaluation_set.csv
# (see docs/GOLDEN_SET_METHODOLOGY.md)

python evaluation.py         # baselines + RAG agent + LLM judge
mlflow ui                    # optional: browse logged metrics
```

### Free-tier tips (Groq ~30 RPM / ~1k RPD for this model)
- Defaults: `EVAL_SAMPLE_SIZE=30`, `JUDGE_SAMPLE_SIZE=20`, ~2.2s between API calls
- Full golden labeling (200) ≈ a few hundred calls — run when you have quota
- For a quick agent smoke test: `python agent.py`

### Manual steps you must do yourself
1. Create a Groq key and put it in `.env`
2. Create a Kaggle API token (`KGAT_...`) at https://www.kaggle.com/settings/api and set `KAGGLE_API_TOKEN` in `.env` (or unzip the dataset into `data/`)
3. **Hand-review** ≥150 golden labels (`human_intent` / `human_action` / `reviewed`)
4. Rate ~20 agent replies in `data/human_reply_ratings.csv` (copy from `data/human_reply_ratings.example.csv`) so judge↔human agreement can be computed
5. Regenerate the PDF after you have real metrics: `python generate_report_pdf.py`

---

## Architecture

```text
Customer tweet
    │
    ├─► ChromaDB retrieve top-3 historical Q/A pairs (local embeddings)
    │
    └─► Groq openai/gpt-oss-120b
            ├─ intent
            ├─ draft_reply (tone grounded in retrieved brand replies)
            ├─ action: auto-handle | escalate
            └─ reason
```

| Module | Role |
|--------|------|
| `download_data.py` | Fetch Kaggle dataset |
| `data_processing.py` | Filter `@AmazonHelp`, build interaction pairs |
| `rag_pipeline.py` | Build / query ChromaDB |
| `llm_client.py` | Shared Groq client + throttle + JSON schema parsing |
| `agent.py` | Intent + RAG draft + escalation |
| `golden_set_generator.py` | 200-example eval set (silver → human review) |
| `evaluation.py` | Trivial + simple baselines, RAG agent, LLM judge, agreement |
| `generate_report_pdf.py` | Assignment report PDF |

**Intents:** `Shipping`, `Account/Billing`, `Product/Device`, `Refund/Return`, `General Support`

---

## Report

### Problem framing
For `@AmazonHelp`, “good” means: correct intent, a concise Twitter-length reply that mirrors how Amazon historically asks for DMs / order details, and escalation when money, account access, or high emotion is involved.

**Deliberately not built:** multi-turn thread state, live order-DB lookups, fine-tuned classifiers, production ticket routing.

### Results vs baselines
| System | What it does |
|--------|----------------|
| **Trivial** | Always predict majority intent; always escalate |
| **Simple** | Zero-shot Groq, no RAG |
| **Agent RAG** | Groq + top-3 historical AmazonHelp replies |

Metrics logged by `evaluation.py` / MLflow: intent accuracy, action accuracy, LLM-judge means (groundedness, helpfulness, tone, safety, overall), and judge↔human agreement when `data/human_reply_ratings.csv` exists.

*(Trivial: majority intent + always escalate. 
Simple: zero-shot Groq, no RAG. 
Agent RAG: Groq + top-3 historical pairs. 
Metrics from evaluation.py on n=30 golden examples (LLM judge n=18; judge vs human 
overlap n=17) 
Trivial - intent acc: 0.40 | action acc: 0.57 
Simple (zero-shot) - intent acc: 0.87 | action acc: 0.77 
Agent RAG - intent acc: 0.90 | action acc: 0.67 
LLM judge means - overall: 4.39, groundedness: 4.11, helpfulness: 4.39, tone: 4.83, safety: 
4.94 
Judge vs human - exact: 0.47, within+/-1: 1.00, Cohen kappa: 0.07 (n=17) )*

### Failure analysis (top 5)
1. **Ambiguous mentions** — bare “@AmazonHelp” → unactionable; agent may invent a helpful tone.
2. **Sarcasm / frustration** — “Great job losing my package” can look like praise to shallow classifiers.
3. **Multi-intent** — billing + shipping in one tweet; single-label taxonomy forced a primary intent.
4. **Noisy RAG neighbors** — bad historical replies can pull drafts off-policy.
5. **Heuristic escalation** — without account APIs, refund-like issues always escalate; some recoverable FAQs still escalate.

### What is misleading about my headline number?
Intent accuracy on a golden set that began as **LLM silver labels** overstates reliability if humans have not finished reviewing. The LLM-as-judge may also share style preferences with the drafting model (same Groq family), inflating reply scores. Accuracy also depends on a coarse 5-class taxonomy that collapses real multi-intent tickets. Free-tier eval uses a **subset** of the golden set, so headline numbers are noisier than a full-set run.

### What I’d do with one more week
- Finish full human review of 200 labels + expand judge agreement set to 50+
- Add a second, independent judge model for bias control
- Thread-aware context (parent tweets)
- Lightweight intent classifier to cut LLM calls on free tier
- Streamlit demo for live walkthroughs

---

## Decision log
1. **Brand = AmazonHelp** — clear transactional intents; dense historical reply patterns for RAG.
2. **LLM = Groq `openai/gpt-oss-120b`** — free tier, strong structured JSON, enough quality for draft + judge.
3. **Left Gemini / paid OpenAI** — assignment must run on free APIs for reproduction.
4. **Local MiniLM embeddings via Chroma default** — zero embedding cost / rate limits.
5. **ChromaDB persistent local store** — no Docker or hosted vector DB.
6. **JSON-object structured outputs** — stable parsing without provider-specific schema APIs.
7. **Client-side throttle (~2.2s)** — stay under ~30 RPM free limits.
8. **Default eval n=30 / judge n=20** — fit a demo inside daily request quotas; overridable via env.
9. **Silver→human golden set** — LLM proposes labels; humans must verify for assignment compliance.
10. **Intent-balanced sampling** — avoid majority-class-only golden sets.
11. **Escalation rules in prompt** — no live account API; refunds/PII-ish asks escalate by policy.
12. **Single-turn pairs only** — full Twitter graphs out of scope for MVP.
13. **LLM-as-judge rubric (5 dims)** — meets reply-quality deliverable beyond intent accuracy.
14. **Judge↔human CSV contract** — explicit evidence channel for judge alignment.
15. **MLflow logging** — comparable runs for trivial / simple / RAG agent.

---

## Citation / borrowing
- Dataset: [thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
- Stack: OpenAI Python SDK → Groq OpenAI-compatible API, ChromaDB, MLflow, scikit-learn, Pydantic
