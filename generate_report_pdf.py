"""
Generate the assignment report PDF (max ~6 pages of content).

Usage:
  python generate_report_pdf.py
  # writes docs/AmazonHelp_AI_Support_Agent_Report.pdf
"""

from pathlib import Path

from fpdf import FPDF

OUT_PATH = Path("docs/AmazonHelp_AI_Support_Agent_Report.pdf")


class ReportPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(100, 100, 100)
        self.set_x(self.l_margin)
        self.cell(0, 8, "AmazonHelp AI Support Agent", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def h1(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(20, 20, 20)
        self.multi_cell(0, 8, text)
        self.ln(2)

    def h2(self, text: str):
        self.ln(2)
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 7, text)
        self.ln(1)

    def body(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.2, text)
        self.ln(1)

    def bullet(self, text: str):
        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.2, f"- {text}")


def build_pdf() -> Path:
    pdf = ReportPDF(format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.h1("AmazonHelp AI Support Agent")
    pdf.body(
        "Hiver SDE Intern take-home. Goal: turn messy Twitter support data into a trustworthy "
        "AI agent for one brand, then prove it with evaluation. Brand: @AmazonHelp. Dataset: "
        "Kaggle thoughtvector/customer-support-on-twitter. LLM: Groq openai/gpt-oss-120b "
        "(free tier). Embeddings: local all-MiniLM-L6-v2 via ChromaDB (no paid embedding API)."
    )

    pdf.h2("1. End-to-end project walkthrough")
    pdf.body(
        "Step A - Data: download_data.py pulls the Kaggle CSV; data_processing.py keeps only "
        "AmazonHelp replies linked to the customer tweet they answered, producing "
        "customer_text / brand_text pairs plus a keyword heuristic_intent for exploration."
    )
    pdf.body(
        "Step B - Retrieval: rag_pipeline.py indexes ~3000 customer questions into a local "
        "ChromaDB collection and stores the historical AmazonHelp reply as metadata."
    )
    pdf.body(
        "Step C - Agent (agent.py): for each new message, retrieve top-3 similar past issues, "
        "then call Groq to return structured JSON: intent, draft_reply, action "
        "(auto-handle|escalate), and reason. Drafts are constrained to Twitter length and "
        "grounded in retrieved brand tone (often DM / ask for order details)."
    )
    pdf.body(
        "Step D - Golden set: golden_set_generator.py samples ~200 intent-balanced examples, "
        "proposes silver labels with Groq, and leaves human_intent / human_action / reviewed "
        "columns for mandatory manual verification (see docs/GOLDEN_SET_METHODOLOGY.md)."
    )
    pdf.body(
        "Step E - Evaluation harness: evaluation.py compares (1) trivial baseline, (2) "
        "zero-shot Groq without RAG, (3) full RAG agent on intent/action accuracy; runs an "
        "LLM-as-judge rubric on drafts; optionally measures judge vs human agreement from "
        "data/human_reply_ratings.csv; logs everything to MLflow."
    )

    pdf.h2("2. Problem framing")
    pdf.body(
        "What 'good' means for AmazonHelp: correct intent; a concise, empathetic reply that "
        "mirrors how Amazon historically resolves similar tweets; escalate when money, "
        "account access, or high emotion appears; auto-handle only clear FAQs / status asks."
    )
    pdf.body(
        "What we chose not to build: multi-turn conversation state, live order-database "
        "lookups, fine-tuned classifiers, Banking77 intent transfer, and production ticket "
        "routing. Proof of trustworthiness matters more than product surface area."
    )

    pdf.h2("3. Results vs baselines")
    pdf.body(
        "Trivial: majority intent + always escalate. Simple: zero-shot Groq, no RAG. "
        "Agent RAG: Groq + top-3 historical pairs. Metrics from evaluation.py on n=30 "
        "golden examples (LLM judge n=18; judge vs human overlap n=17):"
    )
    pdf.bullet("Trivial - intent acc: 0.40  | action acc: 0.57")
    pdf.bullet("Simple (zero-shot) - intent acc: 0.87  | action acc: 0.77")
    pdf.bullet("Agent RAG - intent acc: 0.90  | action acc: 0.67")
    pdf.bullet(
        "LLM judge means - overall: 4.39, groundedness: 4.11, helpfulness: 4.39, "
        "tone: 4.83, safety: 4.94"
    )
    pdf.bullet(
        "Judge vs human - exact: 0.47, within+/-1: 1.00, Cohen kappa: 0.07 (n=17)"
    )

    pdf.h2("4. Failure analysis (top 5 with hypotheses)")
    pdf.bullet(
        "Ambiguous mentions ('@AmazonHelp' only): unactionable; agent may invent a helpful "
        "tone. Hypothesis: missing parent-thread context."
    )
    pdf.bullet(
        "Sarcasm ('Great job losing my package'): polarity flips. Hypothesis: short tweets "
        "lack affective cues for reliable detection."
    )
    pdf.bullet(
        "Multi-intent (billing + shipping): single-label taxonomy forces one primary class. "
        "Hypothesis: mutually exclusive intents by design."
    )
    pdf.bullet(
        "Noisy RAG neighbors: generic historical replies pollute drafts. Hypothesis: "
        "embedding similarity does not equal resolution quality."
    )
    pdf.bullet(
        "Escalation over-triggering on refund keywords: FAQs still escalate. Hypothesis: "
        "no order-state API; policy prefers false escalations over silent auto-refunds."
    )

    pdf.h2("5. What is misleading about my headline number?")
    pdf.body(
        "Intent accuracy on LLM silver labels overstates quality until humans finish review. "
        "The LLM-as-judge may share style preferences with the drafter (same Groq family), "
        "inflating reply scores. Coarse 5-class intents hide multi-intent errors. Free-tier "
        "defaults (EVAL_SAMPLE_SIZE=30) make headline accuracy noisier than a full 200-example "
        "run. Always read accuracy together with judge scores and failure cases."
    )

    pdf.h2("6. What I'd do with one more week")
    pdf.body(
        "Finish human review of all 200 labels; expand human reply ratings to 50+; add a "
        "second independent judge model; restore parent-tweet context; train a cheap local "
        "intent classifier to save free-tier quota; ship a Streamlit demo for live reviews."
    )

    pdf.h2("7. Decision log (15 non-obvious choices)")
    decisions = [
        "Brand = AmazonHelp: dense transactional reply patterns suit RAG grounding.",
        "Switched Gemini -> Groq openai/gpt-oss-120b for free, reproducible inference.",
        "OpenAI-compatible JSON object mode for portable structured outputs.",
        "Local MiniLM embeddings via Chroma default: zero embedding API cost/limits.",
        "Client throttle (~2.2s) to stay under ~30 RPM free limits.",
        "Default eval n=30 / judge n=20; overridable via env for fuller runs.",
        "Silver labels then human review to reach 150-250 examples practically.",
        "Intent-balanced sampling to avoid majority-class-only golden sets.",
        "Escalation policy encoded in prompt (no live account/order API).",
        "Single-turn Q->A pairs only; full Twitter graphs out of MVP scope.",
        "Five-dimension LLM-as-judge rubric for reply quality deliverable.",
        "Explicit human_reply_ratings.csv contract for judge alignment evidence.",
        "MLflow for side-by-side trivial / simple / RAG comparison.",
        "Prefer false-positive escalations over silent auto-handling of money issues.",
        "Separate methodology doc so reviewers can audit labeling choices.",
    ]
    for d in decisions:
        pdf.bullet(d)

    pdf.h2("8. Manual setup (free tier) and submission")
    pdf.body(
        "1) Create Groq key at console.groq.com/keys and set GROQ_API_KEY in .env. "
        "2) Install: python3 -m venv venv && source venv/bin/activate && "
        "pip install -r requirements.txt. "
        "3) Data: python download_data.py (or unzip twcs.csv into data/). "
        "4) python data_processing.py && python rag_pipeline.py. "
        "5) python golden_set_generator.py then hand-review labels. "
        "6) Copy data/human_reply_ratings.example.csv -> human_reply_ratings.csv and score "
        "agent drafts. "
        "7) python evaluation.py && python generate_report_pdf.py. "
        "Submit via Notion form only (repo link + this report). Do not email."
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT_PATH))
    return OUT_PATH


if __name__ == "__main__":
    path = build_pdf()
    print(f"Wrote {path}")
