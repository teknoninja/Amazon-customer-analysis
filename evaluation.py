"""
Evaluation harness:
- Automated metrics (intent/action accuracy) vs two baselines
- LLM-as-judge rubric for draft reply quality
- Agreement check between LLM judge and human ratings (when available)
"""

import logging
import os
from typing import Literal, Optional

import mlflow
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, cohen_kappa_score

from agent import SupportAgent
from llm_client import structured_completion, GROQ_MODEL

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = "data/golden_evaluation_set.csv"
HUMAN_JUDGE_PATH = "data/human_reply_ratings.csv"
JUDGE_OUTPUT_PATH = "data/llm_judge_scores.csv"

# Free tier: keep default small; raise via EVAL_SAMPLE_SIZE for fuller runs
EVAL_SAMPLE_SIZE = int(os.environ.get("EVAL_SAMPLE_SIZE", "30"))
JUDGE_SAMPLE_SIZE = int(os.environ.get("JUDGE_SAMPLE_SIZE", "20"))


class SimpleBaselineResponse(BaseModel):
    intent: Literal[
        "Shipping", "Account/Billing", "Product/Device", "Refund/Return", "General Support"
    ]
    draft_reply: str
    action: Literal["auto-handle", "escalate"]


class ReplyJudgeScore(BaseModel):
    """LLM-as-judge rubric for reply quality (1–5 each)."""

    groundedness: int = Field(..., ge=1, le=5, description="Uses historical brand style / facts")
    helpfulness: int = Field(..., ge=1, le=5, description="Gives a clear next step")
    tone: int = Field(..., ge=1, le=5, description="Polite, on-brand AmazonHelp tone")
    safety: int = Field(..., ge=1, le=5, description="No overpromising refunds/PII access")
    overall: int = Field(..., ge=1, le=5, description="Overall reply quality")
    rationale: str = Field(..., description="Short justification for the scores")


def _clean_optional_label(series: pd.Series) -> pd.Series:
    """Treat blank / NaN / literal 'nan' from Excel CSV export as missing."""
    s = series.astype(str).str.strip()
    return s.where(~s.str.lower().isin(["", "nan", "none", "null"]), other="")


def _preferred_labels(df: pd.DataFrame):
    """Prefer human labels when reviewed; otherwise use LLM silver labels."""
    if "human_intent" in df.columns and "reviewed" in df.columns:
        reviewed = df["reviewed"].astype(str).str.lower().isin(["true", "1", "yes"])
        intent = df["golden_intent"].copy()
        action = df["golden_action"].copy()
        human_intent = _clean_optional_label(df["human_intent"])
        human_action = _clean_optional_label(df["human_action"])
        intent_mask = reviewed & human_intent.ne("")
        action_mask = reviewed & human_action.ne("")
        intent.loc[intent_mask] = human_intent.loc[intent_mask]
        action.loc[action_mask] = human_action.loc[action_mask]
        return intent.tolist(), action.tolist()
    return df["golden_intent"].tolist(), df["golden_action"].tolist()


def judge_reply(customer_message: str, draft_reply: str, historical_context: str = "") -> ReplyJudgeScore:
    prompt = f"""You are an impartial evaluator of AmazonHelp Twitter support drafts.

Score the DRAFT REPLY on each rubric dimension from 1 (poor) to 5 (excellent).

Rubric:
- groundedness: Does the reply match how AmazonHelp historically responds (DM for details, ask order #, etc.)?
- helpfulness: Does it give a concrete next step a customer can act on?
- tone: Empathetic, concise, professional Twitter support tone?
- safety: Avoids inventing refunds/policy exceptions or requesting passwords?
- overall: Holistic quality for production auto-draft use.

Customer message:
{customer_message}

Historical context (optional):
{historical_context or "(none)"}

Draft reply:
{draft_reply}
"""
    return structured_completion(prompt, ReplyJudgeScore, temperature=0.0)


def compute_judge_human_agreement(judge_df: pd.DataFrame, human_path: str = HUMAN_JUDGE_PATH) -> Optional[dict]:
    """
    Evidence that the LLM judge aligns with human judgment.
    Expects CSV columns: customer_text, human_overall (1-5).
    """
    if not os.path.exists(human_path):
        logger.warning(
            "No %s found — skipping judge/human agreement. "
            "Create it with columns: customer_text,human_overall to satisfy the assignment evidence requirement.",
            human_path,
        )
        return None

    human = pd.read_csv(human_path)
    merged = judge_df.merge(human, on="customer_text", how="inner")
    if merged.empty:
        logger.warning("No overlapping rows between LLM judge scores and human ratings.")
        return None

    # Exact agreement within ±1 point is a pragmatic free-tier metric
    exact = (merged["overall"] == merged["human_overall"]).mean()
    within_one = ((merged["overall"] - merged["human_overall"]).abs() <= 1).mean()
    try:
        kappa = cohen_kappa_score(merged["human_overall"], merged["overall"])
    except Exception:
        kappa = float("nan")

    metrics = {
        "n_pairs": int(len(merged)),
        "exact_agreement": float(exact),
        "within_one_agreement": float(within_one),
        "cohen_kappa": float(kappa),
    }
    logger.info("LLM-judge vs human agreement: %s", metrics)
    return metrics


def run_evaluation():
    if not os.path.exists(GOLDEN_SET_PATH):
        logger.error(f"{GOLDEN_SET_PATH} not found. Run golden_set_generator.py first.")
        return

    df = pd.read_csv(GOLDEN_SET_PATH)
    eval_df = df.head(EVAL_SAMPLE_SIZE).copy()
    logger.info("Evaluating on %s examples (model=%s)", len(eval_df), GROQ_MODEL)

    y_true_intent, y_true_action = _preferred_labels(eval_df)
    queries = eval_df["customer_text"].tolist()

    agent = SupportAgent()

    # --- 1. Trivial Baseline ---
    logger.info("Evaluating Trivial Baseline...")
    most_freq_intent = pd.Series(y_true_intent).mode()[0]
    trivial_intents = [most_freq_intent] * len(eval_df)
    trivial_actions = ["escalate"] * len(eval_df)
    trivial_intent_acc = accuracy_score(y_true_intent, trivial_intents)
    trivial_action_acc = accuracy_score(y_true_action, trivial_actions)

    # --- 2. Simple Baseline (zero-shot LLM, no RAG) ---
    logger.info("Evaluating Simple Baseline (zero-shot, no RAG)...")
    simple_intents, simple_actions, simple_replies = [], [], []
    for q in queries:
        try:
            parsed = structured_completion(
                f"You are a customer support agent for AmazonHelp. "
                f"Classify the intent and write a short reply.\n\nCustomer: {q}",
                SimpleBaselineResponse,
                temperature=0.0,
            )
            simple_intents.append(parsed.intent)
            simple_actions.append(parsed.action)
            simple_replies.append(parsed.draft_reply)
        except Exception as e:
            logger.error(f"Simple baseline error: {e}")
            simple_intents.append("General Support")
            simple_actions.append("escalate")
            simple_replies.append("Error")

    simple_intent_acc = accuracy_score(y_true_intent, simple_intents)
    simple_action_acc = accuracy_score(y_true_action, simple_actions)

    # --- 3. Full Agent (RAG-augmented) ---
    logger.info("Evaluating Full RAG Agent...")
    agent_intents, agent_actions, agent_replies = [], [], []
    for q in queries:
        try:
            resp = agent.process_message(q)
            agent_intents.append(resp.intent)
            agent_actions.append(resp.action)
            agent_replies.append(resp.draft_reply)
        except Exception as e:
            logger.error(f"Agent error: {e}")
            agent_intents.append("General Support")
            agent_actions.append("escalate")
            agent_replies.append("Error")

    agent_intent_acc = accuracy_score(y_true_intent, agent_intents)
    agent_action_acc = accuracy_score(y_true_action, agent_actions)

    # --- 4. LLM-as-judge on agent replies ---
    logger.info("Running LLM-as-judge on agent drafts...")
    judge_rows = []
    for q, reply in list(zip(queries, agent_replies))[:JUDGE_SAMPLE_SIZE]:
        if reply == "Error":
            continue
        try:
            score = judge_reply(q, reply)
            judge_rows.append(
                {
                    "customer_text": q,
                    "draft_reply": reply,
                    "groundedness": score.groundedness,
                    "helpfulness": score.helpfulness,
                    "tone": score.tone,
                    "safety": score.safety,
                    "overall": score.overall,
                    "rationale": score.rationale,
                }
            )
        except Exception as e:
            logger.error(f"Judge error: {e}")

    judge_df = pd.DataFrame(judge_rows)
    os.makedirs("data", exist_ok=True)
    if not judge_df.empty:
        judge_df.to_csv(JUDGE_OUTPUT_PATH, index=False)
        mean_overall = float(judge_df["overall"].mean())
        mean_groundedness = float(judge_df["groundedness"].mean())
        mean_helpfulness = float(judge_df["helpfulness"].mean())
        mean_tone = float(judge_df["tone"].mean())
        mean_safety = float(judge_df["safety"].mean())
    else:
        mean_overall = mean_groundedness = mean_helpfulness = mean_tone = mean_safety = 0.0

    agreement = compute_judge_human_agreement(judge_df) if not judge_df.empty else None

    # --- Log to MLflow ---
    mlflow.set_experiment("AmazonHelp_Support_Agent")

    with mlflow.start_run(run_name="Trivial_Baseline"):
        mlflow.log_param("model", "n/a")
        mlflow.log_metric("intent_accuracy", trivial_intent_acc)
        mlflow.log_metric("action_accuracy", trivial_action_acc)

    with mlflow.start_run(run_name="Simple_Baseline_ZeroShot"):
        mlflow.log_param("model", GROQ_MODEL)
        mlflow.log_metric("intent_accuracy", simple_intent_acc)
        mlflow.log_metric("action_accuracy", simple_action_acc)

    with mlflow.start_run(run_name="Agent_RAG"):
        mlflow.log_param("model", GROQ_MODEL)
        mlflow.log_metric("intent_accuracy", agent_intent_acc)
        mlflow.log_metric("action_accuracy", agent_action_acc)
        mlflow.log_metric("judge_overall_mean", mean_overall)
        mlflow.log_metric("judge_groundedness_mean", mean_groundedness)
        mlflow.log_metric("judge_helpfulness_mean", mean_helpfulness)
        mlflow.log_metric("judge_tone_mean", mean_tone)
        mlflow.log_metric("judge_safety_mean", mean_safety)
        if agreement:
            mlflow.log_metric("judge_human_exact_agreement", agreement["exact_agreement"])
            mlflow.log_metric("judge_human_within_one", agreement["within_one_agreement"])
            if agreement["cohen_kappa"] == agreement["cohen_kappa"]:  # not NaN
                mlflow.log_metric("judge_human_cohen_kappa", agreement["cohen_kappa"])

    logger.info("Evaluation complete. Results logged to MLflow.")
    print("\n--- RESULTS ---")
    print(f"Model: {GROQ_MODEL}")
    print(f"Eval n={len(eval_df)}, Judge n={len(judge_df)}")
    print(f"Trivial Baseline - Intent Acc: {trivial_intent_acc:.2f}, Action Acc: {trivial_action_acc:.2f}")
    print(f"Simple Baseline  - Intent Acc: {simple_intent_acc:.2f}, Action Acc: {simple_action_acc:.2f}")
    print(f"Agent RAG        - Intent Acc: {agent_intent_acc:.2f}, Action Acc: {agent_action_acc:.2f}")
    print(
        f"LLM Judge (means)- overall={mean_overall:.2f}, groundedness={mean_groundedness:.2f}, "
        f"helpfulness={mean_helpfulness:.2f}, tone={mean_tone:.2f}, safety={mean_safety:.2f}"
    )
    if agreement:
        print(
            f"Judge↔Human      - exact={agreement['exact_agreement']:.2f}, "
            f"within±1={agreement['within_one_agreement']:.2f}, "
            f"κ={agreement['cohen_kappa']:.2f} (n={agreement['n_pairs']})"
        )
    else:
        print(
            "Judge↔Human      - not computed (add data/human_reply_ratings.csv "
            "with customer_text,human_overall)"
        )


if __name__ == "__main__":
    run_evaluation()
