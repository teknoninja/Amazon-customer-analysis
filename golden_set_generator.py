import logging
import os

import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

from llm_client import structured_completion, GROQ_MODEL

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROCESSED_DATA_PATH = "data/amazon_help_interactions.csv"
GOLDEN_SET_PATH = "data/golden_evaluation_set.csv"
CHECKPOINT_EVERY = int(os.environ.get("GOLDEN_CHECKPOINT_EVERY", "10"))


class GoldenLabel(BaseModel):
    intent: Literal[
        "Shipping", "Account/Billing", "Product/Device", "Refund/Return", "General Support"
    ]
    expected_action: Literal["auto-handle", "escalate"]


def _build_sample(df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    if "heuristic_intent" in df.columns and len(df) >= sample_size:
        per_class = max(1, sample_size // df["heuristic_intent"].nunique())
        parts = []
        for _, group in df.groupby("heuristic_intent"):
            n = min(len(group), per_class)
            parts.append(group.sample(n=n, random_state=123))
        df_eval = pd.concat(parts).sample(frac=1, random_state=123).head(sample_size).copy()
        if len(df_eval) < sample_size:
            remaining = df.drop(df_eval.index).sample(
                n=min(sample_size - len(df_eval), len(df) - len(df_eval)),
                random_state=123,
            )
            df_eval = pd.concat([df_eval, remaining])
        return df_eval.reset_index(drop=True)

    return df.sample(n=min(sample_size, len(df)), random_state=123).reset_index(drop=True)


def _save(df_eval: pd.DataFrame) -> None:
    os.makedirs("data", exist_ok=True)
    df_eval.to_csv(GOLDEN_SET_PATH, index=False)


def generate_golden_set(sample_size: int = 200):
    """
    Create a 150–250 example golden set.

    Labels are LLM-proposed (silver labels), then intended for manual review.
    Re-running resumes from the last unfinished row if a partial CSV exists.
    See docs/GOLDEN_SET_METHODOLOGY.md for sampling + labeling protocol.
    """
    if not os.path.exists(PROCESSED_DATA_PATH):
        logger.error(f"{PROCESSED_DATA_PATH} not found. Run data_processing.py first.")
        return

    logger.info("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)
    df_eval = _build_sample(df, sample_size)

    # Resume support: if a partial file exists with matching size, continue unlabeled rows
    start_at = 0
    if os.path.exists(GOLDEN_SET_PATH):
        existing = pd.read_csv(GOLDEN_SET_PATH)
        if len(existing) == len(df_eval) and "golden_intent" in existing.columns:
            df_eval = existing
            labeled = df_eval["golden_intent"].notna() & (
                df_eval["golden_intent"].astype(str).str.len() > 0
            )
            # Rows marked as failed placeholders still count; resume only truly empty
            start_at = int(labeled.sum())
            if start_at >= len(df_eval):
                logger.info("Golden set already complete at %s (%s rows).", GOLDEN_SET_PATH, len(df_eval))
                return
            logger.info("Resuming golden set from row %s/%s", start_at + 1, len(df_eval))
        else:
            df_eval["golden_intent"] = None
            df_eval["golden_action"] = None
    else:
        df_eval["golden_intent"] = None
        df_eval["golden_action"] = None

    if "human_intent" not in df_eval.columns:
        df_eval["human_intent"] = ""
    if "human_action" not in df_eval.columns:
        df_eval["human_action"] = ""
    if "reviewed" not in df_eval.columns:
        df_eval["reviewed"] = False

    logger.info(
        "Generating golden labels for %s examples using %s (interval friendly to free tier)...",
        len(df_eval),
        GROQ_MODEL,
    )

    for i in range(start_at, len(df_eval)):
        text = str(df_eval.at[i, "customer_text"])
        try:
            label = structured_completion(
                f"You are an expert annotator for AmazonHelp customer service. "
                f"Classify the customer intent and determine if it should be "
                f"auto-handled or escalated.\n\nCustomer message: {text}",
                GoldenLabel,
                temperature=0.0,
            )
            df_eval.at[i, "golden_intent"] = label.intent
            df_eval.at[i, "golden_action"] = label.expected_action
        except Exception as e:
            logger.error("Error labeling row %s: %s", i, e)
            df_eval.at[i, "golden_intent"] = "General Support"
            df_eval.at[i, "golden_action"] = "escalate"

        done = i + 1
        if done % CHECKPOINT_EVERY == 0 or done == len(df_eval):
            _save(df_eval)
            logger.info("Checkpoint saved (%s/%s)", done, len(df_eval))

    _save(df_eval)
    logger.info("Golden evaluation set saved to %s", GOLDEN_SET_PATH)
    logger.info(
        "NOTE: Review labels manually (fill human_intent/human_action, set reviewed=True). "
        "Assignment expects hand-labelled / human-verified examples."
    )


if __name__ == "__main__":
    generate_golden_set()
