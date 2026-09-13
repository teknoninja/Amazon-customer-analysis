# Golden Set Sampling & Labeling Methodology

## Goal
Produce a **150–250 example** evaluation set for `@AmazonHelp` that can support
intent accuracy, escalation accuracy, and reply-quality judging.

## Sampling
1. Start from processed AmazonHelp customer↔brand pairs in `data/amazon_help_interactions.csv`.
2. Draw ~200 examples with `random_state=123`.
3. Prefer **intent-balanced sampling** across heuristic buckets
   (`Shipping`, `Account/Billing`, `Product/Device`, `Refund/Return`, `General Support`)
   so the set is not dominated by one frequent class.
4. Keep raw `customer_text` and the historical `brand_text` for grounding checks.

## Labeling protocol
1. **Silver labels (LLM):** `golden_set_generator.py` proposes `golden_intent` and
   `golden_action` using Groq (`openai/gpt-oss-120b`) at temperature 0.
2. **Human review (required for assignment spirit):** open the CSV and for each row
   (or a large reviewed subset ≥150):
   - Fill `human_intent` / `human_action` when you disagree with the silver label
   - Set `reviewed=True`
3. Evaluation prefers human labels when `reviewed=True`; otherwise falls back to silver labels.
4. Document disagreements — ambiguous tweets (sarcasm, multi-intent) are expected failure modes.

## Escalation guidelines used by annotators
- **Escalate:** refunds/returns, account access, billing disputes, angry/threat-to-churn,
  anything needing private order lookup.
- **Auto-handle:** status FAQs, how-to questions answerable with a public link / DM-ask pattern.

## LLM-as-judge agreement evidence
Create `data/human_reply_ratings.csv` with columns:
```text
customer_text,human_overall
```
where `human_overall` is an integer 1–5 for the agent's draft reply.
`evaluation.py` merges these with LLM judge scores and reports exact agreement,
within-±1 agreement, and Cohen's κ.
