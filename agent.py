import logging

from pydantic import BaseModel, Field
from typing import Literal
from rag_pipeline import RAGRetriever
from llm_client import structured_completion, GROQ_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentResponse(BaseModel):
    intent: Literal[
        "Shipping", "Account/Billing", "Product/Device", "Refund/Return", "General Support"
    ] = Field(
        ...,
        description="Classify the intent of the customer message into one of the specified categories.",
    )
    draft_reply: str = Field(
        ...,
        description=(
            "A drafted reply grounded in how the brand has historically resolved similar issues. "
            "Must be polite, helpful, and concise."
        ),
    )
    action: Literal["auto-handle", "escalate"] = Field(
        ...,
        description="Decide whether the message should be auto-handled or escalated to a human.",
    )
    reason: str = Field(
        ...,
        description="A stated reason for the chosen action (auto-handle or escalate).",
    )


class SupportAgent:
    def __init__(self):
        self.retriever = RAGRetriever()
        logger.info("SupportAgent ready (model=%s)", GROQ_MODEL)

    def process_message(self, customer_message: str) -> AgentResponse:
        # 1. Retrieve similar historical issues
        context = self.retriever.retrieve(customer_message, k=3)

        # 2. Format context for the LLM
        context_str = ""
        for i, c in enumerate(context):
            context_str += (
                f"\nExample {i + 1}:\n"
                f"Customer: {c['similar_customer_question']}\n"
                f"AmazonHelp: {c['historical_brand_reply']}\n"
            )

        prompt = f"""You are an expert AI customer support agent for @AmazonHelp on Twitter.
Your job is to:
1. Classify the customer's intent into one of: "Shipping", "Account/Billing", "Product/Device", "Refund/Return", "General Support".
2. Draft a reply based on how the brand historically resolved similar issues.
3. Decide whether this can be "auto-handle"d or needs to "escalate" to a human.
   - Escalate if: the issue requires accessing sensitive account info, processing a refund, or if the customer is extremely angry.
   - Auto-handle if: it is a general question, standard shipping update, or can be solved with a generic instruction.
4. Provide a stated reason for your decision.

Here is some context of how similar issues were handled historically:
{context_str}

Use this historical context to ground your drafted reply in the brand's tone. Keep the reply short (under 280 characters) as it's for Twitter.

Customer Message: {customer_message}
"""

        # 3. Call Groq with structured JSON output
        return structured_completion(prompt, AgentResponse, temperature=0.2)


if __name__ == "__main__":
    agent = SupportAgent()
    test_msg = (
        "My package was supposed to arrive yesterday but the tracking hasn't updated. Where is it?"
    )
    print(f"\nTesting message: {test_msg}\n")
    result = agent.process_message(test_msg)
    print("=" * 60)
    print(f"  Intent     : {result.intent}")
    print(f"  Action     : {result.action}")
    print(f"  Reason     : {result.reason}")
    print(f"  Draft Reply: {result.draft_reply}")
    print("=" * 60)
