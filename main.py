"""
main.py — Entry point for the AI Workflow Assistant.

Run:
    python main.py

Make sure your .env file has OPENAI_API_KEY set.
"""

import os
from dotenv import load_dotenv
from workflow import build_workflow

load_dotenv()


def print_result(state: dict):
    print("\n" + "=" * 60)
    print("📋  WORKFLOW RESULT")
    print("=" * 60)
    print(f"Intent   : {state.get('intent', 'N/A')}")
    print(f"Sentiment: {state.get('sentiment', 'N/A')}")
    print(f"Urgency  : {state.get('urgency', 'N/A')}")
    print(f"Escalated: {'Yes ⚠️' if state.get('escalated') else 'No'}")
    print("\n📝  Summary:")
    print(state.get("summary", "N/A"))
    print("\n✅  Recommended Actions:")
    print(state.get("actions", "N/A"))
    print("=" * 60)


def run(customer_message: str):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Copy .env.example to .env and add your key.")

    print(f"\n🚀 Processing: \"{customer_message[:80]}...\"" if len(customer_message) > 80 else f"\n🚀 Processing: \"{customer_message}\"")

    workflow = build_workflow()

    initial_state = {
        "customer_message": customer_message,
        "intent": "",
        "sentiment": "",
        "urgency": "",
        "summary": "",
        "actions": "",
        "escalated": False,
        "api_key": api_key,
    }

    final_state = workflow.invoke(initial_state)
    print_result(final_state)


if __name__ == "__main__":
    # ── Demo interactions — swap these out to test different flows ──
    # Change demo_1 to demo_2 or demo_3 to test other flows
    # Case 1: Billing issue — moderate urgency, neutral tone
    demo_1 = (
        "Hi, I was charged twice for my subscription this month. "
        "I'd like to get a refund for the duplicate charge. Please follow up soon."
    )

    # Case 2: Critical escalation — angry + urgent
    demo_2 = (
        "This is absolutely unacceptable! Your platform crashed right before my client demo "
        "and I lost the deal. I need someone to call me ASAP. This is a critical emergency."
    )

    # Case 3: Upgrade inquiry — positive sentiment
    demo_3 = (
        "Hey! I've been loving the product. I want to upgrade my plan to get access "
        "to the advanced analytics features. Can you walk me through the options?"
    )

    # Run one at a time — change the variable to test others
    run(demo_1)
