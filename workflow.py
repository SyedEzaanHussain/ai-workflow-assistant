"""
workflow.py — LangGraph state machine that orchestrates the full AI workflow.

Flow:
  extract_info → route → [summarize_and_plan | escalate] → done
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from tools import extract_customer_intent, extract_customer_sentiment, extract_urgency_level
from agents import run_crew


# ─── State Schema ────────────────────────────────────────────────────────────

class WorkflowState(TypedDict):
    customer_message: str
    intent: str
    sentiment: str
    urgency: str
    summary: str
    actions: str
    escalated: bool
    api_key: str


# ─── Node Functions ───────────────────────────────────────────────────────────

def extract_info(state: WorkflowState) -> WorkflowState:
    """Node 1: Run LangChain tools to extract intent, sentiment, and urgency."""
    message = state["customer_message"]

    intent = extract_customer_intent.invoke(message)
    sentiment = extract_customer_sentiment.invoke(message)
    urgency = extract_urgency_level.invoke(message)

    print(f"\n  [Extract] Intent={intent} | Sentiment={sentiment} | Urgency={urgency}")

    return {**state, "intent": intent, "sentiment": sentiment, "urgency": urgency}


def route_interaction(state: WorkflowState) -> str:
    """
    Conditional edge: decides whether to escalate or proceed to AI summarization.
    High urgency + negative sentiment → escalate immediately.
    """
    if state["urgency"] == "high" and state["sentiment"] == "negative":
        print("  [Router] → Escalating to human agent (high urgency + negative sentiment)")
        return "escalate"
    else:
        print("  [Router] → Sending to AI summarization pipeline")
        return "summarize"


def summarize_and_plan(state: WorkflowState) -> WorkflowState:
    """Node 2a: Run CrewAI to generate summary + follow-up actions."""
    print("  [CrewAI] Running summarizer and action planner agents...")

    result = run_crew(
        customer_message=state["customer_message"],
        intent=state["intent"],
        sentiment=state["sentiment"],
        urgency=state["urgency"],
        api_key=state["api_key"],
    )

    return {**state, "summary": result["summary"], "actions": result["actions"], "escalated": False}


def escalate(state: WorkflowState) -> WorkflowState:
    """Node 2b: Flag interaction for immediate human escalation."""
    summary = (
        f"⚠️  ESCALATED — This interaction requires immediate human attention.\n"
        f"Customer expressed strong dissatisfaction with high urgency.\n"
        f"Original message: {state['customer_message'][:120]}..."
    )
    actions = (
        "1. Assign to senior support agent within 15 minutes.\n"
        "2. Call the customer directly — do not rely on email.\n"
        "3. Log escalation in CRM and notify team lead."
    )
    return {**state, "summary": summary, "actions": actions, "escalated": True}


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    graph = StateGraph(WorkflowState)

    # Register nodes
    graph.add_node("extract_info", extract_info)
    graph.add_node("summarize_and_plan", summarize_and_plan)
    graph.add_node("escalate", escalate)

    # Entry point
    graph.set_entry_point("extract_info")

    # Conditional routing after extraction
    graph.add_conditional_edges(
        "extract_info",
        route_interaction,
        {
            "summarize": "summarize_and_plan",
            "escalate": "escalate",
        },
    )

    # Both paths end the workflow
    graph.add_edge("summarize_and_plan", END)
    graph.add_edge("escalate", END)

    return graph.compile()
