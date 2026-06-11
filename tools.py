"""
tools.py — LangChain tools for information extraction from customer interactions.
"""

from langchain.tools import tool


@tool
def extract_customer_intent(text: str) -> str:
    """
    Extracts the primary intent or problem from a raw customer message.
    Returns a short label like 'billing issue', 'technical support', 'cancellation', etc.
    """
    text_lower = text.lower()

    if any(word in text_lower for word in ["bill", "charge", "invoice", "payment", "refund"]):
        return "billing issue"
    elif any(word in text_lower for word in ["cancel", "unsubscribe", "stop", "quit"]):
        return "cancellation request"
    elif any(word in text_lower for word in ["bug", "error", "crash", "broken", "not working", "issue"]):
        return "technical support"
    elif any(word in text_lower for word in ["upgrade", "plan", "feature", "premium"]):
        return "upgrade inquiry"
    else:
        return "general inquiry"


@tool
def extract_customer_sentiment(text: str) -> str:
    """
    Detects the emotional tone of a customer message.
    Returns 'positive', 'neutral', or 'negative'.
    """
    text_lower = text.lower()

    negative_words = ["angry", "frustrated", "terrible", "worst", "hate", "disappointed", "upset", "unacceptable"]
    positive_words = ["happy", "great", "love", "excellent", "amazing", "thank", "appreciate", "pleased"]

    neg_score = sum(1 for w in negative_words if w in text_lower)
    pos_score = sum(1 for w in positive_words if w in text_lower)

    if neg_score > pos_score:
        return "negative"
    elif pos_score > neg_score:
        return "positive"
    else:
        return "neutral"


@tool
def extract_urgency_level(text: str) -> str:
    """
    Determines how urgently a customer needs a response.
    Returns 'high', 'medium', or 'low'.
    """
    text_lower = text.lower()

    high_urgency = ["asap", "urgent", "immediately", "now", "critical", "emergency", "deadline"]
    medium_urgency = ["soon", "today", "this week", "follow up", "waiting"]

    if any(word in text_lower for word in high_urgency):
        return "high"
    elif any(word in text_lower for word in medium_urgency):
        return "medium"
    else:
        return "low"
