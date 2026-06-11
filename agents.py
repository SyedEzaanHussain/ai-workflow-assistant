"""
agents.py — CrewAI agents that handle summarization and follow-up action generation.
"""

from crewai import Agent, Task, Crew, LLM


def build_crew(api_key: str):
    """
    Builds and returns a CrewAI crew with two agents:
      1. Summarizer — condenses a customer interaction into a clean summary.
      2. Action Planner — recommends follow-up actions based on that summary.
    """

    llm = LLM(model="openai/gpt-4o-mini", api_key=api_key)

    # --- Agent 1: Summarizer ---
    summarizer = Agent(
        role="Customer Interaction Summarizer",
        goal="Read a raw customer interaction and produce a concise, professional summary.",
        backstory=(
            "You are an expert customer success analyst. "
            "You distill lengthy customer messages into clear, actionable summaries "
            "that support teams can quickly read and act on."
        ),
        llm=llm,
        verbose=False,
    )

    # --- Agent 2: Action Planner ---
    action_planner = Agent(
        role="Follow-Up Action Planner",
        goal="Based on a customer interaction summary, recommend the best follow-up actions.",
        backstory=(
            "You are a seasoned customer operations manager. "
            "Given a summary of a customer problem, you decide what concrete steps "
            "the support team should take — such as issuing a refund, escalating to engineering, "
            "or sending a check-in email within 24 hours."
        ),
        llm=llm,
        verbose=False,
    )

    return summarizer, action_planner, llm


def run_crew(customer_message: str, intent: str, sentiment: str, urgency: str, api_key: str) -> dict:
    """
    Runs the full CrewAI pipeline for a single customer interaction.
    Returns a dict with 'summary' and 'actions'.
    """

    summarizer, action_planner, llm = build_crew(api_key)

    context = f"""
Customer Message: {customer_message}

Extracted Metadata:
- Intent: {intent}
- Sentiment: {sentiment}
- Urgency: {urgency}
"""

    # Task 1: Summarize
    summarize_task = Task(
        description=f"Summarize the following customer interaction in 2-3 sentences:\n{context}",
        expected_output="A 2-3 sentence professional summary of the customer interaction.",
        agent=summarizer,
    )

    # Task 2: Plan follow-up (depends on summary from Task 1)
    action_task = Task(
        description=(
            "Based on the customer interaction summary provided by the summarizer, "
            "list 2-3 specific, actionable follow-up steps the support team should take. "
            "Be concrete — no vague advice."
        ),
        expected_output="A numbered list of 2-3 specific follow-up actions.",
        agent=action_planner,
        context=[summarize_task],
    )

    crew = Crew(
        agents=[summarizer, action_planner],
        tasks=[summarize_task, action_task],
        verbose=False,
    )

    result = crew.kickoff()

    # Extract individual task outputs
    summary = summarize_task.output.raw if summarize_task.output else "N/A"
    actions = action_task.output.raw if action_task.output else "N/A"

    return {"summary": summary, "actions": actions}
