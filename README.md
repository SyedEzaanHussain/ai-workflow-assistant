# AI Workflow Assistant

An AI-powered system that automates customer interaction handling — extracting intent, routing interactions, summarizing conversations, and generating follow-up actions using LangChain, LangGraph, and CrewAI.

---

## What Problem Does This Solve?

In customer support, agents manually read every incoming message, figure out urgency, write summaries, and decide next steps. This is slow, inconsistent, and doesn't scale.

This project automates that entire triage pipeline using AI agents and an orchestrated workflow.

---

## How It Works — Step by Step

```
Customer Message
      │
      ▼
[LangChain Tools] ──── Extract intent, sentiment, urgency
      │
      ▼
[LangGraph Router] ─── Is it high urgency + negative sentiment?
      │                        │
      │ No                     │ Yes
      ▼                        ▼
[CrewAI Agents]          [Escalation Node]
Summarize + Plan         Flag for human agent
      │                        │
      └──────────┬─────────────┘
                 ▼
          Final Output:
          Summary + Actions
```

---

## Tech Stack

| Technology | Version | Role |
|------------|---------|------|
| Python | 3.10+ | Core language |
| LangChain | 0.3.x | Information extraction tools |
| LangGraph | 0.4.x | Workflow orchestration (state machine) |
| CrewAI | 0.130.x | Multi-agent summarization and planning |
| OpenAI GPT-4o-mini | latest | LLM powering the agents |
| python-dotenv | 1.x | Environment variable management |

---

## Project Structure

```
ai_workflow_assistant/
│
├── tools.py          # LangChain @tool functions — extract intent, sentiment, urgency
├── agents.py         # CrewAI agents — Summarizer and Action Planner
├── workflow.py       # LangGraph state machine — routes and orchestrates everything
├── main.py           # Entry point — run this to see the full pipeline in action
├── requirements.txt  # All dependencies
└── .env.example      # Copy this to .env and add your OpenAI API key
```

---

## File-by-File Breakdown

### `tools.py` — Information Extraction (LangChain)
Three LangChain `@tool` decorated functions that analyze a raw customer message:

- **`extract_customer_intent`** — classifies the message as billing issue, technical support, cancellation request, upgrade inquiry, or general inquiry
- **`extract_customer_sentiment`** — detects emotional tone: positive, neutral, or negative
- **`extract_urgency_level`** — determines priority: high, medium, or low

These are rule-based for speed and clarity, but can be swapped for LLM chains in production.

---

### `agents.py` — AI Agents (CrewAI)
Two specialized agents that work in sequence:

**Summarizer Agent**
- Role: Customer Interaction Summarizer
- Goal: Condense a raw customer message into a 2-3 sentence professional summary
- Why CrewAI: Agents have defined roles and backstories, making their output more focused and consistent than a raw LLM call

**Action Planner Agent**
- Role: Follow-Up Action Planner
- Goal: Read the summary and output 2-3 concrete, actionable next steps
- Receives the Summarizer's output as context (task chaining)

---

### `workflow.py` — State Machine (LangGraph)
The brain of the system. Uses a `StateGraph` with a typed state dictionary.

**Nodes:**
- `extract_info` — runs the LangChain tools
- `summarize_and_plan` — runs the CrewAI crew
- `escalate` — flags urgent/negative interactions for immediate human handling

**Conditional Edge (the routing logic):**
```python
if urgency == "high" and sentiment == "negative":
    → escalate
else:
    → summarize_and_plan
```

This is why LangGraph is used — it manages stateful branching cleanly, something a simple function pipeline can't do.

---

### `main.py` — Entry Point
Three demo customer messages covering different scenarios:

| Demo | Situation | Expected Route |
|------|-----------|----------------|
| `demo_1` | Duplicate billing charge | AI summary + actions |
| `demo_2` | Platform crashed before client demo (angry, urgent) | Escalation |
| `demo_3` | Wants to upgrade plan (happy) | AI summary + actions |

---

## Setup and Run

**1. Clone the repo**
```bash
git clone https://github.com/SyedEzaanHussain/ai-workflow-assistant.git
cd ai-workflow-assistant
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add your OpenAI API key**
```bash
cp .env.example .env
# Open .env and replace "your_openai_api_key_here" with your actual key
```

**4. Run**
```bash
python main.py
```

---

## Sample Output

```
🚀 Processing: "Hi, I was charged twice for my subscription this month..."

  [Extract] Intent=billing issue | Sentiment=neutral | Urgency=medium
  [Router] → Sending to AI summarization pipeline
  [CrewAI] Running summarizer and action planner agents...

============================================================
📋  WORKFLOW RESULT
============================================================
Intent   : billing issue
Sentiment: neutral
Urgency  : medium
Escalated: No

📝  Summary:
The customer reports being charged twice for their subscription this month
and is requesting a refund for the duplicate transaction. They are not
expressing strong frustration but expect a timely response.

✅  Recommended Actions:
1. Verify duplicate charge in billing system and initiate refund within 24 hours.
2. Send a confirmation email to the customer once refund is processed.
3. Check if the double-charge is a recurring system issue and flag to the billing team.
============================================================
```

---

## Key Concepts — Interview Reference

**Why LangGraph instead of just calling functions in order?**
LangGraph manages *state* across steps and supports *conditional branching*. A plain function pipeline would require messy if/else logic at the top level. LangGraph keeps each node clean and the routing logic separate and explicit.

**Why CrewAI instead of one LLM call?**
Single LLM calls produce generic output. CrewAI agents have roles, goals, and backstories that constrain behavior — the Summarizer focuses only on summarizing, the Action Planner focuses only on next steps. They also pass context between tasks, which produces better chained output than one large prompt.

**Why LangChain tools?**
The `@tool` decorator makes extraction functions reusable across different chains and agents. They can be plugged into any LangChain agent or LangGraph node without rewriting logic.

**What is a LangGraph StateGraph?**
It is a directed graph where each node is a Python function that receives a typed state dictionary, modifies it, and returns it. Edges define the flow. Conditional edges let the graph branch based on state values.

**What is CrewAI task context?**
When you pass `context=[task_1]` to `task_2`, CrewAI automatically feeds the output of task_1 as input context to task_2. This is how the Action Planner reads the Summarizer's output without you manually wiring it.

---

## Possible Extensions

- Connect to a real CRM (HubSpot, Salesforce) to pull live customer data
- Add a memory layer so agents remember past interactions with the same customer
- Replace rule-based tools with LLM chains for more nuanced extraction
- Add a Slack or email integration to send follow-up actions automatically
- Build a dashboard UI to visualize workflows in real time

---

## Author

**Syed Ezaan Hussain**  
Computer Science — FAST NUCES  
[GitHub](https://github.com/SyedEzaanHussain)
