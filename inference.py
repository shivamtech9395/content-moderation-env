"""
Inference Script
================
AI Agent that uses an LLM to moderate social media posts.
Reads credentials from environment variables:
  - API_BASE_URL  : LLM API endpoint
  - MODEL_NAME    : Model to use
  - HF_TOKEN      : Hugging Face API key

Run:
  python inference.py
"""
from dotenv import load_dotenv
load_dotenv()
import os
import json
from openai import OpenAI
from environment import ContentModerationEnv, Action, Observation
from tasks import TaskGrader, TASKS

# ─────────────────────────────────────────────
#  Config from environment variables
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "your-hf-token-here")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")

TEMPERATURE  = 0.1
MAX_TOKENS   = 100

# ─────────────────────────────────────────────
#  System Prompt — tells the LLM what to do
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a content moderation AI assistant.
Your job is to classify social media posts into one of these categories:

  - safe          : Normal, harmless content
  - spam          : Unwanted promotional or scam content
  - hate_speech   : Content targeting people based on identity
  - misinformation: False or misleading factual claims
  - harmful       : Dangerous advice or content

Rules:
1. Reply with ONLY a JSON object — no explanation, no extra text.
2. Format: {"decision": "<category>", "confidence": <0.0 to 1.0>, "reason": "<one sentence>"}
3. Choose exactly one category from the list above.
4. confidence must be a number between 0.0 and 1.0

Example reply:
{"decision": "spam", "confidence": 0.95, "reason": "Offers prize in exchange for personal info."}
"""


# ─────────────────────────────────────────────
#  Build user prompt from observation
# ─────────────────────────────────────────────

def build_prompt(obs: Observation) -> str:
    prompt = f"""Classify this social media post:

Post: "{obs.text}"
"""
    if obs.context:
        prompt += f'Context: {obs.context}\n'
    prompt += '\nReply with JSON only.'
    return prompt


# ─────────────────────────────────────────────
#  Parse LLM response into Action
# ─────────────────────────────────────────────

VALID_DECISIONS = {"safe", "spam", "hate_speech", "misinformation", "harmful"}

def parse_response(response_text: str) -> Action:
    """Parse LLM JSON response into an Action object."""
    try:
        # Clean up response
        text = response_text.strip()
        # Find JSON object in response
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        data = json.loads(text)
        decision   = str(data.get("decision", "safe")).lower().strip()
        confidence = float(data.get("confidence", 0.5))
        reason     = str(data.get("reason", ""))

        if decision not in VALID_DECISIONS:
            decision = "safe"

        confidence = max(0.0, min(1.0, confidence))
        return Action(decision=decision, confidence=confidence, reason=reason)

    except Exception:
        # Fallback if parsing fails
        return Action(decision="safe", confidence=0.3, reason="Parse error — defaulting to safe")


# ─────────────────────────────────────────────
#  LLM Agent Function
# ─────────────────────────────────────────────

def make_agent(client: OpenAI):
    """Returns an agent function that uses the LLM to classify posts."""

    def agent(obs: Observation) -> Action:
        user_prompt = build_prompt(obs)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""

        except Exception as exc:
            print(f"    ⚠️  LLM call failed: {exc}")
            response_text = '{"decision": "safe", "confidence": 0.1}'

        action = parse_response(response_text)
        return action

    return agent


# ─────────────────────────────────────────────
#  Main — Run agent on all 3 tasks
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Content Moderation — AI Agent Baseline Inference")
    print("=" * 60)
    print(f"  Model      : {MODEL_NAME}")
    print(f"  API URL    : {API_BASE_URL}")
    print("=" * 60)

    # Initialize OpenAI client pointing to HuggingFace router
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    agent = make_agent(client)
    all_scores = {}

    for task_name in ["easy", "medium", "hard"]:
        task_info = TASKS[task_name]
        print(f"\n📋 Running Task: {task_info['name']} ({task_info['difficulty']})")
        print(f"   Objective : {task_info['objective']}")
        print("-" * 60)

        grader = TaskGrader(task_name, shuffle=False)
        result = grader.run_with_agent(agent)

        print(f"\n   Steps completed : {result['total_steps']}")
        print(f"   Final Score     : {result['final_score']}")
        print(f"   Result          : {result['summary']}")

        # Print step-by-step results
        print("\n   Step Details:")
        for step in result["step_results"]:
            status = "✅" if step["correct"] else "❌"
            print(
                f"     Step {step['step']:>2}: {status} "
                f"Decided={step['decision']:<15} "
                f"Expected={step['expected']:<15} "
                f"Score={step['score']:.2f}"
            )

        all_scores[task_name] = result["final_score"]

    # Final summary
    print("\n" + "=" * 60)
    print("  BASELINE SCORES SUMMARY")
    print("=" * 60)
    for task_name, score in all_scores.items():
        task_info = TASKS[task_name]
        status = "✅ PASS" if score >= task_info["success_threshold"] else "❌ FAIL"
        print(f"  {task_info['difficulty']:<8} | {task_info['name']:<35} | Score: {score:.2f} | {status}")

    overall = sum(all_scores.values()) / len(all_scores)
    print("-" * 60)
    print(f"  Overall Average Score: {overall:.2f}")
    print("=" * 60)

    return all_scores


if __name__ == "__main__":
    main()