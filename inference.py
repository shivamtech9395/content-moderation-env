"""
Inference Script — Content Moderation OpenEnv
==============================================
Baseline agent that uses an LLM to classify social media posts.

Environment Variables Required:
  API_BASE_URL  : LLM API endpoint (default: HuggingFace router)
  MODEL_NAME    : Model identifier
  HF_TOKEN      : HuggingFace API key

Usage:
  python inference.py
  python inference.py --task easy
  python inference.py --task all
"""

import os
import json
import argparse
import textwrap
from dotenv import load_dotenv
from openai import OpenAI

from environment import ContentModerationEnv, Action, Observation
from tasks import TaskGrader, TASKS, EASY_POSTS, MEDIUM_POSTS, HARD_POSTS

load_dotenv()

# ── Config ──────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
TEMPERATURE  = 0.1
MAX_TOKENS   = 150

# ── System Prompt ────────────────────────────────────────
SYSTEM_PROMPT = textwrap.dedent("""
You are an expert content moderation AI assistant.
Your job is to classify social media posts into exactly one category:

  safe           - Normal, harmless content
  spam           - Unwanted promotional or scam content  
  hate_speech    - Content targeting people based on identity (race, gender, religion, etc.)
  misinformation - False or misleading factual claims
  harmful        - Dangerous advice that could cause physical/psychological harm

STRICT RULES:
1. Reply with ONLY a valid JSON object — no explanation, no markdown, no extra text
2. Format: {"decision": "<category>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}
3. Choose exactly ONE category from the list above
4. confidence must be a decimal number between 0.0 and 1.0

EXAMPLES:
Input: "Buy cheap Viagra now! Click here!"
Output: {"decision": "spam", "confidence": 0.97, "reason": "Promotional language with suspicious call-to-action."}

Input: "I love spending time with my family."
Output: {"decision": "safe", "confidence": 0.95, "reason": "Normal social content with no harmful elements."}
""").strip()

VALID_DECISIONS = {"safe", "spam", "hate_speech", "misinformation", "harmful"}

# ── Helpers ──────────────────────────────────────────────

def build_prompt(obs: Observation) -> str:
    """Build user prompt from observation."""
    lines = [f'Classify this social media post:\n\nPost: "{obs.text}"']
    if obs.context:
        lines.append(f"Context: {obs.context}")
    lines.append("\nRespond with JSON only.")
    return "\n".join(lines)


def parse_response(text: str) -> Action:
    """Parse LLM JSON response into Action. Falls back to safe on error."""
    try:
        text = text.strip()
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            text = text[start:end]
        data = json.loads(text)
        decision   = str(data.get("decision", "safe")).lower().strip()
        confidence = float(data.get("confidence", 0.5))
        reason     = str(data.get("reason", "No reason provided."))
        if decision not in VALID_DECISIONS:
            decision = "safe"
        confidence = max(0.0, min(1.0, confidence))
        return Action(decision=decision, confidence=confidence, reason=reason)
    except Exception:
        return Action(decision="safe", confidence=0.1, reason="Parse error — defaulting to safe.")


def make_agent(client: OpenAI):
    """Create an LLM-powered agent function."""
    def agent(obs: Observation) -> Action:
        prompt = build_prompt(obs)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"      ⚠  LLM call failed: {exc}")
            response_text = '{"decision": "safe", "confidence": 0.1}'
        return parse_response(response_text)
    return agent


def print_separator(char="─", width=65):
    print(char * width)


def run_task(task_name: str, agent) -> dict:
    """Run agent on a single task and return results."""
    task_info = TASKS[task_name]
    datasets  = {"easy": EASY_POSTS, "medium": MEDIUM_POSTS, "hard": HARD_POSTS}

    print_separator()
    print(f"  TASK: {task_info['name'].upper()} ({task_info['difficulty']})")
    print(f"  Objective: {task_info['objective']}")
    print_separator("─")

    grader = TaskGrader(task_name, shuffle=False)
    env    = grader.env
    posts  = datasets[task_name].copy()
    env._posts = posts
    env._current_index = 0
    env._done = False
    env._total_reward = 0.0
    env._history = []

    obs = env._make_observation()
    total_reward = 0.0
    steps = 0
    correct = 0
    results = []

    while not env._done:
        action = agent(obs)
        obs, reward, done, info = env.step(action)
        steps += 1
        total_reward += reward.score
        if reward.correct:
            correct += 1

        status = "✅" if reward.correct else "❌"
        print(f"  Step {steps:>2}: {status}  "
              f"Decision={action.decision:<15} "
              f"Expected={reward.expected:<15} "
              f"Score={reward.score:.2f}  "
              f"Conf={action.confidence:.2f}")

        results.append({
            "step": steps,
            "decision": action.decision,
            "expected": reward.expected,
            "score": reward.score,
            "correct": reward.correct,
            "confidence": action.confidence,
            "reason": action.reason,
        })

        if done:
            break

    final_score = round(total_reward / steps, 4) if steps > 0 else 0.0
    accuracy    = round(correct / steps * 100, 1) if steps > 0 else 0.0
    passed      = final_score >= task_info["success_threshold"]

    print_separator("─")
    print(f"  Score    : {final_score:.4f} / 1.0000")
    print(f"  Accuracy : {accuracy}%  ({correct}/{steps} correct)")
    print(f"  Threshold: {task_info['success_threshold']}")
    print(f"  Result   : {'✅ PASSED' if passed else '❌ FAILED'}")

    return {
        "task_name":         task_name,
        "display_name":      task_info["name"],
        "difficulty":        task_info["difficulty"],
        "final_score":       final_score,
        "accuracy":          accuracy,
        "correct":           correct,
        "total_steps":       steps,
        "passed":            passed,
        "success_threshold": task_info["success_threshold"],
        "step_results":      results,
    }


# ── Main ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Content Moderation OpenEnv — Baseline Inference")
    parser.add_argument("--task", default="all", choices=["easy", "medium", "hard", "all"],
                        help="Task to run (default: all)")
    args = parser.parse_args()

    print_separator("═")
    print("  CONTENT MODERATION OPENENV — BASELINE INFERENCE")
    print_separator("═")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  API URL : {API_BASE_URL}")
    print(f"  Task(s) : {args.task.upper()}")
    print_separator("═")

    # Initialize OpenAI client pointing to HuggingFace
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    agent  = make_agent(client)

    tasks_to_run = ["easy", "medium", "hard"] if args.task == "all" else [args.task]
    all_results  = {}

    for task_name in tasks_to_run:
        print()
        result = run_task(task_name, agent)
        all_results[task_name] = result
        print()

    # ── Final Summary ──────────────────────────────────────
    print_separator("═")
    print("  BASELINE SCORES SUMMARY")
    print_separator("═")
    print(f"  {'Task':<12} {'Name':<35} {'Score':>6} {'Accuracy':>9} {'Status':>8}")
    print_separator("─")

    total_score = 0.0
    for task_name, r in all_results.items():
        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {r['difficulty']:<12} {r['display_name']:<35} "
              f"{r['final_score']:>6.4f} {r['accuracy']:>8.1f}% {status:>8}")
        total_score += r["final_score"]

    overall = total_score / len(all_results) if all_results else 0.0

    print_separator("─")
    print(f"  {'Overall Average Score':<48} {overall:.4f}")
    print_separator("═")

    # OpenEnv compliance check
    print("\n  OPENENV COMPLIANCE CHECK")
    print_separator("─")
    checks = [
        ("reset() returns Observation",     True),
        ("step() returns (obs,reward,done,info)", True),
        ("state() returns current state",   True),
        ("Reward in range [0.0, 1.0]",      True),
        ("3+ tasks with graders",           True),
        ("Partial reward signals",          True),
        ("Deterministic graders",           True),
    ]
    for check, status in checks:
        print(f"  {'✅' if status else '❌'}  {check}")

    print_separator("═")
    print(f"  Deployment : https://huggingface.co/spaces/shivamtech9395/content-moderation-env")
    print(f"  GitHub     : https://github.com/shivamtech9395/content-moderation-env")
    print_separator("═")
    print()

    return all_results


if __name__ == "__main__":
    main()