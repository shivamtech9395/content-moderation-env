"""
Tasks Definition
================
Defines 3 tasks for the Content Moderation Environment:
  - Task 1 (Easy)   : Spam Detection
  - Task 2 (Medium) : Hate Speech Identification  
  - Task 3 (Hard)   : Misinformation & Subtle Harmful Content Detection
"""

from typing import Dict, List
from environment import ContentModerationEnv, Action, EASY_POSTS, MEDIUM_POSTS, HARD_POSTS


TASKS = {
    "easy": {
        "name": "Spam Detection",
        "difficulty": "Easy",
        "description": (
            "Identify obvious spam posts from social media. "
            "Posts contain clear spam signals like prize offers, "
            "suspicious links, and urgency tricks."
        ),
        "objective": "Classify each post as 'safe' or 'spam'.",
        "success_threshold": 0.7,
        "allowed_actions": ["safe", "spam"],
    },
    "medium": {
        "name": "Hate Speech Identification",
        "difficulty": "Medium",
        "description": (
            "Detect hate speech in social media posts. "
            "Posts may target people based on race, gender, religion, or nationality. "
            "Context is provided to help distinguish genuine discussion from hate."
        ),
        "objective": "Classify each post as 'safe' or 'hate_speech'.",
        "success_threshold": 0.65,
        "allowed_actions": ["safe", "hate_speech"],
    },
    "hard": {
        "name": "Misinformation & Harmful Content Detection",
        "difficulty": "Hard",
        "description": (
            "Detect subtle misinformation and harmful content. "
            "Posts may look like real news or scientific claims but contain "
            "dangerous false information. Context clues are critical."
        ),
        "objective": "Classify each post as 'safe', 'misinformation', or 'harmful'.",
        "success_threshold": 0.55,
        "allowed_actions": ["safe", "misinformation", "harmful"],
    },
}


class TaskGrader:
    """
    Runs an agent through a task and produces a final grade.
    Grading is fully deterministic when shuffle=False.
    """

    def __init__(self, task_name: str, shuffle: bool = False):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Choose: easy, medium, hard")
        self.task_name = task_name
        self.task_info = TASKS[task_name]
        self.env = ContentModerationEnv(task_name=task_name)
        self.shuffle = shuffle

    def _setup(self):
        """Reset environment with or without shuffle."""
        import random
        datasets = {"easy": EASY_POSTS, "medium": MEDIUM_POSTS, "hard": HARD_POSTS}
        posts = datasets[self.task_name].copy()
        if self.shuffle:
            random.shuffle(posts)
        self.env._posts = posts
        self.env._current_index = 0
        self.env._done = False
        self.env._total_reward = 0.0
        self.env._history = []

    def run_with_decisions(self, decisions: List[str]) -> Dict:
        """Grade a list of pre-made decisions. Deterministic — no shuffle."""
        self._setup()
        total_reward = 0.0
        steps = 0
        results = []

        for decision in decisions:
            if self.env._done:
                break
            action = Action(decision=decision, confidence=0.9)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward.score
            steps += 1
            results.append({
                "step": steps,
                "decision": decision,
                "expected": reward.expected,
                "score": reward.score,
                "correct": reward.correct,
                "feedback": reward.feedback,
            })

        final_score = round(total_reward / steps, 4) if steps > 0 else 0.0
        passed = final_score >= self.task_info["success_threshold"]

        return {
            "task_name": self.task_name,
            "task_display_name": self.task_info["name"],
            "difficulty": self.task_info["difficulty"],
            "final_score": final_score,
            "success_threshold": self.task_info["success_threshold"],
            "passed": passed,
            "total_steps": steps,
            "step_results": results,
            "summary": (
                f"✅ PASSED — Score {final_score:.2f} >= threshold {self.task_info['success_threshold']}"
                if passed else
                f"❌ FAILED — Score {final_score:.2f} < threshold {self.task_info['success_threshold']}"
            ),
        }

    def run_with_agent(self, agent_fn) -> Dict:
        """Grade a live agent function. agent_fn(obs) -> Action"""
        self._setup()
        self.env._posts = self.env._posts  # already set
        total_reward = 0.0
        steps = 0
        results = []

        obs = self.env._make_observation()
        while not self.env._done:
            action = agent_fn(obs)
            if not isinstance(action, Action):
                action = Action(decision=str(action), confidence=0.5)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward.score
            steps += 1
            results.append({
                "step": steps,
                "decision": action.decision,
                "expected": reward.expected,
                "score": reward.score,
                "correct": reward.correct,
                "feedback": reward.feedback,
            })
            if done:
                break

        final_score = round(total_reward / steps, 4) if steps > 0 else 0.0
        passed = final_score >= self.task_info["success_threshold"]

        return {
            "task_name": self.task_name,
            "task_display_name": self.task_info["name"],
            "difficulty": self.task_info["difficulty"],
            "final_score": final_score,
            "success_threshold": self.task_info["success_threshold"],
            "passed": passed,
            "total_steps": steps,
            "step_results": results,
            "summary": (
                f"✅ PASSED — Score {final_score:.2f} >= threshold {self.task_info['success_threshold']}"
                if passed else
                f"❌ FAILED — Score {final_score:.2f} < threshold {self.task_info['success_threshold']}"
            ),
        }


if __name__ == "__main__":
    print("=" * 55)
    print("  Content Moderation Environment — Task Grader Test")
    print("=" * 55)

    # Perfect agent — uses ground truth labels
    easy_answers   = [p["label"] for p in EASY_POSTS]
    medium_answers = [p["label"] for p in MEDIUM_POSTS]
    hard_answers   = [p["label"] for p in HARD_POSTS]

    print("\n📋 TASK 1: Spam Detection (Easy)")
    g1 = TaskGrader("easy", shuffle=False)
    r1 = g1.run_with_decisions(easy_answers)
    print(f"  Final Score : {r1['final_score']}")
    print(f"  Result      : {r1['summary']}")

    print("\n📋 TASK 2: Hate Speech (Medium)")
    g2 = TaskGrader("medium", shuffle=False)
    r2 = g2.run_with_decisions(medium_answers)
    print(f"  Final Score : {r2['final_score']}")
    print(f"  Result      : {r2['summary']}")

    print("\n📋 TASK 3: Misinformation (Hard)")
    g3 = TaskGrader("hard", shuffle=False)
    r3 = g3.run_with_decisions(hard_answers)
    print(f"  Final Score : {r3['final_score']}")
    print(f"  Result      : {r3['summary']}")

    print("\n" + "=" * 55)
    print("  All tasks graded successfully!")
    print("=" * 55)