"""
Content Moderation Environment
================================
A real-world OpenEnv environment that simulates content moderation tasks.
An AI agent must classify social media posts as safe or harmful.
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
#  Pydantic Models (Typed Data Structures)
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""
    post_id: str = Field(description="Unique ID of the post")
    text: str = Field(description="The social media post text to moderate")
    task_name: str = Field(description="Current task name")
    step: int = Field(description="Current step number")
    total_steps: int = Field(description="Total steps in this episode")
    context: Optional[str] = Field(default=None, description="Extra context if any")


class Action(BaseModel):
    """What the agent can do."""
    decision: str = Field(description="One of: safe | spam | hate_speech | misinformation | harmful")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score 0.0 to 1.0")
    reason: Optional[str] = Field(default=None, description="Why agent made this decision")


class Reward(BaseModel):
    """Feedback given to the agent after each action."""
    score: float = Field(ge=0.0, le=1.0, description="Reward score between 0.0 and 1.0")
    correct: bool = Field(description="Was the decision correct?")
    feedback: str = Field(description="Human-readable feedback")
    expected: str = Field(description="What the correct answer was")


# ─────────────────────────────────────────────
#  Dataset: Posts for each difficulty level
# ─────────────────────────────────────────────

EASY_POSTS = [
    {
        "text": "Buy cheap Viagra now!!! Click here for 90% discount on all meds!!!",
        "label": "spam",
        "context": None
    },
    {
        "text": "Win a FREE iPhone 15! Just send your bank details to claim your prize!",
        "label": "spam",
        "context": None
    },
    {
        "text": "I love spending time with my family on weekends. The weather was beautiful today!",
        "label": "safe",
        "context": None
    },
    {
        "text": "Just finished reading a great book on machine learning. Highly recommend it!",
        "label": "safe",
        "context": None
    },
    {
        "text": "URGENT: You have won $1,000,000! Send your SSN and credit card to collect!",
        "label": "spam",
        "context": None
    },
    {
        "text": "Had a wonderful dinner with friends last night. Tried a new Italian restaurant.",
        "label": "safe",
        "context": None
    },
]

MEDIUM_POSTS = [
    {
        "text": "People from that country should all go back where they came from. They are ruining everything.",
        "label": "hate_speech",
        "context": "Immigration discussion thread"
    },
    {
        "text": "The new park in our city is amazing! Kids are loving the new playground equipment.",
        "label": "safe",
        "context": "Local community forum"
    },
    {
        "text": "Those people are not real humans, they are animals and should be treated as such.",
        "label": "hate_speech",
        "context": "Political discussion"
    },
    {
        "text": "I disagree with the new policy but I think we should discuss it peacefully and find common ground.",
        "label": "safe",
        "context": "Political discussion"
    },
    {
        "text": "Women are too emotional to be leaders. They should stay home and raise children.",
        "label": "hate_speech",
        "context": "Workplace discussion"
    },
    {
        "text": "Our team worked really hard this quarter. Proud of everyone's dedication and effort!",
        "label": "safe",
        "context": "Workplace discussion"
    },
]

HARD_POSTS = [
    {
        "text": "New study from unnamed researchers confirms that 5G towers secretly cause cancer. Share before they delete this!",
        "label": "misinformation",
        "context": "Health discussion — looks like real news but is not"
    },
    {
        "text": "The moon landing was staged in a Hollywood studio. NASA has admitted this internally according to leaked documents.",
        "label": "misinformation",
        "context": "History discussion — uses fake authority to seem credible"
    },
    {
        "text": "Doctors don't want you to know that drinking bleach can cure COVID-19. Big pharma is hiding the truth!",
        "label": "harmful",
        "context": "Health misinformation — dangerous advice disguised as secret truth"
    },
    {
        "text": "Scientists at MIT published a peer-reviewed paper showing climate change is accelerating faster than expected.",
        "label": "safe",
        "context": "Real scientific consensus — do not flag this"
    },
    {
        "text": "Studies show vaccines contain microchips funded by billionaires to track and control the population.",
        "label": "misinformation",
        "context": "Medical misinformation — popular conspiracy theory"
    },
    {
        "text": "According to WHO, regular handwashing significantly reduces the spread of infectious diseases.",
        "label": "safe",
        "context": "Legitimate health advice from a real organization"
    },
]


# ─────────────────────────────────────────────
#  Helper: Score an action against ground truth
# ─────────────────────────────────────────────

def compute_reward(decision: str, expected_label: str, confidence: float) -> Reward:
    """
    Reward function with partial credit:
    - Exact match → 1.0
    - Harmful content marked as any harmful type → 0.5 partial credit
    - Safe marked as harmful OR harmful marked as safe → 0.0
    - Confidence bonus: adds up to 0.1 extra for correct high-confidence decisions
    """
    decision = decision.lower().strip()
    expected = expected_label.lower().strip()

    HARMFUL_LABELS = {"spam", "hate_speech", "misinformation", "harmful"}

    if decision == expected:
        # Perfect match — add confidence bonus
        base_score = 0.9
        confidence_bonus = 0.1 * confidence
        score = round(base_score + confidence_bonus, 2)
        return Reward(
            score=score,
            correct=True,
            feedback=f"✅ Correct! Post was '{expected}'. Great job!",
            expected=expected
        )

    elif decision in HARMFUL_LABELS and expected in HARMFUL_LABELS:
        # Detected harmful but wrong category — partial credit
        score = round(0.4 + 0.1 * confidence, 2)
        return Reward(
            score=score,
            correct=False,
            feedback=f"⚠️ Partially correct. You detected harmful content but wrong type. Expected '{expected}', got '{decision}'.",
            expected=expected
        )

    elif decision == "safe" and expected in HARMFUL_LABELS:
        # Missed harmful content — worst case
        return Reward(
            score=0.0,
            correct=False,
            feedback=f"❌ Missed harmful content! Post was '{expected}' but you said 'safe'. This is dangerous!",
            expected=expected
        )

    elif decision in HARMFUL_LABELS and expected == "safe":
        # False positive — penalized but less severely
        return Reward(
            score=0.1,
            correct=False,
            feedback=f"❌ False alarm! Post was safe but you flagged it as '{decision}'.",
            expected=expected
        )

    else:
        return Reward(
            score=0.0,
            correct=False,
            feedback=f"❌ Wrong. Expected '{expected}', got '{decision}'.",
            expected=expected
        )


# ─────────────────────────────────────────────
#  Main Environment Class
# ─────────────────────────────────────────────

class ContentModerationEnv:
    """
    OpenEnv-compatible Content Moderation Environment.

    The agent reads social media posts and decides:
    safe | spam | hate_speech | misinformation | harmful

    Tasks:
        easy   → Obvious spam detection
        medium → Hate speech identification
        hard   → Subtle misinformation detection
    """

    TASK_DATASETS = {
        "easy": EASY_POSTS,
        "medium": MEDIUM_POSTS,
        "hard": HARD_POSTS,
    }

    def __init__(self, task_name: str = "easy"):
        if task_name not in self.TASK_DATASETS:
            raise ValueError(f"Invalid task '{task_name}'. Choose from: easy, medium, hard")
        self.task_name = task_name
        self._posts: List[Dict] = []
        self._current_index: int = 0
        self._history: List[Dict] = []
        self._total_reward: float = 0.0
        self._done: bool = False

    # ── OpenEnv Interface ──────────────────────

    def reset(self) -> Observation:
        """Start a fresh episode. Shuffle posts for variety."""
        dataset = self.TASK_DATASETS[self.task_name].copy()
        random.shuffle(dataset)
        self._posts = dataset
        self._current_index = 0
        self._history = []
        self._total_reward = 0.0
        self._done = False
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Agent takes one action (classifies current post).
        Returns: (next_observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() first.")

        current_post = self._posts[self._current_index]
        reward = compute_reward(action.decision, current_post["label"], action.confidence)

        self._total_reward += reward.score
        self._history.append({
            "step": self._current_index + 1,
            "post": current_post["text"][:60] + "...",
            "decision": action.decision,
            "expected": current_post["label"],
            "score": reward.score,
            "correct": reward.correct,
        })

        self._current_index += 1

        if self._current_index >= len(self._posts):
            self._done = True
            obs = Observation(
                post_id="EPISODE_END",
                text="Episode complete.",
                task_name=self.task_name,
                step=self._current_index,
                total_steps=len(self._posts),
                context=None
            )
        else:
            obs = self._make_observation()

        info = {
            "total_reward": round(self._total_reward, 2),
            "average_score": round(self._total_reward / self._current_index, 2),
            "steps_done": self._current_index,
            "history": self._history,
        }

        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "task_name": self.task_name,
            "current_step": self._current_index,
            "total_steps": len(self._posts),
            "total_reward": round(self._total_reward, 2),
            "done": self._done,
            "history": self._history,
        }

    # ── Private Helpers ────────────────────────

    def _make_observation(self) -> Observation:
        post = self._posts[self._current_index]
        return Observation(
            post_id=f"{self.task_name}_{self._current_index + 1}",
            text=post["text"],
            task_name=self.task_name,
            step=self._current_index + 1,
            total_steps=len(self._posts),
            context=post.get("context"),
        )