"""
Content Moderation Environment
================================
A real-world OpenEnv environment that simulates content moderation tasks.
An AI agent must classify social media posts as safe or harmful.

Tasks:
  easy   → Spam Detection           (30 posts, threshold 0.70)
  medium → Hate Speech Detection    (30 posts, threshold 0.65)
  hard   → Misinformation & Harmful (30 posts, threshold 0.55)

OpenEnv Interface:
  reset()        → returns initial Observation
  step(action)   → returns (Observation, Reward, done, info)
  state()        → returns current environment state dict
"""

import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


# ── Pydantic Models ───────────────────────────────────────

class Observation(BaseModel):
    """What the agent sees at each step."""
    post_id:     str           = Field(description="Unique post identifier")
    text:        str           = Field(description="Social media post text to moderate")
    task_name:   str           = Field(description="Current task name")
    step:        int           = Field(description="Current step number (1-indexed)")
    total_steps: int           = Field(description="Total steps in this episode")
    context:     Optional[str] = Field(default=None, description="Optional context hint")


class Action(BaseModel):
    """What the agent can do — classify the post."""
    decision:   str            = Field(description="One of: safe | spam | hate_speech | misinformation | harmful")
    confidence: float          = Field(default=1.0, ge=0.0, le=1.0, description="Agent confidence 0.0–1.0")
    reason:     Optional[str]  = Field(default=None, description="Optional explanation")


class Reward(BaseModel):
    """Feedback given to the agent after each action."""
    score:    float = Field(ge=0.0, le=1.0, description="Reward score 0.0–1.0")
    correct:  bool  = Field(description="Was the decision correct?")
    feedback: str   = Field(description="Human-readable feedback message")
    expected: str   = Field(description="Ground truth label")


# ── EASY DATASET: Spam Detection (30 posts) ───────────────

EASY_POSTS = [
    # ── SPAM (15 posts) ──
    {"text": "Buy cheap Viagra now!!! Click here for 90% discount on all meds!!!", "label": "spam", "context": None},
    {"text": "Win a FREE iPhone 15! Just send your bank details to claim your prize!", "label": "spam", "context": None},
    {"text": "URGENT: You have won $1,000,000! Send your SSN and credit card to collect!", "label": "spam", "context": None},
    {"text": "Make $5000 per day working from home! No experience needed! Click now!", "label": "spam", "context": None},
    {"text": "Congratulations! You are our lucky winner today. Claim your reward NOW!!!", "label": "spam", "context": None},
    {"text": "LIMITED OFFER: Get 80% off all luxury watches. Only 10 left. Buy now!", "label": "spam", "context": None},
    {"text": "Your account has been compromised! Click here immediately to verify your identity.", "label": "spam", "context": None},
    {"text": "FREE Bitcoin! Send 0.1 BTC to this address and receive 1 BTC back guaranteed!", "label": "spam", "context": None},
    {"text": "You have been selected for our exclusive survey. Complete it and earn $500!", "label": "spam", "context": None},
    {"text": "FLASH SALE: Designer bags 95% off! Only available for the next 2 hours!!!", "label": "spam", "context": None},
    {"text": "Lose 30 pounds in 30 days with this one weird trick doctors hate! Click here!", "label": "spam", "context": None},
    {"text": "Your Netflix account will be suspended! Update payment info now to avoid interruption.", "label": "spam", "context": None},
    {"text": "Earn passive income! Join our MLM network and make thousands monthly doing nothing!", "label": "spam", "context": None},
    {"text": "WINNER WINNER! Your phone number was randomly selected. Claim $10,000 prize now!", "label": "spam", "context": None},
    {"text": "Hot singles in your area want to meet you! Click to see their photos now!", "label": "spam", "context": None},
    # ── SAFE (15 posts) ──
    {"text": "I love spending time with my family on weekends. The weather was beautiful today!", "label": "safe", "context": None},
    {"text": "Just finished reading a great book on machine learning. Highly recommend it!", "label": "safe", "context": None},
    {"text": "Had a wonderful dinner with friends last night. Tried a new Italian restaurant.", "label": "safe", "context": None},
    {"text": "The sunset at the beach today was absolutely breathtaking. Nature is amazing!", "label": "safe", "context": None},
    {"text": "Finally completed my morning run! 5 km in under 30 minutes. Feeling great!", "label": "safe", "context": None},
    {"text": "Attended a great webinar on Python best practices today. Learned so much!", "label": "safe", "context": None},
    {"text": "Cooking a new recipe tonight — homemade pasta with fresh tomato sauce!", "label": "safe", "context": None},
    {"text": "My garden is blooming beautifully this spring. The roses are stunning!", "label": "safe", "context": None},
    {"text": "Just adopted a puppy from the shelter. She is the most adorable thing ever!", "label": "safe", "context": None},
    {"text": "Finished my thesis after 2 years of hard work. Officially submitting tomorrow!", "label": "safe", "context": None},
    {"text": "The local farmers market had the best fresh vegetables today. Highly recommend!", "label": "safe", "context": None},
    {"text": "My team won the soccer tournament today! So proud of everyone's effort!", "label": "safe", "context": None},
    {"text": "Started learning guitar last month. Making slow progress but loving every moment!", "label": "safe", "context": None},
    {"text": "Watched an incredible documentary about ocean life last night. So educational!", "label": "safe", "context": None},
    {"text": "Coffee with an old friend this morning. Nothing better than catching up!", "label": "safe", "context": None},
]


# ── MEDIUM DATASET: Hate Speech Detection (30 posts) ─────

MEDIUM_POSTS = [
    # ── HATE SPEECH (15 posts) ──
    {"text": "People from that country should all go back where they came from. They are ruining everything.", "label": "hate_speech", "context": "Immigration discussion"},
    {"text": "Those people are not real humans, they are animals and should be treated as such.", "label": "hate_speech", "context": "Political discussion"},
    {"text": "Women are too emotional to be leaders. They should stay home and raise children.", "label": "hate_speech", "context": "Workplace discussion"},
    {"text": "I cannot believe they are letting those people into our neighborhoods. Disgusting!", "label": "hate_speech", "context": "Local community forum"},
    {"text": "People of that religion are all terrorists. They should be banned from this country.", "label": "hate_speech", "context": "News article comments"},
    {"text": "Disabled people are just a burden on society. They contribute nothing useful.", "label": "hate_speech", "context": "Healthcare discussion"},
    {"text": "Those immigrants are stealing our jobs and destroying our culture. Kick them out!", "label": "hate_speech", "context": "Economic discussion"},
    {"text": "People with mental illness are dangerous and should be locked away from normal people.", "label": "hate_speech", "context": "Mental health discussion"},
    {"text": "That ethnic group has always been criminals. It is just in their nature.", "label": "hate_speech", "context": "Crime statistics discussion"},
    {"text": "Gay people should not be allowed to adopt children. It is unnatural and wrong.", "label": "hate_speech", "context": "Family policy discussion"},
    {"text": "Old people should just retire and stop taking up space in the workforce.", "label": "hate_speech", "context": "Employment discussion"},
    {"text": "Poor people are poor because they are lazy. They do not deserve any help.", "label": "hate_speech", "context": "Economic policy discussion"},
    {"text": "Those people from that region smell bad and have no manners. They are uncivilized.", "label": "hate_speech", "context": "Travel discussion"},
    {"text": "Transgender people are mentally ill. They should be banned from public bathrooms.", "label": "hate_speech", "context": "Policy discussion"},
    {"text": "Fat people have no self control. They are disgusting and cost us all in healthcare.", "label": "hate_speech", "context": "Health discussion"},
    # ── SAFE (15 posts) ──
    {"text": "The new park in our city is amazing! Kids are loving the new playground equipment.", "label": "safe", "context": "Local community forum"},
    {"text": "I disagree with the new immigration policy but I think we should discuss it peacefully.", "label": "safe", "context": "Political discussion"},
    {"text": "Our team worked really hard this quarter. Proud of everyone's dedication!", "label": "safe", "context": "Workplace discussion"},
    {"text": "The cultural festival last weekend was so enriching. Loved learning about different traditions.", "label": "safe", "context": "Community event"},
    {"text": "We need better immigration policies that are fair, humane and effective for everyone.", "label": "safe", "context": "Immigration discussion"},
    {"text": "Accessibility improvements in public spaces benefit everyone in the community.", "label": "safe", "context": "Healthcare discussion"},
    {"text": "Mental health awareness is so important. Everyone deserves access to good support.", "label": "safe", "context": "Mental health discussion"},
    {"text": "Diversity in the workplace leads to better innovation and problem solving outcomes.", "label": "safe", "context": "Workplace discussion"},
    {"text": "It is important that we create inclusive policies that protect all citizens equally.", "label": "safe", "context": "Policy discussion"},
    {"text": "I learned so much from my coworkers from different cultural backgrounds this year.", "label": "safe", "context": "Workplace discussion"},
    {"text": "The senior citizens in our community have so much wisdom to share with young people.", "label": "safe", "context": "Community discussion"},
    {"text": "Supporting people in poverty through education and opportunity is good for all of us.", "label": "safe", "context": "Economic policy discussion"},
    {"text": "Every person deserves to be treated with dignity regardless of their background.", "label": "safe", "context": "Social discussion"},
    {"text": "The LGBTQ community festival in our city was a beautiful celebration of diversity!", "label": "safe", "context": "Community event"},
    {"text": "Body positivity is about health and self acceptance, not promoting unhealthy habits.", "label": "safe", "context": "Health discussion"},
]


# ── HARD DATASET: Misinformation & Harmful (30 posts) ────

HARD_POSTS = [
    # ── MISINFORMATION (10 posts) ──
    {"text": "New study from unnamed researchers confirms that 5G towers secretly cause cancer. Share before they delete!", "label": "misinformation", "context": "Health discussion — fabricated study"},
    {"text": "The moon landing was staged in a Hollywood studio. NASA has admitted this in leaked documents.", "label": "misinformation", "context": "History discussion — fake authority claim"},
    {"text": "Studies show vaccines contain microchips funded by billionaires to track and control the population.", "label": "misinformation", "context": "Medical misinformation — conspiracy theory"},
    {"text": "Scientists have proven that the earth is actually flat. Governments are hiding the truth from us all.", "label": "misinformation", "context": "Science discussion — well-known conspiracy"},
    {"text": "Eating raw garlic every morning cures cancer completely. Doctors hide this to sell chemotherapy.", "label": "misinformation", "context": "Health misinformation — fake cure claim"},
    {"text": "COVID-19 was created in a lab by governments to reduce the world population. The evidence is undeniable.", "label": "misinformation", "context": "Medical conspiracy theory"},
    {"text": "The election was stolen through millions of fraudulent ballots. The official results are completely fake.", "label": "misinformation", "context": "Political misinformation — unverified claim"},
    {"text": "Chemtrails from airplanes contain mind-control chemicals sprayed by the government on citizens.", "label": "misinformation", "context": "Conspiracy theory — chemtrail myth"},
    {"text": "Climate change is a hoax invented by scientists to get more research funding from governments.", "label": "misinformation", "context": "Science denial — climate misinformation"},
    {"text": "Autism is caused by vaccines. The CDC is hiding data that proves this connection clearly.", "label": "misinformation", "context": "Medical misinformation — debunked vaccine myth"},
    # ── HARMFUL (5 posts) ──
    {"text": "Doctors don't want you to know that drinking bleach can cure COVID-19. Big pharma is hiding truth!", "label": "harmful", "context": "Dangerous health advice disguised as secret"},
    {"text": "Drinking a mixture of bleach and water in small amounts can boost your immune system naturally.", "label": "harmful", "context": "Extremely dangerous instruction disguised as tip"},
    {"text": "If you want to lose weight fast, stop eating completely for 2 weeks. It is perfectly safe to do.", "label": "harmful", "context": "Dangerous diet advice that could cause harm"},
    {"text": "You can cure diabetes by throwing away your insulin. Just drink this herbal tea instead daily.", "label": "harmful", "context": "Dangerous medical advice — could cause death"},
    {"text": "To treat a deep wound at home, pour hydrogen peroxide and super glue directly into the wound.", "label": "harmful", "context": "Dangerous first aid misinformation"},
    # ── SAFE (15 posts) ──
    {"text": "Scientists at MIT published a peer-reviewed paper showing climate change is accelerating faster.", "label": "safe", "context": "Real scientific consensus — do not flag"},
    {"text": "According to WHO, regular handwashing significantly reduces the spread of infectious diseases.", "label": "safe", "context": "Legitimate health advice from real organization"},
    {"text": "The COVID-19 vaccine was approved by FDA after extensive clinical trials with 40,000 participants.", "label": "safe", "context": "Factual medical information from verified sources"},
    {"text": "Researchers at Stanford found that moderate exercise 3-4 times per week improves mental health.", "label": "safe", "context": "Evidence-based health information"},
    {"text": "NASA confirmed the discovery of water ice on the moon south pole using the LCROSS mission data.", "label": "safe", "context": "Verified scientific discovery — do not flag"},
    {"text": "The UN report on biodiversity shows over one million species face extinction due to human activity.", "label": "safe", "context": "Verified environmental report — do not flag"},
    {"text": "According to CDC guidelines, adults should get 7-9 hours of sleep per night for optimal health.", "label": "safe", "context": "Legitimate health recommendation"},
    {"text": "Eating a balanced diet rich in vegetables and fruits reduces risk of chronic disease significantly.", "label": "safe", "context": "Evidence-based nutritional advice"},
    {"text": "The International Astronomical Union confirmed Pluto was reclassified as a dwarf planet in 2006.", "label": "safe", "context": "Verified scientific fact"},
    {"text": "Oxford University researchers published findings that regular meditation reduces stress and anxiety.", "label": "safe", "context": "Peer-reviewed research finding"},
    {"text": "The historic Paris Agreement was signed by 196 countries to limit global temperature rise.", "label": "safe", "context": "Verified international agreement fact"},
    {"text": "According to UNICEF, access to clean water has improved significantly for millions of people globally.", "label": "safe", "context": "Verified humanitarian statistics"},
    {"text": "Solar panel efficiency has improved dramatically over the past decade according to energy reports.", "label": "safe", "context": "Factual technology information"},
    {"text": "The human genome project successfully mapped all human DNA and was completed in the year 2003.", "label": "safe", "context": "Verified scientific achievement"},
    {"text": "Regular dental checkups every 6 months are recommended by the American Dental Association.", "label": "safe", "context": "Legitimate professional health recommendation"},
]


# ── Reward Function ───────────────────────────────────────

def compute_reward(decision: str, expected_label: str, confidence: float) -> Reward:
    """
    Reward function with partial credit signals:

    Scoring:
      - Exact correct match          → 0.9 + confidence bonus (max 1.0)
      - Harmful detected, wrong type → 0.4 + confidence bonus (max 0.5)
      - False positive (safe→harmful)→ 0.1
      - Dangerous miss (harmful→safe)→ 0.0  ← worst case
      - Other mismatch               → 0.0
    """
    decision = decision.lower().strip()
    expected = expected_label.lower().strip()
    HARMFUL  = {"spam", "hate_speech", "misinformation", "harmful"}

    if decision == expected:
        score = round(min(1.0, 0.9 + 0.1 * confidence), 3)
        return Reward(
            score=score, correct=True,
            feedback=f"✅ Correct! Post correctly classified as '{expected}'. Score: {score:.2f}",
            expected=expected,
        )

    if decision in HARMFUL and expected in HARMFUL:
        score = round(min(0.5, 0.4 + 0.1 * confidence), 3)
        return Reward(
            score=score, correct=False,
            feedback=f"⚠️ Partial credit — detected harmful but wrong type. Expected '{expected}', got '{decision}'.",
            expected=expected,
        )

    if decision == "safe" and expected in HARMFUL:
        return Reward(
            score=0.0, correct=False,
            feedback=f"❌ Dangerous miss! Post was '{expected}' but agent said 'safe'. This is the worst error.",
            expected=expected,
        )

    if decision in HARMFUL and expected == "safe":
        return Reward(
            score=0.1, correct=False,
            feedback=f"❌ False positive! Post was safe but agent flagged as '{decision}'.",
            expected=expected,
        )

    return Reward(
        score=0.0, correct=False,
        feedback=f"❌ Wrong. Expected '{expected}', got '{decision}'.",
        expected=expected,
    )


# ── Environment ───────────────────────────────────────────

class ContentModerationEnv:
    """
    OpenEnv-compatible Content Moderation Environment.

    The agent reads social media posts and decides:
        safe | spam | hate_speech | misinformation | harmful

    Tasks:
        easy   → Spam Detection (30 posts)
        medium → Hate Speech Detection (30 posts)
        hard   → Misinformation & Harmful (30 posts)
    """

    TASK_DATASETS: Dict[str, List[Dict]] = {
        "easy":   EASY_POSTS,
        "medium": MEDIUM_POSTS,
        "hard":   HARD_POSTS,
    }

    def __init__(self, task_name: str = "easy"):
        if task_name not in self.TASK_DATASETS:
            raise ValueError(f"Unknown task '{task_name}'. Choose: easy, medium, hard")
        self.task_name       = task_name
        self._posts:         List[Dict] = []
        self._current_index: int   = 0
        self._history:       List[Dict] = []
        self._total_reward:  float = 0.0
        self._done:          bool  = False

    def reset(self) -> Observation:
        """Start fresh episode. Shuffles posts for variety."""
        dataset = self.TASK_DATASETS[self.task_name].copy()
        random.shuffle(dataset)
        self._posts         = dataset
        self._current_index = 0
        self._history       = []
        self._total_reward  = 0.0
        self._done          = False
        return self._make_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Agent takes one action — classifies current post.
        Returns: (next_observation, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        post   = self._posts[self._current_index]
        reward = compute_reward(action.decision, post["label"], action.confidence)

        self._total_reward += reward.score
        self._history.append({
            "step":     self._current_index + 1,
            "post":     post["text"][:70] + "...",
            "decision": action.decision,
            "expected": post["label"],
            "score":    reward.score,
            "correct":  reward.correct,
        })

        self._current_index += 1

        if self._current_index >= len(self._posts):
            self._done = True
            obs = Observation(
                post_id="EPISODE_END", text="Episode complete.",
                task_name=self.task_name, step=self._current_index,
                total_steps=len(self._posts), context=None,
            )
        else:
            obs = self._make_observation()

        info = {
            "total_reward":  round(self._total_reward, 4),
            "average_score": round(self._total_reward / self._current_index, 4),
            "steps_done":    self._current_index,
            "steps_total":   len(self._posts),
            "history":       self._history,
        }
        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return current environment state."""
        return {
            "task_name":     self.task_name,
            "current_step":  self._current_index,
            "total_steps":   len(self._posts),
            "total_reward":  round(self._total_reward, 4),
            "average_score": round(self._total_reward / max(1, self._current_index), 4),
            "done":          self._done,
            "history":       self._history,
        }

    def _make_observation(self) -> Observation:
        post = self._posts[self._current_index]
        return Observation(
            post_id     = f"{self.task_name}_{self._current_index + 1}",
            text        = post["text"],
            task_name   = self.task_name,
            step        = self._current_index + 1,
            total_steps = len(self._posts),
            context     = post.get("context"),
        )