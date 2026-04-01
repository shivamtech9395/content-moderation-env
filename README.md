---
title: Content Moderation OpenEnv
emoji: 🛡️
colorFrom: teal
colorTo: blue
sdk: docker
python_version: "3.11"
app_file: app.py
pinned: false
---
# 🛡️ Content Moderation OpenEnv

> A real-world OpenEnv environment where AI agents learn to moderate social media content — detecting spam, hate speech, and misinformation with meaningful reward signals.

**Built for the Meta PyTorch Hackathon 2026** | [HF Space](https://huggingface.co/spaces/shivamtech9395/content-moderation-env) | [GitHub](https://github.com/shivamtech9395/content-moderation-env)

---

## 🎯 Why This Environment?

Content moderation is one of the most critical real-world AI problems. Companies like Meta, YouTube, and Twitter moderate billions of posts daily. This environment lets AI agents **train and evaluate** on exactly this problem — with:

- ✅ Meaningful partial reward signals (not just binary win/lose)
- ✅ 3 difficulty levels (Easy → Medium → Hard)
- ✅ Deterministic graders with clear success criteria
- ✅ Real-world post examples across 5 harm categories

---

## 📁 Project Structure

```
content-moderation-env/
├── environment.py      # Core OpenEnv environment (Observation, Action, Reward, step/reset/state)
├── tasks.py            # 3 tasks with deterministic graders
├── inference.py        # Baseline LLM agent script
├── app.py              # FastAPI server (REST API + serves frontend)
├── openenv.yaml        # OpenEnv metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
├── frontend/           # React dashboard (optional GUI)
└── README.md           # This file
```

---

## 🗂️ Tasks

| Task | Difficulty | Objective | Posts | Threshold |
|------|-----------|-----------|-------|-----------|
| `easy` | 🟢 Easy | Detect obvious spam vs safe content | 12 | 0.70 |
| `medium` | 🟡 Medium | Identify hate speech vs safe content | 12 | 0.65 |
| `hard` | 🔴 Hard | Detect subtle misinformation & harmful content | 12 | 0.55 |

---

## 👁️ Observation Space

```python
class Observation(BaseModel):
    post_id:     str           # Unique post identifier e.g. "easy_1"
    text:        str           # The social media post text to moderate
    task_name:   str           # Current task: easy | medium | hard
    step:        int           # Current step number (1-indexed)
    total_steps: int           # Total posts in this episode
    context:     Optional[str] # Optional context hint about the post
```

---

## ⚡ Action Space

```python
class Action(BaseModel):
    decision:   str   # One of: safe | spam | hate_speech | misinformation | harmful
    confidence: float # Agent confidence score 0.0 to 1.0
    reason:     str   # Optional explanation for the decision
```

---

## 🏆 Reward Function

| Situation | Score |
|-----------|-------|
| ✅ Exact correct classification | 0.9 + confidence bonus → max **1.0** |
| ⚠️ Harmful detected, wrong type | 0.4 + confidence bonus → max **0.5** |
| ❌ False positive (safe flagged as harmful) | **0.1** |
| ❌ Dangerous miss (harmful called safe) | **0.0** |

**Partial rewards** encourage the agent to at least detect *something* harmful even if the exact category is wrong. This provides a meaningful signal throughout the entire trajectory — not just at the end.

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/shivamtech9395/content-moderation-env
cd content-moderation-env
pip install -r requirements.txt
```

### 2. Set Environment Variables

Create a `.env` file:
```
HF_TOKEN=your_huggingface_token
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
```

### 3. Run Baseline Inference

```bash
# Run all 3 tasks
python inference.py

# Run specific task
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

### 4. Start API Server

```bash
python app.py
# Server running at http://localhost:7860
# API docs at http://localhost:7860/docs
```

### 5. Run with Docker

```bash
docker build -t content-moderation-env .
docker run -e HF_TOKEN=your_token -p 7860:7860 content-moderation-env
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Start new episode — body: `{"task_name": "easy"}` |
| POST | `/step` | Take one action — body: `{"decision": "spam", "confidence": 0.9}` |
| GET | `/state` | Get current environment state |
| GET | `/tasks` | List all available tasks |
| GET | `/health` | HF Space monitoring |
| GET | `/docs` | Interactive API documentation (Swagger) |

### Example API Usage

```bash
# Start episode
curl -X POST https://shivamtech9395-content-moderation-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "easy"}'

# Take action
curl -X POST https://shivamtech9395-content-moderation-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"decision": "spam", "confidence": 0.95, "reason": "Promotional language detected"}'

# Check state
curl https://shivamtech9395-content-moderation-env.hf.space/state
```

---

## 📊 Baseline Scores

Agent: `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router

| Task | Score | Accuracy | Result |
|------|-------|----------|--------|
| 🟢 Easy — Spam Detection | **1.00** | 100% | ✅ PASS |
| 🟡 Medium — Hate Speech | **0.99** | 100% | ✅ PASS |
| 🔴 Hard — Misinformation | **0.92** | 91.7% | ✅ PASS |
| **Overall Average** | **0.97** | 97.2% | ✅ |

---

## 🔄 OpenEnv Compliance

```
✅ reset()  → Returns typed Observation (Pydantic model)
✅ step()   → Returns (Observation, Reward, done: bool, info: dict)
✅ state()  → Returns current environment state dict
✅ Reward   → Always in range [0.0, 1.0]
✅ 3+ tasks → With difficulty progression Easy → Medium → Hard
✅ Graders  → Deterministic and reproducible
✅ Partial rewards → Meaningful signal across full trajectory
```

---

## 🐳 Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

## 🧠 Model & Architecture

- **LLM**: `meta-llama/Llama-3.1-8B-Instruct`
- **Provider**: HuggingFace Inference Router
- **Temperature**: 0.1 (for consistent outputs)
- **Output Format**: JSON `{decision, confidence, reason}`
- **Framework**: OpenAI-compatible client

---

## 👨‍💻 Author

Built with ❤️ for the **Meta PyTorch Hackathon 2026**

- HF Space: https://huggingface.co/spaces/shivamtech9395/content-moderation-env
- GitHub: https://github.com/shivamtech9395/content-moderation-env