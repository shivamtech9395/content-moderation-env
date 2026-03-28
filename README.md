# Content Moderation OpenEnv 🛡️

A real-world OpenEnv environment where AI agents learn to moderate social media content — detecting spam, hate speech, and misinformation.

Built for the **Meta PyTorch Hackathon 2026**.

---

## 🎯 Why This Environment?

Content moderation is one of the most critical real-world AI problems. Companies like Meta, YouTube, and Twitter spend millions moderating billions of posts daily. This environment lets AI agents train and evaluate on exactly this problem — with meaningful reward signals and increasing difficulty.

---

## 🗂️ Project Structure

```
ai-agent/
├── environment.py      # Core OpenEnv environment
├── tasks.py            # 3 tasks with graders
├── inference.py        # AI agent baseline script
├── app.py              # FastAPI server for HF Spaces
├── openenv.yaml        # Environment metadata
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # This file
```

---

## 📋 Tasks

| Task | Difficulty | Objective | Success Threshold |
|------|-----------|-----------|------------------|
| `easy` | 🟢 Easy | Detect obvious spam posts | 0.70 |
| `medium` | 🟡 Medium | Identify hate speech | 0.65 |
| `hard` | 🔴 Hard | Detect subtle misinformation | 0.55 |

---

## 👁️ Observation Space

Each observation contains:

| Field | Type | Description |
|-------|------|-------------|
| `post_id` | string | Unique post identifier |
| `text` | string | Social media post to moderate |
| `task_name` | string | Current task (easy/medium/hard) |
| `step` | int | Current step number |
| `total_steps` | int | Total steps in episode |
| `context` | string | Optional context about the post |

---

## ⚡ Action Space

Agent must output one of:

| Decision | Meaning |
|----------|---------|
| `safe` | Normal, harmless content |
| `spam` | Unwanted promotional or scam content |
| `hate_speech` | Content targeting people based on identity |
| `misinformation` | False or misleading factual claims |
| `harmful` | Dangerous advice or content |

---

## 🏆 Reward Function

| Situation | Score |
|-----------|-------|
| Exact correct classification | 0.9 + confidence bonus (up to 1.0) |
| Harmful detected but wrong type | 0.4 + confidence bonus |
| Missed harmful content (said safe) | 0.0 |
| False positive (safe flagged as harmful) | 0.1 |

Partial rewards encourage the agent to at least detect *something* harmful even if the exact category is wrong.

---

## 🚀 Setup & Usage

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set environment variables
Create a `.env` file:
```
HF_TOKEN=your_huggingface_token
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
API_BASE_URL=https://router.huggingface.co/v1
```

### 3. Run baseline inference
```bash
python inference.py
```

### 4. Start API server
```bash
python app.py
```

### 5. Test with Docker
```bash
docker build -t content-moderation-env .
docker run -p 7860:7860 content-moderation-env
```

---

## 📊 Baseline Scores

Scores achieved by `meta-llama/Llama-3.1-8B-Instruct`:

| Task | Score | Result |
|------|-------|--------|
| Easy — Spam Detection | 1.00 | ✅ PASS |
| Medium — Hate Speech | 0.99 | ✅ PASS |
| Hard — Misinformation | 0.92 | ✅ PASS |
| **Overall Average** | **0.97** | ✅ |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|---------|-------------|
| GET | `/` | Health check |
| POST | `/reset` | Start new episode |
| POST | `/step` | Take one action |
| GET | `/state` | Get current state |
| GET | `/tasks` | List all tasks |
| GET | `/health` | HF Space monitoring |

---

## 🐳 Docker

```bash
docker build -t content-moderation-env .
docker run -e HF_TOKEN=your_token -p 7860:7860 content-moderation-env
```

---

## 👨‍💻 Author

Built with ❤️ for the Meta PyTorch Hackathon 2026.