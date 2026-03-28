"""
FastAPI App — Content Moderation OpenEnv
=========================================
This is the main server that runs on Hugging Face Spaces.
Judges will ping this server to validate the environment.

Endpoints:
  GET  /          → Health check
  POST /reset     → Start new episode
  POST /step      → Take one action
  GET  /state     → Get current state
  GET  /tasks     → List all tasks
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

from environment import ContentModerationEnv, Action, Observation
from tasks import TASKS

# ─────────────────────────────────────────────
#  FastAPI App
# ─────────────────────────────────────────────

app = FastAPI(
    title="Content Moderation OpenEnv",
    description=(
        "A real-world OpenEnv environment for training and evaluating "
        "AI agents on content moderation tasks. "
        "Tasks: spam detection, hate speech identification, "
        "and misinformation detection."
    ),
    version="1.0.0",
)

# Global environment instance
env: Optional[ContentModerationEnv] = None


# ─────────────────────────────────────────────
#  Request / Response Models
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_name: str = "easy"

class StepRequest(BaseModel):
    decision: str
    confidence: float = 1.0
    reason: Optional[str] = None


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "Content Moderation OpenEnv",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "message": "Use /reset to start, /step to act, /state to observe.",
    }


@app.post("/reset")
def reset(request: ResetRequest):
    """
    Start a new episode.
    
    Args:
        task_name: one of 'easy', 'medium', 'hard'
    
    Returns:
        First observation
    """
    global env

    if request.task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task '{request.task_name}'. Choose from: easy, medium, hard"
        )

    env = ContentModerationEnv(task_name=request.task_name)
    obs = env.reset()

    return {
        "observation": obs.model_dump(),
        "task_info": TASKS[request.task_name],
        "message": f"Episode started for task: {request.task_name}",
    }


@app.post("/step")
def step(request: StepRequest):
    """
    Take one action in the environment.
    
    Args:
        decision: classification decision
        confidence: how confident the agent is (0.0 to 1.0)
        reason: optional explanation
    
    Returns:
        next observation, reward, done flag, info
    """
    global env

    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    if env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode."
        )

    valid_decisions = {"safe", "spam", "hate_speech", "misinformation", "harmful"}
    if request.decision not in valid_decisions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision '{request.decision}'. Choose from: {valid_decisions}"
        )

    action = Action(
        decision=request.decision,
        confidence=request.confidence,
        reason=request.reason,
    )

    obs, reward, done, info = env.step(action)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state():
    """Get current environment state."""
    global env

    if env is None:
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    return env.state()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": TASKS,
        "total": len(TASKS),
    }


@app.get("/health")
def health():
    """Simple health check for HF Space monitoring."""
    return {"status": "healthy"}


# ─────────────────────────────────────────────
#  Run server
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)