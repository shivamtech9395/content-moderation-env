"""
FastAPI App — Content Moderation OpenEnv
Serves React frontend + API endpoints
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

from environment import ContentModerationEnv, Action
from tasks import TASKS

app = FastAPI(
    title="Content Moderation OpenEnv",
    description="Real-world content moderation environment for AI agents.",
    version="1.0.0",
)

# ── API ENDPOINTS ──

class ResetRequest(BaseModel):
    task_name: str = "easy"

class StepRequest(BaseModel):
    decision: str
    confidence: float = 1.0
    reason: Optional[str] = None

env: Optional[ContentModerationEnv] = None

@app.get("/api/health")
def health():
    return {"status": "healthy"}

@app.get("/api")
def root():
    return {
        "status": "ok",
        "environment": "Content Moderation OpenEnv",
        "version": "1.0.0",
        "tasks": list(TASKS.keys()),
        "message": "Use /reset to start, /step to act, /state to observe.",
    }

@app.post("/reset")
def reset(request: ResetRequest):
    global env
    if request.task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Invalid task '{request.task_name}'")
    env = ContentModerationEnv(task_name=request.task_name)
    obs = env.reset()
    return {
        "observation": obs.model_dump(),
        "task_info": TASKS[request.task_name],
        "message": f"Episode started for task: {request.task_name}",
    }

@app.post("/step")
def step(request: StepRequest):
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    if env._done:
        raise HTTPException(status_code=400, detail="Episode done. Call /reset.")
    valid = {"safe", "spam", "hate_speech", "misinformation", "harmful"}
    if request.decision not in valid:
        raise HTTPException(status_code=400, detail=f"Invalid decision '{request.decision}'")
    action = Action(decision=request.decision, confidence=request.confidence, reason=request.reason)
    obs, reward, done, info = env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}

@app.get("/state")
def state():
    global env
    if env is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    return env.state()

@app.get("/tasks")
def list_tasks():
    return {"tasks": TASKS, "total": len(TASKS)}

# ── SERVE REACT FRONTEND ──
# Only mount if frontend/dist exists (after build)
frontend_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_path):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_path, "assets")), name="assets")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        # API routes ko skip karo
        if full_path.startswith("api") or full_path in ["reset","step","state","tasks","health","docs","openapi.json"]:
            raise HTTPException(status_code=404)
        index = os.path.join(frontend_path, "index.html")
        return FileResponse(index)
else:
    @app.get("/")
    def root_fallback():
        return {
            "status": "ok",
            "environment": "Content Moderation OpenEnv",
            "version": "1.0.0",
            "tasks": list(TASKS.keys()),
            "message": "Frontend not built. Run: cd frontend && npm run build",
        }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)