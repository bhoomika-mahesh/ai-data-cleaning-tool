from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
import uvicorn

from environment import DataCleaningEnv, Action, Observation, Reward

app = FastAPI(
    title="DataCleaningEnv",
    description="An OpenEnv environment for training AI agents to clean real-world datasets.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global env instances per task
envs = {
    "easy": DataCleaningEnv(task="easy"),
    "medium": DataCleaningEnv(task="medium"),
    "hard": DataCleaningEnv(task="hard"),
}


class ResetRequest(BaseModel):
    task: Optional[str] = "easy"

class StepRequest(BaseModel):
    task: Optional[str] = "easy"
    operation: str
    column: Optional[str] = None
    value: Optional[Any] = None
    dtype: Optional[str] = None

class StepResponse(BaseModel):
    observation: dict
    reward: float
    reward_reason: str
    done: bool
    info: dict


@app.get("/")
def root():
    return {
        "name": "DataCleaningEnv",
        "description": "OpenEnv environment for AI-driven data cleaning",
        "tasks": ["easy", "medium", "hard"],
        "endpoints": ["/reset", "/step", "/state", "/openenv.yaml"]
    }


@app.post("/reset")
def reset(req: ResetRequest):
    task = req.task or "easy"
    if task not in envs:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from: {list(envs.keys())}")
    obs = envs[task].reset()
    return obs.dict()


@app.post("/step")
def step(req: StepRequest):
    task = req.task or "easy"
    if task not in envs:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from: {list(envs.keys())}")
    action = Action(
        operation=req.operation,
        column=req.column,
        value=req.value,
        dtype=req.dtype,
    )
    obs, reward, done, info = envs[task].step(action)
    return StepResponse(
        observation=obs.dict(),
        reward=reward.value,
        reward_reason=reward.reason,
        done=done,
        info=info,
    ).dict()


@app.get("/state")
def state(task: str = "easy"):
    if task not in envs:
        raise HTTPException(status_code=400, detail=f"Invalid task. Choose from: {list(envs.keys())}")
    return envs[task].state()


@app.get("/openenv.yaml")
def openenv_yaml():
    from fastapi.responses import PlainTextResponse
    yaml_content = """
name: DataCleaningEnv
version: 1.0.0
description: >
  A real-world OpenEnv environment where an AI agent must clean messy datasets
  by performing operations like filling nulls, removing duplicates, fixing dtypes,
  and removing outliers.
tasks:
  - id: easy
    description: Fill null values in a single column using mean
    difficulty: easy
    max_steps: 5
    reward_range: [0.0, 1.0]
  - id: medium
    description: Fix nulls, remove duplicates, and correct dtype
    difficulty: medium
    max_steps: 10
    reward_range: [0.0, 1.0]
  - id: hard
    description: Handle outliers, nulls, duplicates across multiple columns
    difficulty: hard
    max_steps: 15
    reward_range: [0.0, 1.0]
action_space:
  type: discrete_structured
  operations:
    - fill_null
    - drop_duplicates
    - fix_dtype
    - drop_column
    - remove_outliers
    - done
observation_space:
  type: structured
  fields:
    - dataset
    - columns
    - dtypes
    - null_counts
    - duplicate_count
    - step_number
    - max_steps
    - task_description
reward:
  type: continuous
  range: [-0.2, 1.0]
  description: Partial reward per operation + final grade on task completion
endpoints:
  reset: POST /reset
  step: POST /step
  state: GET /state
"""
    return PlainTextResponse(yaml_content.strip(), media_type="text/yaml")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
