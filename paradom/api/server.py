from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os

from paradom.core.loader import ModelLoader
from paradom.core.equivalence import EquivalenceIdentifier
from paradom.core.swap_engine import SwapEngine
from paradom.core.writer import BufferedMmapWriter, StreamingSwapper

app = FastAPI(title="Paradom API", version="0.1.0")

class SwapRequest(BaseModel):
    source_path: str
    target_config_path: str
    output_path: str
    paradigm: str = "llm"

class JobStatus(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = None

# In-memory job store for demo purposes
jobs: Dict[str, str] = {}

def execute_swap_task(request: SwapRequest, job_id: str):
    try:
        import yaml
        with open(request.target_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        config["paradigm"] = request.paradigm
        
        loader = ModelLoader(request.source_path)
        identifier = EquivalenceIdentifier()
        engine = SwapEngine()
        writer = BufferedMmapWriter(request.output_path)
        
        swapper = StreamingSwapper(loader, identifier, engine, writer)
        swapper.run(config)
        
        jobs[job_id] = "COMPLETED"
    except Exception as e:
        jobs[job_id] = f"FAILED: {str(e)}"

@app.post("/swap", response_model=JobStatus)
async def start_swap(request: SwapRequest, background_tasks: BackgroundTasks):
    """Triggers a new weight swap operation in the background."""
    if not os.path.exists(request.source_path):
        raise HTTPException(status_code=404, detail="Source model not found")
        
    job_id = f"job_{len(jobs) + 1}"
    jobs[job_id] = "RUNNING"
    
    background_tasks.add_task(execute_swap_task, request, job_id)
    
    return JobStatus(job_id=job_id, status="STARTED")

@app.get("/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """Checks the status of a swap job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(job_id=job_id, status=jobs[job_id])

@app.get("/")
async def root():
    return {"message": "Paradom API is online.", "research": "3 = 4 - 1"}
