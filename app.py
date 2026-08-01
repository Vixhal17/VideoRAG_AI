import sys
# Configure standard streams to use UTF-8 to prevent charmap/UnicodeEncodeError on Windows
if sys.platform.startswith("win"):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

import os
import threading
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Pipeline imports
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decision, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

app = FastAPI()

# Global Pipeline state
pipeline_state = {
    "status": "idle",  # idle, running, done, error
    "error": None,
    "steps": {
        "audio": "pending",
        "transcript": "pending",
        "title": "pending",
        "summary": "pending",
        "extract": "pending",
        "rag": "pending",
    },
    "result": None,
    "rag_chain": None
}

state_lock = threading.Lock()


class AnalyzeRequest(BaseModel):
    source: str
    language: str


class ChatRequest(BaseModel):
    question: str


def update_step_status(step: str, status: str):
    with state_lock:
        pipeline_state["steps"][step] = status


def set_pipeline_status(status: str, error: str = None):
    with state_lock:
        pipeline_state["status"] = status
        if error:
            pipeline_state["error"] = error


def run_pipeline_thread(source: str):
    try:
        set_pipeline_status("running")

        # 1. Audio Processing
        update_step_status("audio", "active")
        chunks = process_input(source)
        update_step_status("audio", "done")

        # 2. Transcription
        update_step_status("transcript", "active")
        transcript = transcribe_all(chunks)
        update_step_status("transcript", "done")

        # 3. Title Generation
        update_step_status("title", "active")
        title = generate_title(transcript)
        update_step_status("title", "done")

        # 4. Summarization
        update_step_status("summary", "active")
        summary = summarize(transcript)
        update_step_status("summary", "done")

        # 5. Extract Insights
        update_step_status("extract", "active")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decision(transcript)
        questions = extract_questions(transcript)
        update_step_status("extract", "done")

        # 6. RAG Engine Indexing
        update_step_status("rag", "active")
        rag_chain = build_rag_chain(transcript)
        update_step_status("rag", "done")

        with state_lock:
            pipeline_state["result"] = {
                "title": title,
                "summary": summary,
                "transcript": transcript,
                "action_items": action_items,
                "decisions": decisions,
                "questions": questions
            }
            pipeline_state["rag_chain"] = rag_chain
            pipeline_state["status"] = "done"

    except Exception as e:
        import traceback
        traceback.print_exc()
        set_pipeline_status("error", str(e))
        # Reset active steps to pending on crash
        with state_lock:
            for k, v in pipeline_state["steps"].items():
                if v == "active":
                    pipeline_state["steps"][k] = "pending"


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    with state_lock:
        if pipeline_state["status"] == "running":
            raise HTTPException(status_code=400, detail="Pipeline is already running.")

        # Reset state for a new run
        pipeline_state["status"] = "idle"
        pipeline_state["error"] = None
        pipeline_state["result"] = None
        pipeline_state["rag_chain"] = None
        for k in pipeline_state["steps"]:
            pipeline_state["steps"][k] = "pending"

    background_tasks.add_task(run_pipeline_thread, req.source)
    return {"message": "Pipeline analysis started successfully."}


@app.get("/api/status")
async def get_status():
    with state_lock:
        return {
            "status": pipeline_state["status"],
            "error": pipeline_state["error"],
            "steps": pipeline_state["steps"]
        }


@app.get("/api/result")
async def get_result():
    with state_lock:
        if pipeline_state["status"] != "done":
            raise HTTPException(status_code=400, detail="Analysis is not completed yet.")
        return {"result": pipeline_state["result"]}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    # Retrieve chain safely
    chain = None
    with state_lock:
        chain = pipeline_state["rag_chain"]

    if not chain:
        raise HTTPException(status_code=400, detail="No active session found. Please run analysis first.")

    answer = ask_question(chain, req.question)
    return {"answer": answer}


# Serve static web files
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8505, reload=True)