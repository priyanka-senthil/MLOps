from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
import json
from datetime import datetime

router = APIRouter()

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


def get_metadata():
    """Load model metadata"""
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return None


def get_log_stats():
    """Get basic log statistics"""
    app_log = os.path.join(LOG_DIR, "app.log")
    train_log = os.path.join(LOG_DIR, "train.log")
    
    stats = {
        "app_log_lines": 0,
        "train_log_lines": 0,
        "last_prediction": None,
        "last_training": None
    }
    
    if os.path.exists(app_log):
        with open(app_log) as f:
            lines = f.readlines()
            stats["app_log_lines"] = len(lines)
            if lines:
                stats["last_prediction"] = lines[-1].split("|")[0].strip()
    
    if os.path.exists(train_log):
        with open(train_log) as f:
            lines = f.readlines()
            stats["train_log_lines"] = len(lines)
            if lines:
                stats["last_training"] = lines[-1].split("|")[0].strip()
    
    return stats


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Render the dashboard"""
    metadata = get_metadata()
    log_stats = get_log_stats()
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "metadata": metadata,
        "log_stats": log_stats,
        "current_time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    })