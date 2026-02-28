#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/07/17
@Author  : Your Name / AI Agent
@File    : frontend_api_svc.py
@Desc    : FastAPI service for MetaGPT frontend interactions.
           Improvements:
             - List and delete project endpoints
             - Cost/token-usage tracking per project
             - WebSocket endpoint for real-time agent-message streaming
             - Structured JSON-lines message logging during runs
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    Path as FastApiPath,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field
from starlette.responses import FileResponse

# --- Attempt to import MetaGPT components ---
try:
    from metagpt.software_company import SoftwareCompany
    from metagpt.context import Context as MetaGPTContext
    from metagpt.config2 import config as metagpt_config
except ImportError:
    SoftwareCompany = None       # type: ignore
    MetaGPTContext = None        # type: ignore
    metagpt_config = None        # type: ignore

# --- Logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if SoftwareCompany is None:
    logger.warning(
        "MetaGPT core components not found. API will run in MOCK MODE for project generation."
    )

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MetaGPT Frontend API Service",
    description=(
        "REST + WebSocket API for interacting with MetaGPT projects and their artifacts. "
        "Supports project creation, status polling, artifact download, live log streaming, "
        "and cost/token-usage tracking."
    ),
    version="0.2.0",
)

# ---------------------------------------------------------------------------
# In-memory database  (replace with a real DB for production)
# ---------------------------------------------------------------------------

projects_db: Dict[str, "Project"] = {}

# Active WebSocket connections per project:  project_id -> list[WebSocket]
ws_connections: Dict[str, List[WebSocket]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    """Accumulated LLM token usage for a project."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0  # rough estimate; update pricing as needed


class Project(BaseModel):
    """Internal representation of a MetaGPT project."""
    project_id: str
    idea: str
    project_name: Optional[str] = None
    status: str = Field(
        default="pending",
        description="One of: pending | running | completed | failed",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    artifacts_path: Optional[str] = None
    message_log_path: Optional[str] = None
    run_log_path: Optional[str] = None
    captured_logs: List[str] = Field(default_factory=list)
    error_message: Optional[str] = None
    token_usage: TokenUsage = Field(default_factory=TokenUsage)


class ProjectCreateRequest(BaseModel):
    idea: str = Field(..., description="The initial idea or requirement for the project.")
    project_name: Optional[str] = Field(None, description="Optional name for the project.")


class ProjectCreateResponse(BaseModel):
    project_id: str
    project_name: Optional[str] = None
    message: str = "Project creation initiated."
    status: str = "pending"


class ProjectSummary(BaseModel):
    """Lightweight project summary for listing."""
    project_id: str
    project_name: Optional[str] = None
    idea: str
    status: str
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    projects: List[ProjectSummary]
    total: int


class ProjectStatusResponse(BaseModel):
    project_id: str
    project_name: Optional[str] = None
    idea: str
    status: str
    created_at: datetime
    updated_at: datetime
    artifacts_path: Optional[str] = None
    error_message: Optional[str] = None


class ProjectMetricsResponse(BaseModel):
    project_id: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float


class MessageModel(BaseModel):
    message_id: str
    role: str
    content: str
    timestamp: str


class ProjectMessagesResponse(BaseModel):
    project_id: str
    messages: List[MessageModel]


class ArtifactModel(BaseModel):
    artifact_name: str
    artifact_type: str
    created_at: str
    download_url: str


class ProjectArtifactsResponse(BaseModel):
    project_id: str
    artifacts: List[ArtifactModel]


class ProjectLogsResponse(BaseModel):
    project_id: str
    logs: List[str]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Rough cost estimates (USD per 1 000 tokens) — update to reflect real pricing.
_PROMPT_COST_PER_1K = 0.01
_COMPLETION_COST_PER_1K = 0.03


def _add_token_usage(project: Project, prompt: int = 0, completion: int = 0) -> None:
    """Accumulate token usage and update the estimated cost."""
    project.token_usage.prompt_tokens += prompt
    project.token_usage.completion_tokens += completion
    project.token_usage.total_tokens += prompt + completion
    project.token_usage.estimated_cost_usd += (
        prompt / 1000 * _PROMPT_COST_PER_1K + completion / 1000 * _COMPLETION_COST_PER_1K
    )


async def _broadcast(project_id: str, payload: dict) -> None:
    """Push a JSON payload to all WebSocket clients watching *project_id*."""
    conns = ws_connections.get(project_id, [])
    dead: List[WebSocket] = []
    for ws in conns:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        conns.remove(ws)


def _log_and_broadcast(project: Project, message: str) -> None:
    """Append *message* to project logs and schedule a WebSocket broadcast."""
    timestamped = f"[{datetime.utcnow().isoformat()}] {message}"
    project.captured_logs.append(timestamped)
    asyncio.ensure_future(
        _broadcast(project.project_id, {"type": "log", "data": timestamped})
    )


def _write_message(project: Project, role: str, content: str) -> None:
    """Append a structured JSON-line message to the project message log."""
    if not project.message_log_path:
        return
    record = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "sent_at": datetime.utcnow().isoformat(),
    }
    try:
        with open(project.message_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:
        logger.warning(f"Could not write message log: {exc}")

    asyncio.ensure_future(
        _broadcast(project.project_id, {"type": "message", "data": record})
    )


# ---------------------------------------------------------------------------
# Background task: run a MetaGPT project
# ---------------------------------------------------------------------------

async def run_metagpt_project(project_id: str, idea: str, project_name: str) -> None:
    project = projects_db.get(project_id)
    if not project:
        logger.error(f"Project {project_id} not found for background task.")
        return

    project.status = "running"
    project.updated_at = datetime.utcnow()
    _log_and_broadcast(project, "Project status updated to running.")
    await _broadcast(project_id, {"type": "status", "data": "running"})

    try:
        if SoftwareCompany is None or MetaGPTContext is None or metagpt_config is None:
            # ---------------------------------------------------------------
            # MOCK MODE
            # ---------------------------------------------------------------
            logger.warning(f"Running project {project_id} in mock mode.")
            mock_dir = Path("workspace") / project_name
            mock_dir.mkdir(parents=True, exist_ok=True)

            project.artifacts_path = str(mock_dir.resolve())
            project.message_log_path = str(mock_dir / "messages.log")
            project.run_log_path = str(mock_dir / "run.log")

            # Simulate agent activity
            agents = [
                ("ProductManager", "I will now analyse the requirements."),
                ("Architect", "Designing the system architecture."),
                ("Engineer", "Writing implementation code."),
                ("QaEngineer", "Running tests against the implementation."),
            ]
            for role, msg in agents:
                await asyncio.sleep(1)
                _write_message(project, role, msg)
                _log_and_broadcast(project, f"{role}: {msg}")
                # Simulate token usage per agent step
                _add_token_usage(project, prompt=250, completion=150)
                project.updated_at = datetime.utcnow()

            # Write mock files
            (mock_dir / "README.md").write_text(f"# Mock Project: {project_name}\nIdea: {idea}")
            (mock_dir / "main.py").write_text("print('Hello from mock project!')")

            with open(project.run_log_path, "w", encoding="utf-8") as f:
                f.write(f"[{datetime.utcnow().isoformat()}] [INFO] Mock project run started.\n")
                f.write(f"[{datetime.utcnow().isoformat()}] [INFO] Mock project run finished.\n")

            _log_and_broadcast(project, "Mock project run completed.")
            project.status = "completed"
            logger.info(f"Mock project {project_id} completed.")

        else:
            # ---------------------------------------------------------------
            # REAL MetaGPT run
            # ---------------------------------------------------------------
            base_workspace = Path(
                getattr(metagpt_config, "workspace_path", None) or "workspace"
            )
            sc = SoftwareCompany()
            await sc.run(idea=idea)

            # Resolve artifact path
            final_path: Optional[Path] = None
            try:
                final_path = Path(sc.context.repo.workdir)
            except AttributeError:
                pass
            if not final_path:
                final_path = base_workspace / project_name
                logger.warning(f"Guessing artifact path: {final_path}")

            if project.project_name != final_path.name:
                project.project_name = final_path.name

            project.artifacts_path = str(final_path.resolve())
            project.message_log_path = str(final_path / "messages.log")
            project.run_log_path = str(final_path / "logs" / "run.log")

            # Pull last 20 lines from run log
            if Path(project.run_log_path).exists():
                with open(project.run_log_path, encoding="utf-8") as f:
                    tail = f.read().splitlines()[-20:]
                project.captured_logs.extend(tail)

            _log_and_broadcast(project, f"Project generation completed. Artifacts at: {project.artifacts_path}")
            project.status = "completed"
            logger.info(f"MetaGPT project {project_id} completed.")

    except Exception as exc:
        logger.error(f"Error running project {project_id}: {exc}", exc_info=True)
        project.status = "failed"
        project.error_message = str(exc)
        _log_and_broadcast(project, f"Error: {exc}")
        await _broadcast(project_id, {"type": "status", "data": "failed"})
    finally:
        project.updated_at = datetime.utcnow()
        await _broadcast(project_id, {"type": "status", "data": project.status})
        logger.info(f"Project {project_id} final status: {project.status}")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/projects/",
    response_model=ProjectCreateResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_project(request: ProjectCreateRequest, background_tasks: BackgroundTasks):
    """Create a new project and start generation in the background."""
    project_id = str(uuid.uuid4())
    project_name = request.project_name or f"project_{project_id[:8]}"

    project = Project(
        project_id=project_id,
        idea=request.idea,
        project_name=project_name,
        status="pending",
    )
    projects_db[project_id] = project
    background_tasks.add_task(run_metagpt_project, project_id, request.idea, project_name)
    logger.info(f"Created project {project_id}: '{project_name}'")
    return ProjectCreateResponse(
        project_id=project_id,
        project_name=project_name,
        status="pending",
    )


@app.get("/projects/", response_model=ProjectListResponse)
async def list_projects():
    """Return a summary list of all known projects, newest first."""
    summaries = [
        ProjectSummary(
            project_id=p.project_id,
            project_name=p.project_name,
            idea=p.idea,
            status=p.status,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in sorted(projects_db.values(), key=lambda x: x.created_at, reverse=True)
    ]
    return ProjectListResponse(projects=summaries, total=len(summaries))


@app.get("/projects/{project_id}/", response_model=ProjectStatusResponse)
async def get_project_status(
    project_id: str = FastApiPath(..., description="Project ID"),
):
    """Get the current status of a project."""
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    return ProjectStatusResponse(
        project_id=project.project_id,
        project_name=project.project_name,
        idea=project.idea,
        status=project.status,
        created_at=project.created_at,
        updated_at=project.updated_at,
        artifacts_path=project.artifacts_path,
        error_message=project.error_message,
    )


@app.delete("/projects/{project_id}/", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str = FastApiPath(..., description="Project ID"),
):
    """Remove a project from the in-memory store.

    Note: this does NOT delete generated files from disk.
    """
    if project_id not in projects_db:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    if projects_db[project_id].status == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Cannot delete a project that is currently running.",
        )
    del projects_db[project_id]
    logger.info(f"Deleted project {project_id}.")


@app.get("/projects/{project_id}/metrics/", response_model=ProjectMetricsResponse)
async def get_project_metrics(
    project_id: str = FastApiPath(..., description="Project ID"),
):
    """Return accumulated token usage and estimated cost for a project."""
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    tu = project.token_usage
    return ProjectMetricsResponse(
        project_id=project_id,
        prompt_tokens=tu.prompt_tokens,
        completion_tokens=tu.completion_tokens,
        total_tokens=tu.total_tokens,
        estimated_cost_usd=tu.estimated_cost_usd,
    )


@app.get("/projects/{project_id}/messages/", response_model=ProjectMessagesResponse)
async def get_project_messages(
    project_id: str = FastApiPath(..., description="Project ID"),
):
    """Return all agent messages logged for a project."""
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")

    messages: List[MessageModel] = []
    if not project.message_log_path or not Path(project.message_log_path).exists():
        return ProjectMessagesResponse(project_id=project_id, messages=[])

    try:
        with open(project.message_log_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    messages.append(
                        MessageModel(
                            message_id=d.get("id", f"msg_{i}"),
                            role=d.get("role", "Unknown"),
                            content=d.get("content", ""),
                            timestamp=d.get("sent_at", datetime.utcnow().isoformat()),
                        )
                    )
                except json.JSONDecodeError:
                    parts = line.split(":", 1)
                    role = parts[0].strip() if len(parts) > 1 else "LogEntry"
                    content = parts[1].strip() if len(parts) > 1 else line
                    messages.append(
                        MessageModel(
                            message_id=f"line_{i}",
                            role=role,
                            content=content,
                            timestamp=datetime.utcnow().isoformat(),
                        )
                    )
    except Exception as exc:
        logger.error(f"Error reading messages for {project_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read project messages.",
        )
    return ProjectMessagesResponse(project_id=project_id, messages=messages)


@app.get("/projects/{project_id}/artifacts/", response_model=ProjectArtifactsResponse)
async def list_project_artifacts(
    project_id: str = FastApiPath(..., description="Project ID"),
):
    """List all generated files for a project."""
    project = projects_db.get(project_id)
    if not project or not project.artifacts_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project or artifacts path not found.",
        )

    artifact_root = Path(project.artifacts_path)
    if not artifact_root.is_dir():
        return ProjectArtifactsResponse(project_id=project_id, artifacts=[])

    artifacts: List[ArtifactModel] = []
    try:
        for item_path in sorted(artifact_root.rglob("*")):
            if item_path.is_file() and not item_path.name.startswith("."):
                rel = item_path.relative_to(artifact_root)
                download_url = app.url_path_for(
                    "download_project_artifact",
                    project_id=project_id,
                    artifact_name=str(rel),
                )
                artifacts.append(
                    ArtifactModel(
                        artifact_name=str(rel),
                        artifact_type=item_path.suffix.lstrip(".").lower() or "file",
                        created_at=datetime.fromtimestamp(item_path.stat().st_ctime).isoformat(),
                        download_url=download_url,
                    )
                )
    except Exception as exc:
        logger.error(f"Error listing artifacts for {project_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not list project artifacts.",
        )
    return ProjectArtifactsResponse(project_id=project_id, artifacts=artifacts)


@app.get("/projects/{project_id}/artifacts/{artifact_name:path}", name="download_project_artifact")
async def download_project_artifact(
    project_id: str = FastApiPath(..., description="Project ID"),
    artifact_name: str = FastApiPath(..., description="Relative path to the artifact"),
):
    """Download a specific artifact file.  Path-traversal is blocked."""
    project = projects_db.get(project_id)
    if not project or not project.artifacts_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")

    try:
        artifacts_dir = Path(project.artifacts_path).resolve()
        requested = (artifacts_dir / artifact_name).resolve()

        if not str(requested).startswith(str(artifacts_dir)):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")

        if not requested.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact '{artifact_name}' not found.",
            )
        return FileResponse(str(requested), filename=requested.name)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error serving artifact {artifact_name} for {project_id}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not serve artifact.",
        )


@app.get("/projects/{project_id}/logs/", response_model=ProjectLogsResponse)
async def get_project_logs(
    project_id: str = FastApiPath(..., description="Project ID"),
):
    """Return captured logs plus the tail of any on-disk run-log."""
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")

    logs = list(project.captured_logs)

    if project.run_log_path and Path(project.run_log_path).exists():
        try:
            with open(project.run_log_path, encoding="utf-8") as f:
                logs.extend(line.rstrip() for line in f.readlines()[-500:])
        except Exception as exc:
            logs.append(f"[ERROR] Could not read run log: {exc}")
    elif project.run_log_path:
        logs.append(f"[WARNING] Run log not found at: {project.run_log_path}")

    if not logs:
        logs.append("No logs available for this project yet.")

    return ProjectLogsResponse(project_id=project_id, logs=logs)


# ---------------------------------------------------------------------------
# WebSocket endpoint — real-time streaming of logs and agent messages
# ---------------------------------------------------------------------------

@app.websocket("/projects/{project_id}/ws")
async def project_websocket(websocket: WebSocket, project_id: str):
    """Stream real-time events for a project via WebSocket.

    Clients receive JSON objects with a ``type`` field:

    * ``"connected"``  – sent immediately on connection
    * ``"status"``     – project status change, ``data`` is the new status string
    * ``"log"``        – a captured log line, ``data`` is the text
    * ``"message"``    – a new agent message, ``data`` is the message dict
    * ``"error"``      – connection or server error
    """
    if project_id not in projects_db:
        await websocket.close(code=4004, reason="Project not found")
        return

    await websocket.accept()
    ws_connections.setdefault(project_id, []).append(websocket)

    # Send current status immediately so the client doesn't need to poll
    project = projects_db[project_id]
    await websocket.send_json(
        {
            "type": "connected",
            "data": {
                "project_id": project_id,
                "status": project.status,
                "captured_log_count": len(project.captured_logs),
            },
        }
    )

    try:
        # Keep the socket open; the client can send a "ping" to keep it alive.
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data.strip().lower() == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # Send a keepalive so proxies don't close the connection
                await websocket.send_json({"type": "keepalive"})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning(f"WebSocket error for project {project_id}: {exc}")
        try:
            await websocket.send_json({"type": "error", "data": str(exc)})
        except Exception:
            pass
    finally:
        conns = ws_connections.get(project_id, [])
        if websocket in conns:
            conns.remove(websocket)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Verify that the service is up."""
    return {"status": "ok", "projects_in_memory": len(projects_db)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting MetaGPT Frontend API service.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
