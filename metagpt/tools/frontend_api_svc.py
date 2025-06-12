#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/07/17
@Author  : Your Name / AI Agent
@File    : frontend_api_svc.py
@Desc    : FastAPI service for MetaGPT frontend interactions.
"""

from fastapi import FastAPI, HTTPException, Path as FastApiPath, status, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from pathlib import Path
import uuid
import asyncio
from datetime import datetime

# Attempt to import SoftwareCompany, will be None if metagpt is not in PYTHONPATH during standalone run
try:
    from metagpt.software_company import SoftwareCompany
    from metagpt.context import Context # Needed for SoftwareCompany
    from metagpt.config2 import config as metagpt_config # To get workspace path
except ImportError:
    SoftwareCompany = None
    Context = None
    metagpt_config = None # type: ignore
    # logger will be defined below, so can't use it here yet.
    # print("WARNING: MetaGPT parts not found. API will run in mock mode for project generation.")

import os
import json # For parsing messages if stored in JSON lines format
from starlette.responses import FileResponse # For downloading artifacts


# --- Logger Setup ---
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if SoftwareCompany is None :
    logger.warning("MetaGPT's SoftwareCompany or other core components not found. API will run in MOCK MODE for project generation.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="MetaGPT Frontend API Service",
    description="Provides API endpoints for interacting with MetaGPT projects and their artifacts from a frontend.",
    version="0.1.0",
)


# --- In-memory Database for Projects ---
projects_db: Dict[str, 'Project'] = {}


# --- Pydantic Models ---

class Project(BaseModel):
    """Represents a project in the system."""
    project_id: str
    idea: str
    project_name: Optional[str] = None
    status: str = Field(default="pending", description="Status: pending, running, completed, failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    artifacts_path: Optional[str] = None # Path where generated artifacts are stored
    message_log_path: Optional[str] = None # Path to the message log file
    run_log_path: Optional[str] = None # Path to the detailed run log file
    captured_logs: List[str] = Field(default_factory=list) # For logs captured directly during run
    error_message: Optional[str] = None

class ProjectCreateRequest(BaseModel):
    """Request model for creating a new project."""
    idea: str = Field(..., description="The initial idea or requirement for the project.")
    project_name: Optional[str] = Field(None, description="Optional name for the project.")

class ProjectCreateResponse(BaseModel):
    """Response model for project creation."""
    project_id: str = Field(..., description="Unique identifier for the newly created project.")
    message: str = Field(default="Project creation initiated.", description="Status message.")
    status: str = Field(default="pending", description="Initial status of the project.")

class ProjectStatusResponse(BaseModel):
    """Response model for project status."""
    project_id: str
    project_name: Optional[str] = None
    idea: str
    status: str = Field(..., description="Current status of the project (e.g., 'running', 'completed', 'failed').")
    created_at: datetime
    updated_at: datetime
    artifacts_path: Optional[str] = None
    error_message: Optional[str] = None

class MessageModel(BaseModel):
    """Model for a single message within a project."""
    message_id: str
    role: str
    content: str
    timestamp: str

class ProjectMessagesResponse(BaseModel):
    """Response model for project messages."""
    project_id: str
    messages: List[MessageModel]

class ArtifactModel(BaseModel):
    """Model for a single project artifact."""
    artifact_name: str
    artifact_type: str
    created_at: str
    download_url: str # Added download URL

class ProjectArtifactsResponse(BaseModel):
    """Response model for project artifacts list."""
    project_id: str
    artifacts: List[ArtifactModel]

class ProjectLogsResponse(BaseModel):
    """Response model for project logs."""
    project_id: str
    logs: List[str]


# --- API Endpoints ---

@app.post("/projects/", response_model=ProjectCreateResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_project(request: ProjectCreateRequest, background_tasks: BackgroundTasks): # Added BackgroundTasks
    logger.info(f"Received project creation request: {request.model_dump()}")
    project_id = str(uuid.uuid4())
    project_name = request.project_name or f"project_{project_id[:8]}"

    new_project = Project(
        project_id=project_id,
        idea=request.idea,
        project_name=project_name,
        status="pending"
    )
    projects_db[project_id] = new_project
    background_tasks.add_task(run_metagpt_project, project_id, request.idea, project_name)
    return ProjectCreateResponse(project_id=project_id, status=new_project.status)

async def run_metagpt_project(project_id: str, idea: str, project_name: str):
    project = projects_db.get(project_id)
    if not project:
        logger.error(f"Project {project_id} not found in db for background task.")
        return

    project.status = "running"
    project.updated_at = datetime.utcnow()
    project.captured_logs.append(f"[{datetime.utcnow().isoformat()}] Project status updated to running.")
    logger.info(f"Project {project_id} status updated to running.")

    try:
        if SoftwareCompany is None or Context is None or metagpt_config is None:
            logger.warning(f"SoftwareCompany, Context, or metagpt_config not imported. Simulating project {project_id} run.")
            await asyncio.sleep(5) # Simulate work

            # Mock artifacts path
            mock_artifacts_path = Path("workspace") / project_name
            mock_artifacts_path.mkdir(parents=True, exist_ok=True)
            project.artifacts_path = str(mock_artifacts_path.resolve())

            # Create dummy files for mock mode
            (mock_artifacts_path / "README.md").write_text(f"# Mock Project: {project_name}\nIdea: {idea}")
            (mock_artifacts_path / "main.py").write_text("print('Hello from mock project!')")
            project.message_log_path = str(mock_artifacts_path / "messages.log")
            project.run_log_path = str(mock_artifacts_path / "run.log")

            with open(project.message_log_path, "w", encoding="utf-8") as f_msg:
                json.dump({"id": "msg_user_1", "role": "User", "content": idea, "timestamp": datetime.utcnow().isoformat()}, f_msg)
                f_msg.write("\n")
                json.dump({"id": "msg_ai_1", "role": "AI Assistant", "content": "Okay, I will start planning.", "timestamp": datetime.utcnow().isoformat()}, f_msg)
                f_msg.write("\n")

            with open(project.run_log_path, "w", encoding="utf-8") as f_log:
                f_log.write(f"[{datetime.utcnow().isoformat()}] [INFO] Mock project run started.\n")
                f_log.write(f"[{datetime.utcnow().isoformat()}] [INFO] Mock project run finished successfully.\n")

            project.captured_logs.append(f"[{datetime.utcnow().isoformat()}] Mock project run completed.")
            project.status = "completed"
            logger.info(f"Mock project {project_id} completed. Artifacts at: {project.artifacts_path}")
        else:
            software_company = SoftwareCompany()
            base_workspace = Path(metagpt_config.workspace_path if metagpt_config.workspace_path else "workspace")

            # SoftwareCompany might derive its own project name / directory structure
            # We will try to retrieve it from the context after the run.
            await software_company.run(idea=idea) # This is a blocking call in current MetaGPT

            final_project_path = None
            if hasattr(software_company, 'context') and software_company.context and \
               hasattr(software_company.context, 'repo') and software_company.context.repo and \
               hasattr(software_company.context.repo, 'workdir') and software_company.context.repo.workdir:
                final_project_path = Path(software_company.context.repo.workdir)
            elif hasattr(software_company, 'project_path') and software_company.project_path: # Older MetaGPT versions
                 final_project_path = Path(software_company.project_path)

            if not final_project_path: # Fallback if path not found
                # This guess might be inaccurate if MetaGPT uses a different naming scheme (e.g., from idea)
                final_project_path = base_workspace / project_name
                logger.warning(f"Could not reliably determine project artifact path for {project_id}. Guessed: {final_project_path}")

            if project.project_name != final_project_path.name:
                logger.info(f"Project name in DB ('{project.project_name}') differs from generated ('{final_project_path.name}'). Updating.")
                project.project_name = final_project_path.name

            project.artifacts_path = str(final_project_path.resolve())
            project.status = "completed"
            project.message_log_path = str(final_project_path / "messages.log") # Standard MetaGPT practice
            project.run_log_path = str(final_project_path / "logs" / "run.log") # Common log location

            # Capture some final logs
            project.captured_logs.append(f"[{datetime.utcnow().isoformat()}] Project generation completed. Artifacts at: {project.artifacts_path}")
            if Path(project.run_log_path).exists():
                 with open(project.run_log_path, "r", encoding="utf-8") as f_log:
                    project.captured_logs.extend(f_log.read().splitlines()[-20:]) # Capture last 20 lines
            logger.info(f"MetaGPT project {project_id} generation completed. Artifacts at: {project.artifacts_path}")

    except Exception as e:
        logger.error(f"Error running MetaGPT project {project_id}: {e}", exc_info=True)
        project.status = "failed"
        project.error_message = str(e)
        project.captured_logs.append(f"[{datetime.utcnow().isoformat()}] Error: {str(e)}")
    finally:
        project.updated_at = datetime.utcnow()
        logger.info(f"Project {project_id} final status: {project.status}")

@app.get("/projects/{project_id}/", response_model=ProjectStatusResponse)
async def get_project_status(project_id: str = FastApiPath(..., description="The ID of the project to query.")):
    logger.info(f"Fetching status for project_id: {project_id}")
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")
    return ProjectStatusResponse(
        project_id=project.project_id, project_name=project.project_name, idea=project.idea,
        status=project.status, created_at=project.created_at, updated_at=project.updated_at,
        artifacts_path=project.artifacts_path, error_message=project.error_message
    )

@app.get("/projects/{project_id}/messages/", response_model=ProjectMessagesResponse)
async def get_project_messages(project_id: str = FastApiPath(..., description="The ID of the project.")):
    logger.info(f"Fetching messages for project_id: {project_id}")
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")

    messages = []
    if not project.message_log_path or not Path(project.message_log_path).exists():
        logger.warning(f"Message log file not found for project {project_id} at {project.message_log_path}")
        return ProjectMessagesResponse(project_id=project_id, messages=[])

    try:
        with open(project.message_log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    msg_data = json.loads(line) # Assuming JSON line format from MetaGPT's Message.model_dump_json()
                    messages.append(MessageModel(
                        message_id=msg_data.get("id", f"msg_{i+1}_{project_id}"),
                        role=msg_data.get("role", "Unknown"),
                        content=msg_data.get("content", ""),
                        timestamp=msg_data.get("sent_at", datetime.utcnow().isoformat()) # 'sent_at' is in Message
                    ))
                except json.JSONDecodeError: # Fallback for non-JSON lines
                    parts = line.split(":", 1)
                    role, content = (parts[0].strip(), parts[1].strip()) if len(parts) > 1 else ("LogEntry", line)
                    messages.append(MessageModel(message_id=f"line_{i+1}", role=role, content=content, timestamp=datetime.utcnow().isoformat()))
    except Exception as e:
        logger.error(f"Error reading messages for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not read project messages.")
    return ProjectMessagesResponse(project_id=project_id, messages=messages)

@app.get("/projects/{project_id}/artifacts/", response_model=ProjectArtifactsResponse)
async def list_project_artifacts(project_id: str = FastApiPath(..., description="The ID of the project.")):
    logger.info(f"Listing artifacts for project_id: {project_id}")
    project = projects_db.get(project_id)
    if not project or not project.artifacts_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project artifacts path not found.")

    artifact_root = Path(project.artifacts_path)
    if not artifact_root.is_dir():
        logger.warning(f"Artifacts directory {artifact_root} does not exist for project {project_id}.")
        return ProjectArtifactsResponse(project_id=project_id, artifacts=[])

    artifacts = []
    try:
        for item_path in artifact_root.rglob("*"):
            if item_path.is_file() and not item_path.name.startswith('.'):
                relative_path = item_path.relative_to(artifact_root)
                download_url = app.url_path_for("download_project_artifact", project_id=project_id, artifact_name=str(relative_path))
                artifacts.append(ArtifactModel(
                    artifact_name=str(relative_path),
                    artifact_type=item_path.suffix.lstrip('.').lower() or "file",
                    created_at=datetime.fromtimestamp(item_path.stat().st_ctime).isoformat(),
                    download_url=download_url
                ))
    except Exception as e:
        logger.error(f"Error listing artifacts for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not list project artifacts.")
    return ProjectArtifactsResponse(project_id=project_id, artifacts=artifacts)

@app.get("/projects/{project_id}/artifacts/{artifact_name:path}")
async def download_project_artifact(
    project_id: str = FastApiPath(..., description="The ID of the project."),
    artifact_name: str = FastApiPath(..., description="The relative path of the artifact to download.")
):
    logger.info(f"Request to download artifact: {artifact_name} from project: {project_id}")
    project = projects_db.get(project_id)
    if not project or not project.artifacts_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project or artifacts path not found.")

    try:
        artifacts_dir = Path(project.artifacts_path).resolve()
        requested_path = (artifacts_dir / artifact_name).resolve()

        if not str(requested_path).startswith(str(artifacts_dir)):
            logger.error(f"Path traversal attempt: {artifact_name} for project {project_id}")
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied: Invalid path.")

        if not requested_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Artifact '{artifact_name}' not found or is not a file.")

        return FileResponse(str(requested_path), filename=requested_path.name)
    except HTTPException: # Re-raise HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error serving artifact {artifact_name} for project {project_id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not serve artifact.")

@app.get("/projects/{project_id}/logs/", response_model=ProjectLogsResponse)
async def get_project_logs(project_id: str = FastApiPath(..., description="The ID of the project.")):
    logger.info(f"Fetching logs for project_id: {project_id}")
    project = projects_db.get(project_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found.")

    project_logs = list(project.captured_logs)

    if project.run_log_path and Path(project.run_log_path).exists():
        try:
            with open(project.run_log_path, 'r', encoding='utf-8') as f:
                run_log_lines = f.readlines()
                project_logs.extend([line.strip() for line in run_log_lines[-500:]])
        except Exception as e:
            logger.error(f"Error reading run log file for project {project_id}: {e}")
            project_logs.append(f"[ERROR] Could not read run log file: {project.run_log_path}")
    elif project.run_log_path:
        project_logs.append(f"[WARNING] Run log file configured but not found at: {project.run_log_path}")

    if not project_logs: # Provide a default message if no logs are available
        project_logs.append("No logs available for this project yet.")

    return ProjectLogsResponse(project_id=project_id, logs=project_logs)


# --- Health Check Endpoint ---
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint to verify the API is running."""
    return {"status": "ok"}


# --- Main Entry Point (for local debugging) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI service for MetaGPT Frontend API with Uvicorn.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
