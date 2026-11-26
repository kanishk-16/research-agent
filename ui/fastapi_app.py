"""
ui/fastapi_app.py

Section 8: Asynchronous FastAPI Backend with OpenAPI/Swagger Documentation,
Background Task Queue Integration, and Server-Sent Events (SSE) Live Streaming.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from search_agent.auth_manager import JWTAuthManager, MultiTenantWorkspaceManager
from search_agent.task_queue import BackgroundTaskQueue


class ResearchRunRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="Scientific or technical research topic to investigate.")
    research_mode: Optional[str] = Field(default="AUTONOMOUS", description="Execution mode.")
    org_id: Optional[str] = Field(default=None, description="Optional multi-tenant organization ID.")


class ResearchJobStatusResponse(BaseModel):
    job_id: str
    topic: str
    status: str
    phase: str
    events: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None


def create_fastapi_app(
    task_queue: Optional[BackgroundTaskQueue] = None,
    workspace_mgr: Optional[MultiTenantWorkspaceManager] = None,
):
    """
    Create the FastAPI application (or lightweight OpenAPI-compatible ASGI adapter if
    `fastapi` package is not installed in the current environment).
    """
    queue = task_queue or BackgroundTaskQueue()
    ws_mgr = workspace_mgr or MultiTenantWorkspaceManager(db_path=":memory:")
    jwt_mgr = JWTAuthManager()

    try:
        from fastapi import FastAPI, HTTPException, Header
        from fastapi.responses import StreamingResponse

        app = FastAPI(
            title="ARA — Autonomous Research Agent API",
            description="Asynchronous FastAPI backend with Pydantic structured extraction, background job queue, and Server-Sent Events (SSE) progress streaming.",
            version="2.0.0",
        )

        @app.get("/api/health", tags=["System"])
        async def health_check() -> Dict[str, Any]:
            return {"status": "ok", "engine": "fastapi", "version": "2.0.0"}

        @app.post("/api/jobs", response_model=ResearchJobStatusResponse, tags=["Research"])
        async def submit_research_job(
            req: ResearchRunRequest,
            authorization: Optional[str] = Header(default=None),
        ) -> Dict[str, Any]:
            if authorization and authorization.startswith("Bearer "):
                token = authorization[len("Bearer "):].strip()
                try:
                    jwt_mgr.verify_jwt_token(token)
                except ValueError as err:
                    raise HTTPException(status_code=401, detail=str(err))

            def _bg_task(on_event=None):
                if on_event:
                    on_event("planning", {"topic": req.topic})
                return {"topic": req.topic, "status": "completed"}

            job_id = queue.enqueue(_bg_task, topic=req.topic)
            return queue.get_job(job_id)

        @app.get("/api/jobs/{job_id}", response_model=ResearchJobStatusResponse, tags=["Research"])
        async def get_job_status(job_id: str) -> Dict[str, Any]:
            job = queue.get_job(job_id)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found")
            return job

        @app.get("/api/stream/{job_id}", tags=["Streaming"])
        async def stream_job_progress(job_id: str):
            return StreamingResponse(
                queue.stream_sse(job_id),
                media_type="text/event-stream",
            )

        return app

    except ImportError:
        # Lightweight OpenAPI & route adapter when fastapi package is not installed locally
        class FastAPIAdapter:
            title = "ARA — Autonomous Research Agent API"
            version = "2.0.0"

            def __init__(self):
                self.queue = queue
                self.workspace_mgr = ws_mgr
                self.jwt_mgr = jwt_mgr

            def openapi(self) -> Dict[str, Any]:
                return {
                    "openapi": "3.1.0",
                    "info": {"title": self.title, "version": self.version},
                    "paths": {
                        "/api/health": {"get": {"summary": "Health Check"}},
                        "/api/jobs": {"post": {"summary": "Enqueue Background Research Run"}},
                        "/api/jobs/{job_id}": {"get": {"summary": "Get Background Job Status"}},
                        "/api/stream/{job_id}": {"get": {"summary": "Stream Server-Sent Events (SSE)"}},
                    },
                }

            def submit_job(self, req: ResearchRunRequest, bearer_token: Optional[str] = None) -> Dict[str, Any]:
                if bearer_token:
                    self.jwt_mgr.verify_jwt_token(bearer_token)

                def _bg_task(on_event=None):
                    if on_event:
                        on_event("planning", {"topic": req.topic})
                    return {"topic": req.topic, "status": "completed"}

                jid = self.queue.enqueue(_bg_task, topic=req.topic)
                return self.queue.get_job(jid)

            def stream_job_sse(self, job_id: str):
                return self.queue.stream_sse(job_id)

        return FastAPIAdapter()
