"""
search_agent/task_queue.py

Section 8: Background Task Queue & Real-Time Server-Sent Events (SSE) Streaming
Executes 30-90s research runs as background tasks to prevent HTTP request timeouts,
and formats live stage transitions and spinners as Server-Sent Events (SSE).
"""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Generator, List, Optional


def format_sse_event(event_type: str, data: Dict[str, Any], event_id: Optional[str] = None) -> str:
    """Format a dictionary payload as a W3C-compliant Server-Sent Events (SSE) frame."""
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    if event_type:
        lines.append(f"event: {event_type}")
    lines.append(f"data: {json.dumps(data, ensure_ascii=False)}")
    return "\n".join(lines) + "\n\n"


class BackgroundTaskQueue:
    """
    Asynchronous background worker queue for long-running research jobs (30-90s).
    Supports local thread-pool workers out-of-the-box and optional Redis broker URL
    for distributed Celery/ARQ workers.
    """

    def __init__(self, max_workers: int = 4, redis_url: Optional[str] = None):
        self.max_workers = max_workers
        self.redis_url = redis_url
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def enqueue(
        self,
        fn: Callable[..., Any],
        *args: Any,
        topic: str = "",
        **kwargs: Any,
    ) -> str:
        """Enqueue a background research run and return its job_id immediately."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        job_state: Dict[str, Any] = {
            "job_id": job_id,
            "topic": topic,
            "status": "queued",
            "phase": "queued",
            "events": [],
            "result": None,
            "error": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._jobs[job_id] = job_state

        def _emit_job_event(event_name: str, payload: Dict[str, Any]) -> None:
            evt = {
                "index": len(job_state["events"]),
                "event": event_name,
                "payload": payload,
                "timestamp": time.time(),
            }
            job_state["events"].append(evt)
            job_state["phase"] = event_name
            job_state["updated_at"] = time.time()

        def _runner():
            job_state["status"] = "running"
            _emit_job_event("job_started", {"job_id": job_id, "topic": topic})
            try:
                call_kwargs = dict(kwargs)
                if "on_event" not in call_kwargs:
                    call_kwargs["on_event"] = _emit_job_event
                result = fn(*args, **call_kwargs)
                job_state["result"] = result
                job_state["status"] = "completed"
                _emit_job_event("job_completed", {"job_id": job_id})
                return result
            except Exception as exc:
                job_state["status"] = "failed"
                job_state["error"] = str(exc)
                _emit_job_event("job_failed", {"job_id": job_id, "error": str(exc)})
                raise

        job_state["_future"] = self._pool.submit(_runner)
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        return {k: v for k, v in job.items() if k != "_future"}

    def wait_for_job(self, job_id: str, timeout: float = 30.0) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if not job:
            return None
        fut = job.get("_future")
        if fut is not None:
            try:
                fut.result(timeout=timeout)
            except Exception:
                pass
        return self.get_job(job_id)

    def stream_sse(self, job_id: str, poll_interval: float = 0.05) -> Generator[str, None, None]:
        """
        Generator yielding Server-Sent Events (SSE) frames for live stage transitions
        until the background job completes or fails.
        """
        cursor = 0
        while True:
            job = self._jobs.get(job_id)
            if not job:
                yield format_sse_event("error", {"error": f"Unknown job_id: {job_id}"})
                return

            events: List[Dict[str, Any]] = job["events"]
            while cursor < len(events):
                evt = events[cursor]
                yield format_sse_event(
                    evt["event"],
                    {"job_id": job_id, "phase": evt["event"], **evt["payload"]},
                    event_id=str(evt["index"]),
                )
                cursor += 1

            if job["status"] in ("completed", "failed"):
                yield format_sse_event(
                    "done",
                    {"job_id": job_id, "status": job["status"], "error": job["error"]},
                )
                return

            time.sleep(poll_interval)
