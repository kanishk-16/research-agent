"""
search_agent/checkpoint_manager.py

Pipeline State Checkpointing Engine:
Persists execution state after:
- Phase 1 (Plan): research_plan
- Phase 2 (Search/Passages): ranked_sources, selection_bundle, retrieval_stats
- Phase 3 (Evidence): extraction_bundle, sufficiency_results, evidence_artifact

Allows instant resumption after transient API crashes without re-running completed phases.
"""

import hashlib
import json
import os
import time
from typing import Any, Dict, Optional


VALID_PHASES = ("phase1_plan", "phase2_search_passages", "phase3_evidence")


class PipelineCheckpointManager:
    """
    Persists and loads per-phase JSON checkpoints for autonomous research runs.
    """

    def __init__(self, checkpoint_dir: str = "data/checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @staticmethod
    def run_key_for_topic(topic: str, run_id: Optional[str] = None) -> str:
        if run_id and str(run_id).strip():
            return re.sub(r"[^A-Za-z0-9_\-]", "_", str(run_id).strip())
        digest = hashlib.sha256(str(topic or "").strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"ckpt_{digest}"

    def _checkpoint_path(self, key: str) -> str:
        return os.path.join(self.checkpoint_dir, f"{key}.json")

    def save_phase(
        self,
        topic: str,
        phase: str,
        state_payload: Dict[str, Any],
        run_id: Optional[str] = None,
    ) -> str:
        """
        Save checkpoint state after completing a pipeline phase ('phase1_plan',
        'phase2_search_passages', or 'phase3_evidence').
        """
        if phase not in VALID_PHASES:
            raise ValueError(f"Unsupported checkpoint phase: {phase}")

        key = self._safe_key(topic, run_id)
        path = self._checkpoint_path(key)
        existing = self.load_checkpoint(topic, run_id=run_id) or {
            "run_key": key,
            "topic": topic,
            "completed_phases": [],
            "phases": {},
        }

        if phase not in existing["completed_phases"]:
            existing["completed_phases"].append(phase)
        existing["latest_phase"] = phase
        existing["updated_at"] = time.time()
        existing["phases"][phase] = state_payload

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        return path

    def load_checkpoint(
        self,
        topic: str,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load the checkpoint bundle for a topic or run_id if present."""
        key = self._safe_key(topic, run_id)
        path = self._checkpoint_path(key)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def get_phase_state(
        self,
        topic: str,
        phase: str,
        run_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ckpt = self.load_checkpoint(topic, run_id=run_id)
        if not ckpt:
            return None
        return ckpt.get("phases", {}).get(phase)

    def clear_checkpoint(self, topic: str, run_id: Optional[str] = None) -> bool:
        key = self._safe_key(topic, run_id)
        path = self._checkpoint_path(key)
        if os.path.exists(path):
            os.remove(path)
            return True
        return False

    @staticmethod
    def _safe_key(topic: str, run_id: Optional[str] = None) -> str:
        if run_id and str(run_id).strip():
            clean = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(run_id).strip())
            return clean
        digest = hashlib.sha256(str(topic or "").strip().lower().encode("utf-8")).hexdigest()[:16]
        return f"ckpt_{digest}"
