"""Mutable session state for the agent-dev (code-authoring) track.

This is distinct from agent_pipeline.session_state.SessionState: this
session doesn't hold a dataset or pipeline, it holds a code-review
workflow, at most one pending file write proposed but not yet applied,
plus a results_store for the chat widget's generic result-rendering
(shared with agent_pipeline/chat_widget.py).
"""

import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class ResultRecord:
    result_id: str
    tool_name: str
    args: dict
    summary: dict
    created_at: str


@dataclass
class PendingWrite:
    file_path: str
    content: str
    explanation: str


class DevSessionState:
    def __init__(self, workspace_dir="agent_dev_workspace"):
        self.workspace_dir = workspace_dir
        self.pending_write: Optional[PendingWrite] = None
        self.results_store: dict[str, dict] = {}
        self.pending_figure: Optional[Any] = None  # unused here, kept for chat_widget compatibility
        self._counter = itertools.count(1)

    def next_result_id(self, tool_prefix: str) -> str:
        return f"{tool_prefix}_{next(self._counter)}"

    def store_result(self, tool_name: str, args: dict, summary: dict, full_report=None) -> str:
        result_id = summary["result_id"]
        self.results_store[result_id] = {
            "record": ResultRecord(
                result_id=result_id,
                tool_name=tool_name,
                args=args,
                summary=summary,
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
            "full_report": full_report,
        }
        return result_id
