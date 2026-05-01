from __future__ import annotations

import os
import time

TASKS: dict[str, dict[str, float | str]] = {}
TASK_TTL_SECONDS = 60 * 60
TASK_TIMEOUT_SECONDS = int(os.getenv("TASK_TIMEOUT_SECONDS", "1800"))


def _prune_tasks(now: float | None = None) -> None:
    current = now if now is not None else time.time()
    expired: list[str] = []
    for task_id, task in TASKS.items():
        created_at = float(task.get("created_at", 0))
        if current - created_at > TASK_TTL_SECONDS:
            expired.append(task_id)
    for task_id in expired:
        TASKS.pop(task_id, None)


def _apply_timeout(task_id: str, task: dict[str, float | str], now: float) -> None:
    status = str(task.get("status", "pending"))
    if status not in {"pending", "processing"}:
        return
    created_at = float(task.get("created_at", 0))
    if now - created_at <= TASK_TIMEOUT_SECONDS:
        return
    task["status"] = "failed"
    task["error"] = "Task timed out"


def create_task(task_id: str) -> None:
    _prune_tasks()
    TASKS[task_id] = {"status": "pending", "created_at": time.time(), "error": ""}


def set_task_status(task_id: str, status: str) -> None:
    task = TASKS.get(task_id)
    if not task:
        TASKS[task_id] = {"status": status, "created_at": time.time(), "error": ""}
        return
    task["status"] = status


def set_task_error(task_id: str, error: str) -> None:
    task = TASKS.get(task_id)
    if not task:
        TASKS[task_id] = {"status": "failed", "created_at": time.time(), "error": error}
        return
    task["error"] = error


def get_task_status(task_id: str) -> dict[str, float | str] | None:
    _prune_tasks()
    task = TASKS.get(task_id)
    if not task:
        return None
    _apply_timeout(task_id, task, time.time())
    return task
