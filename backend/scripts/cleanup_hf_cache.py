from __future__ import annotations

import os
import shutil
from pathlib import Path

def _get_target_paths() -> list[Path]:
    raw = os.getenv("HF_CACHE_DIRS", "").strip()
    if raw:
        return [Path(path.strip()) for path in raw.split(",") if path.strip()]

    home = Path.home()
    return [home / ".cache" / "huggingface"]


def format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024 and idx < len(units) - 1:
        value /= 1024
        idx += 1
    return f"{value:.2f} {units[idx]}"


def directory_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                continue
    return total


def is_safe_to_delete(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    home = Path.home().resolve()
    return str(resolved).startswith(str(home)) and "huggingface" in str(resolved).lower()


def delete_cache_path(path: Path) -> None:
    print(f"Checking path: {path}")
    if not path.exists():
        print("No old cache found.")
        return

    if not is_safe_to_delete(path):
        raise RuntimeError(f"Safety check failed for path: {path}")

    size = directory_size_bytes(path)
    print(f"Directory size: {format_size(size)} ({size} bytes)")
    print("Deleting old HuggingFace cache from C drive...")
    shutil.rmtree(path)
    print("Deleted successfully.")


if __name__ == "__main__":
    for target in _get_target_paths():
        try:
            delete_cache_path(target)
        except Exception as exc:
            print(f"Failed to delete {target}: {exc}")
