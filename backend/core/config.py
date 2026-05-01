from __future__ import annotations

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    with ENV_PATH.open("r", encoding="utf-8-sig") as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from ..config import get_allowed_origins, get_max_upload_bytes

__all__ = ["get_allowed_origins", "get_max_upload_bytes"]
