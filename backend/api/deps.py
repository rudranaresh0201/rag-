from __future__ import annotations

import os
from fastapi import Header, HTTPException


def require_api_key(x_api_key: str = Header(default="")) -> None:
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="API key not configured.")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")
