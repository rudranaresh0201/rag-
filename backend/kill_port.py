from __future__ import annotations

import re
import subprocess
import sys
from typing import Set

PORT = "8003"


def find_listening_pids(port: str) -> Set[str]:
    result = subprocess.run(
        f"netstat -ano | findstr :{port}",
        capture_output=True,
        text=True,
        shell=True,
    )
    output = result.stdout
    if not output.strip():
        return set()

    pids: set[str] = set()
    for line in output.strip().splitlines():
        if "LISTENING" not in line.upper():
            continue
        parts = re.split(r"\s+", line.strip())
        if not parts:
            continue
        pid = parts[-1]
        if pid.isdigit():
            pids.add(pid)
    return pids


def main() -> int:
    # Step 1: Find process(es) using the target port.
    pids = find_listening_pids(PORT)
    if not pids:
        print(f"No process using port {PORT}")
        return 0

    # Step 2: Kill each PID and retry to handle auto-respawn.
    for attempt in range(1, 4):
        pids = find_listening_pids(PORT)
        if not pids:
            print("Done. Port is free.")
            return 0

        print(f"Attempt {attempt}: found PID(s) {', '.join(sorted(pids))}")
        for pid in sorted(pids):
            print(f"Killing PID: {pid}")
            subprocess.run(
                f"taskkill /F /PID {pid}",
                shell=True,
                check=False,
                capture_output=True,
                text=True,
            )

    remaining = find_listening_pids(PORT)
    if remaining:
        print(
            "Port is still in use by PID(s): "
            + ", ".join(sorted(remaining))
            + ". Run terminal as Administrator and try again."
        )
        return 1

    print("Done. Port is free.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
