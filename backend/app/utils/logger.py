import json
from datetime import datetime
from pathlib import Path

LOG_PATH = Path(r"backend\logs\events.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def log_event(event: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        **event
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
