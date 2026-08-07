import json
from datetime import datetime

class AuditLogger:
    def log(self, inputs, decision):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "inputs": inputs,
            "decision": decision,
        }
        print(f"[AUDIT] {json.dumps(record)}")