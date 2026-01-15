from datetime import datetime


def log_decision(inputs: dict, decision: dict, config: dict) -> None:
    """
    Logs decision details for audit and explainability.
    """

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "inputs": inputs,
        "decision": decision,
        "client": config.get("client_name", "unknown")
    }

    
    print(log_entry)