class SecurityGate:
    def __init__(self, config):
        self.anomaly_threshold = config.get("anomaly_threshold", 0.7)
        self.request_limit = config.get("request_limit", 10)

    def evaluate(self, inputs):
        anomaly_score = inputs.get("anomaly_score", 0)
        requests = inputs.get("request_count_today", 0)

        if anomaly_score > self.anomaly_threshold or requests > self.request_limit:
            return {
                "action": "FLAG",
                "reason": f"Anomaly score {anomaly_score:.2f} or {requests} requests exceeded threshold",
            }
        return {"action": "PASS", "reason": "Security checks passed"}