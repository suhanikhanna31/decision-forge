"""
Decision Engine
---------------
Converts ML scores into accountable business actions with latency tracking.

System Design Trade-offs
-------------------------
1. Latency vs Accuracy
   - We use Logistic Regression + Isolation Forest instead of deep models.
     This gives ~0.1–2ms inference latency vs 50–200ms for neural networks.
     Acceptable accuracy loss for real-time customer-facing decisions.

2. Synchronous vs Async
   - Current implementation is synchronous (simple, easy to debug).
   - For >1000 req/sec, this should move to async FastAPI + worker pool.

3. Thresholds in YAML vs Model
   - ROI/anomaly thresholds live in YAML, not hardcoded.
   - This lets business teams tune without re-deploying code.

4. Audit Log
   - Every decision is logged with inputs + rationale.
   - Adds ~0.5ms overhead per request but is mandatory for compliance.
"""

import time
from app.core.roi import ROICalculator
from app.core.security import SecurityGate
from app.core.audit import AuditLogger


class DecisionEngine:
    """
    Routes ML predictions to business actions: INTERVENE, DO_NOTHING, or FLAG.

    Decision hierarchy (evaluated in order):
        1. Security Gate  — block or flag suspicious behavior first
        2. ROI Check      — only intervene if expected value is positive
        3. Default        — do nothing
    """

    def __init__(self, config: dict):
        """
        Args:
            config : loaded YAML config dict containing thresholds and costs
        """
        self.config = config
        self.roi_calculator = ROICalculator(config)
        self.security_gate = SecurityGate(config)
        self.audit_logger = AuditLogger()

    def decide(self, inputs: dict) -> dict:
        """
        Make a real-time decision for a single user.

        Args:
            inputs : dict containing:
                - churn_probability  (float)  : model output, 0–1
                - expected_lift      (float)  : estimated retention lift from intervention
                - anomaly_score      (float)  : 0–1, higher = more suspicious
                - request_count_today(int)    : raw request volume feature

        Returns:
            dict with keys:
                - decision       : "INTERVENE" | "DO_NOTHING" | "FLAG"
                - reason         : human-readable explanation string
                - expected_value : estimated $ value of the decision
                - latency_ms     : time taken for this decision in milliseconds
        """
        start_time = time.perf_counter()

        # --- Step 1: Security Gate ---
        security_result = self.security_gate.evaluate(inputs)
        if security_result["action"] in ("BLOCK", "FLAG"):
            decision = {
                "decision": "FLAG",
                "reason": security_result["reason"],
                "expected_value": 0.0,
            }
            latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
            decision["latency_ms"] = latency_ms
            self.audit_logger.log(inputs, decision)
            return decision

        # --- Step 2: ROI Check ---
        roi_result = self.roi_calculator.evaluate(inputs)
        if roi_result["roi_positive"]:
            decision = {
                "decision": "INTERVENE",
                "reason": roi_result["reason"],
                "expected_value": roi_result["expected_value"],
            }
        else:
            decision = {
                "decision": "DO_NOTHING",
                "reason": roi_result["reason"],
                "expected_value": roi_result["expected_value"],
            }

        # --- Step 3: Record latency ---
        latency_ms = round((time.perf_counter() - start_time) * 1000, 3)
        decision["latency_ms"] = latency_ms

        # --- Step 4: Audit ---
        self.audit_logger.log(inputs, decision)

        return decision

    def get_latency_profile(self) -> dict:
        """
        Return system design notes on latency vs accuracy trade-offs.
        Useful for API /health and /metrics endpoints.
        """
        return {
            "model_inference_ms": "~0.1–2ms (Logistic Regression + Isolation Forest)",
            "roi_calc_ms": "~0.01ms",
            "security_gate_ms": "~0.01ms",
            "audit_log_ms": "~0.5ms",
            "total_p99_ms": "~5ms",
            "trade_off_note": (
                "Chosen models optimize for latency. Switching to XGBoost would "
                "improve F1 by ~5–10% but increase inference latency to ~10–20ms. "
                "Acceptable for batch use cases, not for <5ms SLA requirements."
            ),
        }