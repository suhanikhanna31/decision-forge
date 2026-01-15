from typing import Dict
from app.core.roi import calculate_expected_value
from app.core.security import is_high_risk
from app.core.audit import log_decision


class DecisionEngine:
    def __init__(self, config: Dict):
        self.config = config

    def decide(self, inputs: Dict) -> Dict:
        if is_high_risk(
            anomaly_score=inputs["anomaly_score"],
            request_count=inputs["request_count_today"],
            config=self.config
        ):
            decision = {
                "decision": "FLAG",
                "reason": "HIGH_RISK",
                "expected_value": 0.0
            }
            log_decision(inputs, decision, self.config)
            return decision

        expected_value = calculate_expected_value(
            expected_lift=inputs["expected_lift"],
            revenue=self.config["revenue_per_user"],
            incentive_cost=self.config["incentive_cost"]
        )

        if expected_value > self.config.get("roi_threshold", 0):
            decision = {
                "decision": "INTERVENE",
                "reason": "ROI_POSITIVE",
                "expected_value": float(round(expected_value, 2))
            }
        else:
            decision = {
                "decision": "DO_NOTHING",
                "reason": "ROI_NEGATIVE",
                "expected_value": float(round(expected_value, 2))
            }

        log_decision(inputs, decision, self.config)
        return decision