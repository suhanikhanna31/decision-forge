class ROICalculator:
    def __init__(self, config):
        self.config = config

    def evaluate(self, inputs):
        churn_prob = inputs.get("churn_probability", 0)
        expected_lift = inputs.get("expected_lift", 0)
        revenue = self.config.get("revenue_per_user", 100)
        cost = self.config.get("incentive_cost", 20)

        expected_value = (churn_prob * expected_lift * revenue) - cost
        roi_positive = expected_value > 0

        return {
            "roi_positive": roi_positive,
            "expected_value": round(expected_value, 2),
            "reason": f"Expected value ${expected_value:.2f} — {'ROI positive' if roi_positive else 'not worth intervening'}",
        }