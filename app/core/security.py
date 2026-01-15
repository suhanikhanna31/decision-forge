# app/core/security.py

def is_high_risk(
    anomaly_score: float,
    request_count: int,
    config: dict
) -> bool:
    """
    Determines whether a user is high risk based on behavioral anomalies
    and request velocity.
    """

    if anomaly_score >= config["security"]["anomaly_threshold"]:
        return True

    if request_count > config["security"]["max_requests_per_day"]:
        return True

    return False