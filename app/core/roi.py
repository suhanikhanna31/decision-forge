def calculate_expected_value(
    expected_lift: float,
    revenue: float,
    incentive_cost: float
) -> float:
    """
    Calculates expected monetary value of an intervention.

    expected_lift: increase in probability of user staying due to action
    revenue: revenue generated if user stays
    incentive_cost: cost of retention action (discount, credit, etc.)
    """

    return (expected_lift * revenue) - incentive_cost