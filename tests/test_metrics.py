from ruma_ppo.evaluation.metrics import metrics, paired_statistics


def test_metrics_counts_success_and_collision():
    rows = [
        {"outcome": "success", "gates": 3, "near_miss": False, "flight_s": 1.0, "min_clearance_m": 0.5},
        {"outcome": "collision", "gates": 1, "near_miss": True, "flight_s": 2.0, "min_clearance_m": -0.1},
        {"outcome": "missed_gate", "gates": 2, "near_miss": False, "flight_s": 3.0, "min_clearance_m": 0.2},
    ]
    result = metrics(rows)
    assert result["cases"] == 3
    assert result["outcomes"] == {"collision": 1, "missed_gate": 1, "success": 1}
    assert abs(result["three_gate_rate"] - 1 / 3) < 1e-9
    assert result["mean_gates"] == 2.0
    assert result["near_miss_count"] == 1
    assert result["mean_success_s"] == 1.0
    assert result["minimum_clearance_m"] == -0.1


def test_paired_statistics_counts_and_ci():
    cases = [
        {"candidate": {"outcome": "success"}, "v12": {"outcome": "collision"}, "route": "r", "level": 2, "seed": 1},
        {"candidate": {"outcome": "success"}, "v12": {"outcome": "success"}, "route": "r", "level": 2, "seed": 1},
        {"candidate": {"outcome": "collision"}, "v12": {"outcome": "success"}, "route": "r", "level": 3, "seed": 2},
        {"candidate": {"outcome": "missed_gate"}, "v12": {"outcome": "missed_gate"}, "route": "r", "level": 3, "seed": 2},
    ]
    result = paired_statistics(cases)
    assert result["candidate_only"] == 1
    assert result["v12_only"] == 1
    assert result["both"] == 1
    assert result["neither"] == 1
    assert abs(result["gain"]) < 1e-9
    assert len(result["group_bootstrap_95_ci"]) == 2
