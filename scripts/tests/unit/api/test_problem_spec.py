from campro.api import ProblemSpec


def test_problem_spec_basic_fields():
    spec = ProblemSpec(
        stroke=20.0,
        cycle_time=1.0,
        phases={"upstroke_percent": 60.0, "zero_accel_percent": 0.0},
        bounds={"max_velocity": 100.0, "max_acceleration": 1000.0, "max_jerk": 10000.0},
        objective="minimum_jerk",
    )
    assert spec.stroke == 20.0
    assert spec.cycle_time == 1.0
    assert spec.phases["upstroke_percent"] == 60.0
    assert spec.bounds["max_velocity"] == 100.0
    assert spec.objective == "minimum_jerk"
