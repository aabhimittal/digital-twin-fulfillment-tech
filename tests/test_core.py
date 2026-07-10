"""Tests for the physics engine (Brain 1)."""

from digital_twin.core import DigitalTwin, Robot, WarehouseConfig


def _small_config(seed=42):
    return WarehouseConfig(num_robots=10, num_stations=4, seed=seed)


def test_distance_is_manhattan():
    assert Robot.calculate_distance((0, 0, 0), (1, 2, 3)) == 6


def test_simulation_runs_and_produces_metrics():
    twin = DigitalTwin(_small_config())
    metrics = twin.run_simulation(duration=1800, num_orders=200)
    assert metrics["robot_utilization"], "expected utilisation samples"
    summary = twin.summary()
    assert summary["total_picks"] > 0
    assert 0.0 <= summary["avg_utilization"] <= 1.0


def test_simulation_is_reproducible():
    a = DigitalTwin(_small_config(seed=123))
    b = DigitalTwin(_small_config(seed=123))
    sa = a.run_simulation(1800, num_orders=200) and a.summary()
    sb = b.run_simulation(1800, num_orders=200) and b.summary()
    assert sa == sb


def test_different_seeds_diverge():
    a = DigitalTwin(_small_config(seed=1))
    b = DigitalTwin(_small_config(seed=2))
    a.run_simulation(1800, num_orders=200)
    b.run_simulation(1800, num_orders=200)
    # Extremely unlikely to be identical across independent seeds.
    assert a.summary() != b.summary()


def test_battery_never_negative():
    twin = DigitalTwin(_small_config())
    twin.run_simulation(3600, num_orders=400)
    assert all(r.battery >= 0.0 for r in twin.robots)
