"""Tests for the chaos injector."""

import numpy as np

from digital_twin.chaos import ChaosInjector, ChaosScenario, IncidentType
from digital_twin.core import DigitalTwin, WarehouseConfig


def _config():
    return WarehouseConfig(num_robots=12, num_stations=4, seed=5)


def test_robot_plague_scenario_size():
    twin = DigitalTwin(_config())
    scenario = ChaosScenario.robot_plague(twin, fraction=0.5, rng=np.random.default_rng(0))
    assert len(scenario) == 6
    assert all(i.type is IncidentType.ROBOT_FAILURE for i in scenario)


def test_station_overload_slows_stations():
    twin = DigitalTwin(_config())
    injector = ChaosInjector(twin)
    before = twin.stations[0].avg_processing_time
    scenario = ChaosScenario.black_friday(twin)
    injector.inject_incident(scenario[0])
    assert twin.stations[0].avg_processing_time > before


def test_chaos_experiment_reports_impact():
    twin = DigitalTwin(_config())
    injector = ChaosInjector(twin)
    scenario = ChaosScenario.robot_plague(twin, fraction=0.5, rng=twin.rng)
    result = injector.run_chaos_experiment(scenario, duration=1800, num_orders=300)
    impact = result["impact"]
    assert "throughput_degradation" in impact
    assert impact["baseline_throughput"] >= 0


def test_bandit_selects_valid_incident():
    twin = DigitalTwin(_config())
    injector = ChaosInjector(twin)
    for _ in range(20):
        assert injector.select_incident() in list(IncidentType)
