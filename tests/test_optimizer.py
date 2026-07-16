"""Tests for the decision-optimization layer."""

import numpy as np

from digital_twin.causal import CausalInferenceEngine
from digital_twin.core import WarehouseConfig
from digital_twin.optimizer import (
    BanditOptimizer,
    GreedyOptimizer,
    Lever,
    default_levers,
    throughput_objective,
)


def _small_config(seed=3):
    return WarehouseConfig(num_robots=10, num_stations=4, seed=seed)


def test_greedy_ranks_and_includes_baseline():
    opt = GreedyOptimizer(_small_config(), duration=1200, num_orders=200)
    recs = opt.recommend()
    names = [r.lever for r in recs]
    assert "do_nothing" in names
    # Ranking is sorted by delta descending.
    deltas = [r.delta for r in recs]
    assert deltas == sorted(deltas, reverse=True)


def test_greedy_more_robots_beats_fewer():
    opt = GreedyOptimizer(_small_config(), objective=throughput_objective,
                          duration=1800, num_orders=400)
    levers = [
        Lever("do_nothing", lambda cfg: None, cost=0.0),
        Lever("add_10", lambda cfg: setattr(cfg, "num_robots", cfg.num_robots + 10), cost=10),
        Lever("remove_5", lambda cfg: setattr(cfg, "num_robots", max(1, cfg.num_robots - 5)), cost=1),
    ]
    recs = {r.lever: r.delta for r in opt.recommend(levers)}
    assert recs["add_10"] > recs["remove_5"]


def test_budget_filters_expensive_levers():
    opt = GreedyOptimizer(_small_config(), duration=900, num_orders=150)
    recs = opt.recommend(budget=3.0)
    assert all(r.cost <= 3.0 for r in recs)
    assert "add_10_robots" not in [r.lever for r in recs]


def test_optimizer_does_not_mutate_base_config():
    cfg = _small_config()
    before = cfg.num_robots
    GreedyOptimizer(cfg, duration=600, num_orders=100).recommend()
    assert cfg.num_robots == before


def test_bandit_optimizer_uses_causal_predictions():
    # Build a causal graph where queue_length -> utilization.
    rng = np.random.default_rng(0)
    n = 200
    queue = rng.normal(0, 1, n)
    util = -0.8 * queue + rng.normal(0, 0.1, n)
    obs = [{"queue_length": float(queue[i]), "utilization": float(util[i])} for i in range(n)]
    engine = CausalInferenceEngine()
    engine.learn(obs)

    bandit = BanditOptimizer(engine, objective_var="utilization", seed=1)
    recs = bandit.recommend()
    assert recs, "expected recommendations"
    # A lever that lowers queue_length should be predicted to raise utilization.
    by_name = {r.lever: r for r in recs}
    assert by_name["add_3_stations"].detail["predicted_effect"] > 0


def test_bandit_update_tracks_realised_reward():
    engine = CausalInferenceEngine()
    engine.learn([{"queue_length": float(i), "utilization": float(-i)} for i in range(20)])
    bandit = BanditOptimizer(engine, seed=2)
    lever = default_levers()[1]
    bandit.update(lever, 1.0)
    bandit.update(lever, 3.0)
    assert bandit._reward_estimates[lever.name] == 2.0
    assert bandit._counts[lever.name] == 2
