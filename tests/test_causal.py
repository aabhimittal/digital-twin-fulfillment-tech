"""Tests for the causal-inference engine (Brain 2)."""

import numpy as np

from digital_twin.causal import CausalInferenceEngine, discover_causal_structure


def test_discovers_known_chain():
    # Build data where x -> y -> z (num_failures drives queue drives throughput).
    rng = np.random.default_rng(0)
    n = 300
    num_failures = rng.normal(0, 1, n)
    queue_length = 2.0 * num_failures + rng.normal(0, 0.2, n)
    throughput = -1.5 * queue_length + rng.normal(0, 0.2, n)
    obs = [
        {"num_failures": float(num_failures[i]),
         "queue_length": float(queue_length[i]),
         "throughput": float(throughput[i])}
        for i in range(n)
    ]
    graph = discover_causal_structure(obs)
    # There should be a directed path from the root cause to the downstream KPI.
    assert graph.has_edge("num_failures", "queue_length")
    assert graph.number_of_edges() >= 2


def test_predict_intervention_propagates():
    rng = np.random.default_rng(1)
    n = 300
    num_failures = rng.normal(0, 1, n)
    queue_length = 2.0 * num_failures + rng.normal(0, 0.2, n)
    obs = [
        {"num_failures": float(num_failures[i]), "queue_length": float(queue_length[i])}
        for i in range(n)
    ]
    engine = CausalInferenceEngine()
    engine.learn(obs)
    effects = engine.predict_intervention({"num_failures": 1.0})
    assert "queue_length" in effects
    assert effects["queue_length"] != 0.0
    assert 0.0 <= effects["confidence"] <= 1.0


def test_empty_observations_return_empty_graph():
    assert discover_causal_structure([]).number_of_nodes() == 0


def test_root_cause_ranking():
    rng = np.random.default_rng(2)
    n = 200
    root = rng.normal(0, 1, n)
    obs = [
        {"num_failures": float(root[i]),
         "queue_length": float(2 * root[i] + rng.normal(0, 0.1)),
         "throughput": float(-root[i] + rng.normal(0, 0.1))}
        for i in range(n)
    ]
    engine = CausalInferenceEngine()
    engine.learn(obs)
    assert engine.rank_root_causes()[0] == "num_failures"
