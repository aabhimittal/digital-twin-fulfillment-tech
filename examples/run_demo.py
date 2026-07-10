"""End-to-end demo: simulate, break, understand, and predict.

Run with::

    python examples/run_demo.py

This exercises all three "brains" plus the chaos injector on a small warehouse so it
finishes in a few seconds.
"""

from __future__ import annotations

from digital_twin import (
    CascadePredictor,
    CausalInferenceEngine,
    ChaosInjector,
    ChaosScenario,
    DigitalTwin,
    WarehouseConfig,
)


def main() -> None:
    config = WarehouseConfig(num_robots=25, num_stations=8, seed=7)

    # -- Brain 1: run the physics engine ------------------------------- #
    print("=" * 66)
    print("BRAIN 1 — Physics Engine (SimPy)")
    print("=" * 66)
    twin = DigitalTwin(config)
    metrics = twin.run_simulation(duration=3600, num_orders=800)
    for k, v in twin.summary().items():
        print(f"  {k:22s}: {v}")

    # -- Chaos: robot plague vs. baseline ------------------------------ #
    print("\n" + "=" * 66)
    print("CHAOS ENGINEERING — 'robot plague' (30% of the fleet fails)")
    print("=" * 66)
    chaos_twin = DigitalTwin(config)
    injector = ChaosInjector(chaos_twin)
    scenario = ChaosScenario.robot_plague(chaos_twin, fraction=0.3, rng=chaos_twin.rng)
    result = injector.run_chaos_experiment(scenario, duration=3600, num_orders=800)
    impact = result["impact"]
    print(f"  baseline throughput : {impact['baseline_throughput']}")
    print(f"  chaos throughput    : {impact['chaos_throughput']}")
    print(f"  degradation         : {impact['throughput_degradation']:.1%}")
    print(f"  recovery time (min) : {impact['recovery_time']}")
    print(f"  cascade detected    : {impact['cascade_detected']}")

    # -- Brain 2: learn the causal graph ------------------------------- #
    print("\n" + "=" * 66)
    print("BRAIN 2 — Causal Graph (why things happen)")
    print("=" * 66)
    engine = CausalInferenceEngine()
    graph = engine.learn_from_twin(metrics)
    if graph.number_of_edges() == 0:
        print("  (no significant causal edges found in this short run)")
    for u, v, d in graph.edges(data=True):
        print(f"  {u:14s} -> {v:14s}  (strength {d['weight']:.3f})")
    print(f"  ranked root causes  : {engine.rank_root_causes()[:3]}")

    intervention = engine.predict_intervention({"num_failures": 1.0}) \
        if "num_failures" in graph else {}
    if intervention:
        print(f"  do(num_failures+1)  : {intervention}")

    # -- Brain 3: forecast cascade risk -------------------------------- #
    print("\n" + "=" * 66)
    print("BRAIN 3 — Prediction Engine (foresight)")
    print("=" * 66)
    queue_series = [m["queue_length"] for m in metrics["robot_utilization"]]
    predictor = CascadePredictor(higher_is_worse=True)
    prediction = predictor.predict(queue_series, horizon=10)
    print(f"  {prediction.explanation}")
    print(f"  contributing factors: {prediction.contributing_factors}")


if __name__ == "__main__":
    main()
