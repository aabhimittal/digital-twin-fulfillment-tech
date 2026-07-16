"""Command-line entry point that wires the three brains together.

Examples
--------
Run a plain simulation::

    python -m digital_twin.cli simulate --robots 30 --duration 3600

Run a chaos experiment and print the impact::

    python -m digital_twin.cli chaos --scenario robot_plague --duration 3600

Learn the causal graph and forecast cascade risk from a run::

    python -m digital_twin.cli analyze --duration 3600

Recommend the best operational intervention within a budget::

    python -m digital_twin.cli optimize --duration 3600 --budget 10
"""

from __future__ import annotations

import argparse
import json
from typing import List

from digital_twin.causal import CausalInferenceEngine
from digital_twin.chaos import ChaosInjector, ChaosScenario
from digital_twin.core import DigitalTwin, WarehouseConfig
from digital_twin.optimizer import GreedyOptimizer, balanced_objective, throughput_objective
from digital_twin.prediction import CascadePredictor


def _config_from_args(args) -> WarehouseConfig:
    return WarehouseConfig(
        num_robots=args.robots,
        num_stations=args.stations,
        seed=args.seed,
    )


def cmd_simulate(args) -> dict:
    twin = DigitalTwin(_config_from_args(args))
    twin.run_simulation(args.duration, num_orders=args.orders)
    return {"summary": twin.summary()}


def cmd_chaos(args) -> dict:
    twin = DigitalTwin(_config_from_args(args))
    injector = ChaosInjector(twin)
    scenario_fn = getattr(ChaosScenario, args.scenario)
    scenario = scenario_fn(twin)
    result = injector.run_chaos_experiment(scenario, args.duration, num_orders=args.orders)
    return {
        "scenario": args.scenario,
        "impact": result["impact"],
        "num_incidents": len(scenario),
    }


def cmd_analyze(args) -> dict:
    twin = DigitalTwin(_config_from_args(args))
    metrics = twin.run_simulation(args.duration, num_orders=args.orders)

    engine = CausalInferenceEngine()
    graph = engine.learn_from_twin(metrics)
    edges = [
        {"cause": u, "effect": v, "strength": round(d.get("weight", 0.0), 3)}
        for u, v, d in graph.edges(data=True)
    ]

    queue_series = [m["queue_length"] for m in metrics["robot_utilization"]]
    predictor = CascadePredictor(higher_is_worse=True)
    prediction = predictor.predict(queue_series, horizon=args.horizon)

    return {
        "summary": twin.summary(),
        "causal_edges": edges,
        "root_causes": engine.rank_root_causes(),
        "cascade_prediction": {
            "cascade_risk": round(prediction.cascade_risk, 3),
            "trend": round(prediction.trend, 4),
            "explanation": prediction.explanation,
        },
    }


def cmd_optimize(args) -> dict:
    objective = balanced_objective if args.objective == "balanced" else throughput_objective
    optimizer = GreedyOptimizer(
        _config_from_args(args),
        objective=objective,
        duration=args.duration,
        num_orders=args.orders,
    )
    recs = optimizer.recommend(budget=args.budget)
    best = recs[0] if recs else None
    return {
        "objective": args.objective,
        "budget": args.budget,
        "best": None if best is None else {
            "lever": best.lever, "delta": round(best.delta, 2), "cost": best.cost,
        },
        "ranking": [
            {"lever": r.lever, "delta": round(r.delta, 2), "cost": r.cost}
            for r in recs
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    # Common options live on a parent parser so they work *after* the subcommand
    # (e.g. ``analyze --robots 20``) as well as before it.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--robots", type=int, default=30)
    common.add_argument("--stations", type=int, default=10)
    common.add_argument("--duration", type=int, default=3600, help="simulated seconds")
    common.add_argument("--orders", type=int, default=1000)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--horizon", type=int, default=10)

    parser = argparse.ArgumentParser(
        prog="digital_twin",
        description="Digital Twin Fulfillment Center — predictive chaos-engineering platform.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("simulate", parents=[common],
                   help="run a plain simulation").set_defaults(func=cmd_simulate)

    chaos = sub.add_parser("chaos", parents=[common], help="run a chaos experiment")
    chaos.add_argument("--scenario", default="robot_plague",
                       choices=["robot_plague", "black_friday", "network_partition"])
    chaos.set_defaults(func=cmd_chaos)

    sub.add_parser("analyze", parents=[common],
                   help="learn causal graph + forecast cascade risk").set_defaults(
        func=cmd_analyze
    )

    optimize = sub.add_parser("optimize", parents=[common],
                              help="recommend the best intervention within a budget")
    optimize.add_argument("--budget", type=float, default=float("inf"))
    optimize.add_argument("--objective", default="throughput",
                          choices=["throughput", "balanced"])
    optimize.set_defaults(func=cmd_optimize)
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = args.func(args)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
