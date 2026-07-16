"""Decision-Optimization Layer — recommend the best intervention.

The first three brains tell you *what* happens, *why*, and *what happens next*.
This layer closes the loop: given a set of candidate operational levers (add/remove
robots, add station capacity, throttle order intake, ...) it searches for the
intervention that maximises an objective while respecting a budget.

Two search strategies are provided, both seedable and dependency-light:

* :class:`GreedyOptimizer` — evaluates each candidate lever with the physics engine
  (a real counterfactual roll-out) and picks the best single move. Accurate but
  costs one simulation per candidate.
* :class:`BanditOptimizer` — an epsilon-greedy contextual bandit that reuses the
  learned causal graph to *predict* an intervention's effect cheaply, so it can
  rank a large lever space without re-simulating every option.

Both return a ranked list of :class:`Recommendation` objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from digital_twin.causal import CausalInferenceEngine
from digital_twin.core import DigitalTwin, WarehouseConfig


# --------------------------------------------------------------------------- #
# Levers & recommendations
# --------------------------------------------------------------------------- #
@dataclass
class Lever:
    """A single operational knob the optimizer may turn.

    ``apply`` mutates a :class:`WarehouseConfig` in place to realise the lever, and
    ``cost`` is charged against the optimizer's budget.
    """

    name: str
    apply: Callable[[WarehouseConfig], None]
    cost: float = 1.0
    # Which causal variable this lever pushes, and by how much — used by the
    # cheap bandit optimizer to predict effects without re-simulating.
    causal_target: Optional[str] = None
    causal_magnitude: float = 0.0


@dataclass
class Recommendation:
    """The optimizer's assessment of one lever."""

    lever: str
    objective: float
    delta: float                       # improvement vs. the do-nothing baseline
    cost: float
    detail: Dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def throughput_objective(twin: DigitalTwin) -> float:
    """Total picks completed — higher is better."""
    return float(twin.summary()["total_picks"])


def balanced_objective(twin: DigitalTwin) -> float:
    """Throughput penalised by leftover queue backlog."""
    s = twin.summary()
    return float(s["total_picks"]) - 5.0 * float(s["final_queue_length"])


# --------------------------------------------------------------------------- #
# Standard lever library
# --------------------------------------------------------------------------- #
def default_levers() -> List[Lever]:
    """A reasonable default set of operational levers."""

    def add_robots(n):
        def _apply(cfg: WarehouseConfig):
            cfg.num_robots = max(1, cfg.num_robots + n)
        return _apply

    def add_station_capacity(n):
        def _apply(cfg: WarehouseConfig):
            cfg.num_stations = max(1, cfg.num_stations + n)
        return _apply

    def throttle_orders(rate_delta):
        def _apply(cfg: WarehouseConfig):
            cfg.order_arrival_rate = max(1.0, cfg.order_arrival_rate + rate_delta)
        return _apply

    return [
        Lever("do_nothing", lambda cfg: None, cost=0.0),
        Lever("add_5_robots", add_robots(5), cost=5.0,
              causal_target="utilization", causal_magnitude=0.15),
        Lever("add_10_robots", add_robots(10), cost=10.0,
              causal_target="utilization", causal_magnitude=0.30),
        Lever("remove_5_robots", add_robots(-5), cost=1.0,
              causal_target="utilization", causal_magnitude=-0.15),
        Lever("add_3_stations", add_station_capacity(3), cost=3.0,
              causal_target="queue_length", causal_magnitude=-0.20),
        Lever("throttle_orders", throttle_orders(-3.0), cost=2.0,
              causal_target="queue_length", causal_magnitude=-0.25),
    ]


# --------------------------------------------------------------------------- #
# Greedy simulation-based optimizer
# --------------------------------------------------------------------------- #
class GreedyOptimizer:
    """Evaluate each lever with a real counterfactual simulation and rank them."""

    def __init__(self, base_config: WarehouseConfig,
                 objective: Callable[[DigitalTwin], float] = throughput_objective,
                 duration: int = 3600, num_orders: int = 1000):
        self.base_config = base_config
        self.objective = objective
        self.duration = duration
        self.num_orders = num_orders

    def _evaluate(self, lever: Lever) -> float:
        # Copy the config so levers never mutate the caller's baseline.
        cfg = WarehouseConfig(**vars(self.base_config))
        lever.apply(cfg)
        twin = DigitalTwin(cfg)
        twin.run_simulation(self.duration, num_orders=self.num_orders)
        return self.objective(twin)

    def recommend(self, levers: Optional[List[Lever]] = None,
                  budget: float = float("inf")) -> List[Recommendation]:
        """Return levers ranked by objective, filtered to those within ``budget``."""
        levers = levers if levers is not None else default_levers()
        baseline = self._evaluate(next(l for l in levers if l.name == "do_nothing")) \
            if any(l.name == "do_nothing" for l in levers) else self._evaluate(
                Lever("_baseline", lambda cfg: None))

        recs: List[Recommendation] = []
        for lever in levers:
            if lever.cost > budget:
                continue
            score = self._evaluate(lever)
            recs.append(Recommendation(
                lever=lever.name,
                objective=score,
                delta=score - baseline,
                cost=lever.cost,
                detail={"baseline": baseline},
            ))
        recs.sort(key=lambda r: r.delta, reverse=True)
        return recs


# --------------------------------------------------------------------------- #
# Cheap causal-bandit optimizer
# --------------------------------------------------------------------------- #
class BanditOptimizer:
    """Rank levers cheaply using the learned causal graph (no re-simulation).

    An epsilon-greedy bandit predicts each lever's effect via
    :meth:`CausalInferenceEngine.predict_intervention` and keeps an online estimate
    of realised rewards for levers it has actually tried.
    """

    def __init__(self, engine: CausalInferenceEngine,
                 objective_var: str = "utilization",
                 epsilon: float = 0.2, seed: Optional[int] = 42):
        if engine.causal_graph is None:
            raise RuntimeError("Pass a CausalInferenceEngine that has already learn()ed.")
        self.engine = engine
        self.objective_var = objective_var
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self._reward_estimates: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def _predicted_effect(self, lever: Lever) -> float:
        if not lever.causal_target:
            return 0.0
        effects = self.engine.predict_intervention(
            {lever.causal_target: lever.causal_magnitude}
        )
        # Reward is the predicted move in the objective variable.
        return float(effects.get(self.objective_var, 0.0))

    def recommend(self, levers: Optional[List[Lever]] = None,
                  budget: float = float("inf")) -> List[Recommendation]:
        """Rank levers by predicted (or learned) effect on the objective variable."""
        levers = levers if levers is not None else default_levers()
        recs: List[Recommendation] = []
        for lever in levers:
            if lever.cost > budget:
                continue
            predicted = self._predicted_effect(lever)
            learned = self._reward_estimates.get(lever.name)
            score = learned if learned is not None else predicted
            recs.append(Recommendation(
                lever=lever.name,
                objective=score,
                delta=score,
                cost=lever.cost,
                detail={"predicted_effect": predicted,
                        "tried": float(self._counts.get(lever.name, 0))},
            ))
        recs.sort(key=lambda r: r.delta, reverse=True)
        return recs

    def select(self, levers: Optional[List[Lever]] = None) -> Lever:
        """Epsilon-greedy pick of the next lever to actually try."""
        levers = levers if levers is not None else default_levers()
        if self.rng.random() < self.epsilon:
            return levers[int(self.rng.integers(len(levers)))]
        return max(levers, key=lambda l: (
            self._reward_estimates.get(l.name, self._predicted_effect(l))
        ))

    def update(self, lever: Lever, realised_reward: float) -> None:
        """Fold a realised reward into the online estimate for ``lever``."""
        n = self._counts.get(lever.name, 0) + 1
        prev = self._reward_estimates.get(lever.name, 0.0)
        self._reward_estimates[lever.name] = prev + (realised_reward - prev) / n
        self._counts[lever.name] = n
