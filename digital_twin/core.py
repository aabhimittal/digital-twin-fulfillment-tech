"""Brain 1 — the Physics Engine (ground truth).

A discrete-event simulation of a fulfillment center built on `SimPy`.  The model
captures the parts of warehouse "physics" that matter for operational decisions:

* Autonomous Mobile Robots (AMRs) that travel, pick, deliver, drain their battery
  and occasionally fail.
* Human/automated picking stations that assemble orders with variable service time.
* A Poisson order-arrival process feeding a shared task queue.

Everything is seedable through :class:`WarehouseConfig.seed` so that simulations are
reproducible — a hard requirement for a *digital twin* whose whole value is being
able to replay and compare scenarios.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import simpy

Location = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Configuration & value objects
# --------------------------------------------------------------------------- #
@dataclass
class WarehouseConfig:
    """Configuration for the fulfillment center."""

    num_robots: int = 50
    num_stations: int = 20
    num_zones: int = 5
    items_per_zone: int = 10_000
    robots_per_station: int = 3
    avg_pick_time: float = 2.0          # seconds
    avg_travel_time: float = 0.2        # seconds per distance unit
    shift_duration: int = 28_800        # 8 hours in seconds
    order_arrival_rate: float = 10.0    # orders per minute (Poisson lambda)
    low_battery_threshold: float = 20.0
    base_failure_rate: float = 0.001    # probability of failure per operation
    repair_time: float = 300.0          # seconds to recover from a failure
    seed: Optional[int] = 42

    def rng(self) -> np.random.Generator:
        """Return a NumPy random generator seeded from this config."""
        return np.random.default_rng(self.seed)


@dataclass
class Item:
    """Individual inventory item."""

    id: str
    zone: int
    bin_location: Location            # (aisle, rack, shelf)
    weight: float                     # kg
    priority: int                     # 1-5, 5 being highest
    temperature_sensitive: bool = False
    fragile: bool = False


@dataclass
class Order:
    """Customer order."""

    id: str
    items: List[Item]
    timestamp: float
    priority: int
    deadline: float


# --------------------------------------------------------------------------- #
# Robot agent
# --------------------------------------------------------------------------- #
class Robot:
    """Autonomous Mobile Robot (AMR) agent.

    Think of this as a chess piece that has limited energy (battery), can only be in
    one place at a time, has probabilistic failures, and learns from experience via a
    tabular Q-learning policy for path selection.
    """

    def __init__(self, id: int, env: simpy.Environment, config: WarehouseConfig,
                 rng: Optional[np.random.Generator] = None):
        self.id = id
        self.env = env
        self.config = config
        self.rng = rng if rng is not None else config.rng()

        self.battery = 100.0                 # percentage
        self.location: Location = (0, 0, 0)
        self.carrying: Optional[str] = None
        self.status = "idle"                 # idle, traveling, picking, charging, failed
        self.total_distance = 0.0
        self.total_picks = 0
        self.failure_count = 0

        # Picking stations this robot can deliver to (wired up by the twin).
        self.stations: List["PickingStation"] = []

        # Probabilistic failure model.
        self.failure_rate = config.base_failure_rate
        self.degradation_rate = 0.01

        # Learning component — Q-values for path selection.
        self.q_table: Dict[str, Dict[Tuple, float]] = defaultdict(lambda: defaultdict(float))
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1

    # -- geometry -------------------------------------------------------- #
    @staticmethod
    def calculate_distance(start: Location, end: Location) -> float:
        """Manhattan distance in 3D warehouse space."""
        return float(sum(abs(a - b) for a, b in zip(start, end)))

    # -- Q-learning path selection --------------------------------------- #
    def _encode_state(self) -> str:
        """Encode the current state for Q-learning."""
        return f"{self.location}_{self.battery:.0f}_{self.status}"

    def select_path(self, available_paths: List[Tuple]) -> Tuple:
        """Q-Learning based path selection.

        ``Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]``
        with epsilon-greedy exploration.
        """
        state = self._encode_state()
        if self.rng.random() < self.epsilon:
            idx = int(self.rng.integers(len(available_paths)))
            return available_paths[idx]
        q_values = [self.q_table[state][path] for path in available_paths]
        return available_paths[int(np.argmax(q_values))]

    def update_q_value(self, state: str, action: Tuple, reward: float, next_state: str):
        """Update the Q-table from a single (s, a, r, s') transition."""
        current_q = self.q_table[state][action]
        next_values = self.q_table[next_state].values()
        max_next_q = max(next_values) if next_values else 0.0
        self.q_table[state][action] = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

    # -- main process ---------------------------------------------------- #
    def process(self, task_queue: simpy.Store):
        """Main robot life-cycle: wait for a task, travel, pick, deliver, charge."""
        while True:
            try:
                task = yield task_queue.get()
                self.status = "traveling"

                start_location = self.location
                target_location = task["pickup_location"]
                distance = self.calculate_distance(start_location, target_location)

                # Energy model: E = k*d + lift term for vertical movement.
                energy_consumed = 0.1 * distance + 0.5 * max(
                    0, target_location[2] - start_location[2]
                )
                self.battery = max(0.0, self.battery - energy_consumed)

                travel_time = (
                    distance
                    * self.config.avg_travel_time
                    * task.get("congestion_factor", 1.0)
                )
                yield self.env.timeout(travel_time)
                self.location = target_location
                self.total_distance += distance

                # Chaos: probabilistic in-flight failure.
                if self.rng.random() < self.failure_rate:
                    self.status = "failed"
                    self.failure_count += 1
                    yield self.env.timeout(self.config.repair_time)
                    self.status = "idle"
                    continue

                # Pick operation.
                self.status = "picking"
                pick_time = max(0.1, self.rng.normal(self.config.avg_pick_time, 0.5))
                yield self.env.timeout(pick_time)
                self.carrying = task["item_id"]
                self.total_picks += 1

                # Deliver to station.
                self.status = "traveling"
                station_location = task["station_location"]
                distance = self.calculate_distance(self.location, station_location)
                yield self.env.timeout(distance * self.config.avg_travel_time)
                self.location = station_location

                # Hand the picked item to a station for order assembly.
                if self.stations:
                    station = self.stations[self.id % len(self.stations)]
                    yield station.queue.put([self.carrying])
                self.carrying = None
                self.status = "idle"

                # Charge if the battery is low.
                if self.battery < self.config.low_battery_threshold:
                    yield self.env.process(self.charge())

            except simpy.Interrupt:
                # Chaos-engineering interrupt handling.
                self.status = "interrupted"
                yield self.env.timeout(60)
                self.status = "idle"

    def charge(self):
        """Charging process with a realistic lithium-ion curve (fast to 80%, slow after)."""
        self.status = "charging"
        while self.battery < 95:
            charge_rate = 2.0 if self.battery < 80 else 0.5
            yield self.env.timeout(1)
            self.battery = min(100.0, self.battery + charge_rate)
        self.status = "idle"


# --------------------------------------------------------------------------- #
# Picking station
# --------------------------------------------------------------------------- #
class PickingStation:
    """Human-operated or automated packing station where items become orders."""

    def __init__(self, id: int, env: simpy.Environment, capacity: int = 3,
                 rng: Optional[np.random.Generator] = None):
        self.id = id
        self.env = env
        self.capacity = capacity
        self.queue: simpy.Store = simpy.Store(env)
        self.processed_orders = 0
        self.avg_processing_time = 30.0     # seconds per item
        self.rng = rng if rng is not None else np.random.default_rng()

    def process(self):
        """Continuously process incoming items."""
        while True:
            items = yield self.queue.get()
            n = len(items) if isinstance(items, (list, tuple)) else 1
            processing_time = self.avg_processing_time * n
            processing_time *= self.rng.uniform(0.8, 1.2)
            yield self.env.timeout(processing_time)
            self.processed_orders += 1


# --------------------------------------------------------------------------- #
# The digital twin orchestrator
# --------------------------------------------------------------------------- #
class DigitalTwin:
    """The 'world' that contains all entities and manages their interactions."""

    def __init__(self, config: Optional[WarehouseConfig] = None):
        self.config = config if config is not None else WarehouseConfig()
        self.env = simpy.Environment()
        self.rng = self.config.rng()

        self.robots: List[Robot] = []
        self.stations: List[PickingStation] = []
        self.task_queue: simpy.Store = simpy.Store(self.env)

        self.metrics: Dict[str, List[dict]] = {
            "robot_utilization": [],
            "throughput": [],
            "queue_lengths": [],
            "incidents": [],
            "energy_consumption": [],
        }

        self._setup_entities()

    # -- setup ----------------------------------------------------------- #
    def _setup_entities(self):
        # Stations first so robots can be given a reference to deliver into them.
        for i in range(self.config.num_stations):
            station = PickingStation(
                i, self.env,
                rng=np.random.default_rng(self.config.seed + 1000 + i
                                          if self.config.seed is not None else None),
            )
            self.stations.append(station)
            self.env.process(station.process())

        for i in range(self.config.num_robots):
            robot = Robot(i, self.env, self.config,
                          rng=np.random.default_rng(self.config.seed + i
                                                    if self.config.seed is not None else None))
            robot.stations = self.stations
            self.robots.append(robot)
            self.env.process(robot.process(self.task_queue))

    # -- workload -------------------------------------------------------- #
    def generate_orders(self, num_orders: int):
        """Generate a realistic order workload via a Poisson arrival process."""
        for i in range(num_orders):
            inter_arrival = self.rng.exponential(60.0 / self.config.order_arrival_rate)
            yield self.env.timeout(inter_arrival)

            num_items = int(self.rng.poisson(3)) + 1
            priority = int(self.rng.choice([1, 2, 3, 4, 5],
                                           p=[0.1, 0.2, 0.4, 0.2, 0.1]))
            station_location: Location = (50, 25, 0)

            for item_idx in range(num_items):
                pickup_location: Location = (
                    int(self.rng.integers(0, 100)),
                    int(self.rng.integers(0, 50)),
                    int(self.rng.integers(0, 10)),
                )
                task = {
                    "order_id": f"order_{i}",
                    "item_id": f"item_{i}_{item_idx}",
                    "pickup_location": pickup_location,
                    "station_location": station_location,
                    "priority": priority,
                    "timestamp": self.env.now,
                }
                yield self.task_queue.put(task)

    # -- metrics --------------------------------------------------------- #
    def collect_metrics(self, interval: int = 60):
        """Collect system metrics every ``interval`` simulated seconds."""
        while True:
            yield self.env.timeout(interval)
            active_robots = sum(1 for r in self.robots if r.status != "idle")
            utilization = active_robots / max(1, len(self.robots))
            total_picks = sum(r.total_picks for r in self.robots)
            avg_battery = float(np.mean([r.battery for r in self.robots]))
            num_failures = sum(r.failure_count for r in self.robots)
            queue_length = len(self.task_queue.items)

            self.metrics["robot_utilization"].append({
                "time": self.env.now,
                "utilization": utilization,
                "total_picks": total_picks,
                "avg_battery": avg_battery,
                "num_failures": num_failures,
                "queue_length": queue_length,
            })

    # -- run ------------------------------------------------------------- #
    def run_simulation(self, duration: int, num_orders: int = 1000,
                       metric_interval: int = 60) -> Dict[str, List[dict]]:
        """Run the simulation for ``duration`` simulated seconds and return metrics."""
        self.env.process(self.generate_orders(num_orders))
        self.env.process(self.collect_metrics(metric_interval))
        self.env.run(until=duration)
        return self.metrics

    # -- convenience ----------------------------------------------------- #
    def summary(self) -> Dict[str, float]:
        """Return headline KPIs for the last run."""
        util = self.metrics["robot_utilization"]
        total_picks = util[-1]["total_picks"] if util else 0
        return {
            "total_picks": total_picks,
            "total_failures": sum(r.failure_count for r in self.robots),
            "avg_utilization": float(np.mean([m["utilization"] for m in util])) if util else 0.0,
            "processed_orders": sum(s.processed_orders for s in self.stations),
            "final_queue_length": len(self.task_queue.items),
        }
