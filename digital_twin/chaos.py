"""Chaos-engineering incident injector.

This module lets you *intentionally break things* inside the digital twin and learn
from the failures.  The :class:`ChaosInjector` does not break things randomly — it
uses an Upper-Confidence-Bound (UCB) multi-armed bandit to spend its "break budget"
on the incident types that historically cause the most damage, so it converges on
the warehouse's most interesting failure modes.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from digital_twin.core import DigitalTwin, Robot


class IncidentType(Enum):
    """Types of incidents we can inject."""

    ROBOT_FAILURE = "robot_failure"
    BATTERY_DRAIN = "battery_drain"
    NETWORK_PARTITION = "network_partition"
    STATION_OVERLOAD = "station_overload"
    SENSOR_MALFUNCTION = "sensor_malfunction"
    SOFTWARE_BUG = "software_bug"
    HUMAN_ERROR = "human_error"
    ENVIRONMENTAL = "environmental"      # fire alarm, power outage, etc.


@dataclass
class Incident:
    """A single chaotic event."""

    type: IncidentType
    severity: float                      # 0.0 to 1.0
    start_time: float
    duration: float
    affected_entities: List[int] = field(default_factory=list)
    cascading_probability: float = 0.0   # chance this triggers other incidents


class ChaosScenario:
    """Predefined chaos scenarios — "stress tests" for your warehouse."""

    @staticmethod
    def black_friday(twin: DigitalTwin) -> List[Incident]:
        """Simulate extreme, sustained load across every station."""
        return [
            Incident(
                type=IncidentType.STATION_OVERLOAD,
                severity=0.9,
                start_time=0,
                duration=3600,
                affected_entities=list(range(len(twin.stations))),
                cascading_probability=0.7,
            )
        ]

    @staticmethod
    def robot_plague(twin: DigitalTwin, fraction: float = 0.3,
                     rng: Optional[np.random.Generator] = None) -> List[Incident]:
        """A software bug knocks out a fraction of the robot fleet."""
        rng = rng if rng is not None else np.random.default_rng()
        num_affected = max(1, int(len(twin.robots) * fraction))
        affected_ids = list(rng.choice(len(twin.robots), size=num_affected, replace=False))
        return [
            Incident(
                type=IncidentType.ROBOT_FAILURE,
                severity=1.0,
                start_time=float(rng.uniform(0, 1800)),
                duration=float(rng.uniform(300, 1800)),
                affected_entities=[int(robot_id)],
                cascading_probability=0.5,
            )
            for robot_id in affected_ids
        ]

    @staticmethod
    def network_partition(twin: DigitalTwin) -> List[Incident]:
        """Half the fleet loses the ability to communicate."""
        partition_point = len(twin.robots) // 2
        return [
            Incident(
                type=IncidentType.NETWORK_PARTITION,
                severity=0.8,
                start_time=900,
                duration=600,
                affected_entities=list(range(partition_point, len(twin.robots))),
                cascading_probability=0.6,
            )
        ]


class ChaosInjector:
    """Intelligent chaos-injection system.

    Uses a UCB multi-armed bandit to decide *which* incident type to explore next::

        UCB(a) = mean_reward(a) + c * sqrt(ln(N) / N(a))

    where the "reward" of an incident is the throughput loss it caused — so the
    bandit is rewarded for finding the failures that hurt the most.
    """

    def __init__(self, twin: DigitalTwin, exploration_c: float = 2.0,
                 rng: Optional[np.random.Generator] = None):
        self.twin = twin
        self.exploration_c = exploration_c
        self.rng = rng if rng is not None else np.random.default_rng(twin.config.seed)
        self.incident_history: List[Tuple[Incident, dict]] = []
        self.incident_rewards: Dict[IncidentType, List[float]] = defaultdict(list)
        self.epsilon = 0.3

    # -- bandit selection ------------------------------------------------ #
    def select_incident(self) -> IncidentType:
        """Pick the next incident type to inject using UCB (with epsilon exploration)."""
        total_trials = sum(len(r) for r in self.incident_rewards.values())
        if total_trials == 0 or self.rng.random() < self.epsilon:
            return list(IncidentType)[int(self.rng.integers(len(IncidentType)))]

        ucb_values: Dict[IncidentType, float] = {}
        for incident_type in IncidentType:
            rewards = self.incident_rewards[incident_type]
            if not rewards:
                return incident_type            # try untested arms first
            avg_reward = float(np.mean(rewards))
            n_trials = len(rewards)
            bonus = self.exploration_c * np.sqrt(np.log(total_trials) / n_trials)
            ucb_values[incident_type] = avg_reward + bonus
        return max(ucb_values, key=ucb_values.get)

    # -- injection ------------------------------------------------------- #
    def inject_incident(self, incident: Incident):
        """Modify the twin's state to realise a failure."""
        if incident.type == IncidentType.ROBOT_FAILURE:
            for robot_id in incident.affected_entities:
                robot = self.twin.robots[robot_id]
                self.twin.env.process(self._cause_robot_failure(robot, incident))

        elif incident.type == IncidentType.BATTERY_DRAIN:
            for robot_id in incident.affected_entities:
                robot = self.twin.robots[robot_id]
                robot.degradation_rate *= (1 + incident.severity)
                robot.battery = max(0.0, robot.battery * (1 - incident.severity))

        elif incident.type == IncidentType.STATION_OVERLOAD:
            for station_id in incident.affected_entities:
                station = self.twin.stations[station_id]
                station.avg_processing_time *= (1 + incident.severity * 2)

        elif incident.type in (IncidentType.SENSOR_MALFUNCTION, IncidentType.SOFTWARE_BUG):
            for robot_id in incident.affected_entities:
                robot = self.twin.robots[robot_id]
                robot.failure_rate = min(1.0, robot.failure_rate * (1 + incident.severity * 50))

        elif incident.type == IncidentType.NETWORK_PARTITION:
            # Partitioned robots stop exploring and freeze their policy.
            for robot_id in incident.affected_entities:
                robot = self.twin.robots[robot_id]
                robot.epsilon = 0.0
        # HUMAN_ERROR / ENVIRONMENTAL are modelled as station slow-downs.
        elif incident.type in (IncidentType.HUMAN_ERROR, IncidentType.ENVIRONMENTAL):
            for station in self.twin.stations:
                station.avg_processing_time *= (1 + incident.severity)

    def _cause_robot_failure(self, robot: Robot, incident: Incident):
        """SimPy coroutine that fails a robot for ``incident.duration`` seconds."""
        yield self.twin.env.timeout(incident.start_time)
        initial_throughput = sum(r.total_picks for r in self.twin.robots)
        robot.status = "failed"
        robot.failure_count += 1

        yield self.twin.env.timeout(incident.duration)
        robot.status = "idle"

        final_throughput = sum(r.total_picks for r in self.twin.robots)
        impact = final_throughput - initial_throughput
        reward = 1.0 - impact / max(1, initial_throughput)   # less progress ⇒ higher reward
        self.incident_rewards[incident.type].append(reward)
        self.incident_history.append((incident, {
            "impact": impact,
            "reward": reward,
            "affected_orders": len(self.twin.task_queue.items),
        }))

    # -- experiments ----------------------------------------------------- #
    def run_chaos_experiment(self, scenario: List[Incident], duration: int,
                             num_orders: int = 1000) -> Dict[str, object]:
        """Run baseline vs. chaos and return an impact analysis.

        A *fresh* baseline twin is built from the same config so the comparison is
        apples-to-apples; the chaos run injects ``scenario`` into ``self.twin``.
        """
        baseline_twin = DigitalTwin(self.twin.config)
        baseline_metrics = baseline_twin.run_simulation(duration, num_orders=num_orders)

        for incident in scenario:
            self.inject_incident(incident)
        chaos_metrics = self.twin.run_simulation(duration, num_orders=num_orders)

        return {
            "baseline": baseline_metrics,
            "chaos": chaos_metrics,
            "impact": self._analyze_impact(baseline_metrics, chaos_metrics),
            "incidents": scenario,
        }

    def _analyze_impact(self, baseline: Dict, chaos: Dict) -> Dict[str, object]:
        base_util = baseline["robot_utilization"]
        chaos_util = chaos["robot_utilization"]
        base_tp = base_util[-1]["total_picks"] if base_util else 0
        chaos_tp = chaos_util[-1]["total_picks"] if chaos_util else 0
        degradation = (base_tp - chaos_tp) / base_tp if base_tp else 0.0
        return {
            "baseline_throughput": base_tp,
            "chaos_throughput": chaos_tp,
            "throughput_degradation": degradation,
            "recovery_time": self._calculate_recovery_time(chaos),
            "cascade_detected": self._detect_cascading_failure(),
        }

    @staticmethod
    def _calculate_recovery_time(metrics: Dict) -> float:
        """Minutes until utilisation returns to 90% of its early-normal level."""
        util = [m["utilization"] for m in metrics["robot_utilization"]]
        if len(util) <= 10:
            return float(len(util))
        normal_util = float(np.mean(util[:10]))
        threshold = normal_util * 0.9
        for i, u in enumerate(util):
            if i > 10 and u >= threshold:
                return float(i)
        return float(len(util))               # never recovered

    def _detect_cascading_failure(self) -> bool:
        """Heuristic: more than one recorded incident implies a cascade."""
        return len(self.incident_history) > 1
