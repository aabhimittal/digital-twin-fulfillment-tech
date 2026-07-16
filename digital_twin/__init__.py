"""Digital Twin of a Fulfillment Center — a predictive chaos-engineering platform.

The package is organised around the "three brains" described in the design:

* :mod:`digital_twin.core` — the **Physics Engine** (a discrete-event SimPy model
  of robots, picking stations and orders that produces the ground-truth telemetry).
* :mod:`digital_twin.causal` — the **Causal Graph** (learns "X causes Y" relations
  from the telemetry using partial-correlation constraint-based discovery, with an
  optional Graph-Attention-Network refinement when PyTorch is installed).
* :mod:`digital_twin.prediction` — the **Prediction Engine** (forecasts how a KPI
  time-series evolves and estimates cascading, second-order effects).

The :mod:`digital_twin.chaos` module sits on top of the physics engine and injects
incidents to discover the most damaging failure modes.
"""

from digital_twin.core import (
    DigitalTwin,
    Item,
    Order,
    PickingStation,
    Robot,
    WarehouseConfig,
)
from digital_twin.chaos import (
    ChaosInjector,
    ChaosScenario,
    Incident,
    IncidentType,
)
from digital_twin.causal import CausalInferenceEngine, discover_causal_structure
from digital_twin.prediction import CascadePredictor, PredictionResult
from digital_twin.optimizer import (
    BanditOptimizer,
    GreedyOptimizer,
    Lever,
    Recommendation,
    default_levers,
)

__version__ = "0.2.0"

__all__ = [
    "DigitalTwin",
    "WarehouseConfig",
    "Robot",
    "PickingStation",
    "Item",
    "Order",
    "ChaosInjector",
    "ChaosScenario",
    "Incident",
    "IncidentType",
    "CausalInferenceEngine",
    "discover_causal_structure",
    "CascadePredictor",
    "PredictionResult",
    "GreedyOptimizer",
    "BanditOptimizer",
    "Lever",
    "Recommendation",
    "default_levers",
    "__version__",
]
