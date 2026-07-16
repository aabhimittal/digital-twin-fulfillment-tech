"""Brain 2 — the Causal Graph (understanding).

We do not just want to know *what* happens, we want to know *why*.  This module
learns a causal DAG over the warehouse telemetry using a constraint-based approach:

1. Start from a fully connected skeleton.
2. Remove edges between variables that are **conditionally independent** given the
   rest (tested via partial correlation).
3. Orient the surviving edges using domain-knowledge temporal precedence.

The core answer we want is the interventional distribution ``P(Y | do(X))`` — what
happens when we *force* X to change — which the :class:`CausalInferenceEngine`
approximates by propagating an intervention along the discovered graph.

The optional :class:`CausalGraphNetwork` (a Graph Attention Network) refines edge
strengths when PyTorch + PyTorch-Geometric are installed; it degrades gracefully to
a no-op when they are not, so the package has **no hard deep-learning dependency**.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


# Domain knowledge: which variables temporally precede which.  Used to orient edges.
PRECEDENCE_RULES: Dict[str, List[str]] = {
    "num_failures": ["queue_length", "throughput", "total_picks", "utilization"],
    "avg_battery": ["utilization", "total_picks"],
    "queue_length": ["throughput", "total_picks"],
    "utilization": ["total_picks", "throughput"],
}


def _comes_before(var_i: str, var_j: str) -> bool:
    """Return True if ``var_i`` is believed to causally precede ``var_j``."""
    for cause, effects in PRECEDENCE_RULES.items():
        if cause in var_i and any(effect in var_j for effect in effects):
            return True
        if cause in var_j and any(effect in var_i for effect in effects):
            return False
    return var_i < var_j                       # weak alphabetical fallback


def _test_conditional_independence(data: np.ndarray, i: int, j: int,
                                   conditioning_set: Sequence[int], alpha: float) -> bool:
    """Partial-correlation conditional-independence test.

    ``H0: X ⟂ Y | Z``.  Returns True when we *fail to reject* H0 (i.e. independent).
    """
    n = len(data)
    if not conditioning_set:
        corr = np.corrcoef(data[:, i], data[:, j])[0, 1]
    else:
        z = data[:, list(conditioning_set)]
        res_i = data[:, i] - LinearRegression().fit(z, data[:, i]).predict(z)
        res_j = data[:, j] - LinearRegression().fit(z, data[:, j]).predict(z)
        corr = np.corrcoef(res_i, res_j)[0, 1]

    if np.isnan(corr):
        return True                            # constant column ⇒ treat as independent
    corr = float(np.clip(corr, -0.999999, 0.999999))
    k = len(conditioning_set)
    dof = n - k - 2
    if dof <= 0:
        return True
    t_stat = corr * np.sqrt(dof) / np.sqrt(1 - corr ** 2)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), dof))
    return p_value > alpha


def discover_causal_structure(observations: List[Dict[str, float]],
                              alpha: float = 0.05) -> nx.DiGraph:
    """Learn a causal graph from observational rows using partial correlation.

    Parameters
    ----------
    observations:
        A list of dictionaries mapping variable name -> value (one row per timestep).
    alpha:
        Significance level for the independence test.
    """
    graph = nx.DiGraph()
    if not observations:
        return graph

    variables = list(observations[0].keys())
    for idx, var in enumerate(variables):
        graph.add_node(var, index=idx)

    data = np.array([[float(row[v]) for v in variables] for row in observations],
                    dtype=float)
    if len(data) < 5:                          # not enough samples to test anything
        return graph

    correlations = np.corrcoef(data.T)
    num_vars = len(variables)
    for i in range(num_vars):
        for j in range(i + 1, num_vars):
            conditioning = [k for k in range(num_vars) if k not in (i, j)]
            independent = _test_conditional_independence(data, i, j, conditioning, alpha)
            if independent:
                continue
            # ``weight`` is the (unsigned) strength for ranking; ``effect`` keeps the
            # sign so interventions can tell "more X ⇒ less Y" from "⇒ more Y".
            signed = float(correlations[i, j])
            weight = abs(signed)
            if _comes_before(variables[i], variables[j]):
                graph.add_edge(variables[i], variables[j], weight=weight, effect=signed)
            else:
                graph.add_edge(variables[j], variables[i], weight=weight, effect=signed)
    return graph


class CausalInferenceEngine:
    """Uses a learned causal graph to predict interventions — ``P(Y | do(X))``."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.causal_graph: Optional[nx.DiGraph] = None
        self.observations: List[Dict[str, float]] = []

    # -- learning -------------------------------------------------------- #
    @staticmethod
    def observations_from_metrics(metrics: Dict[str, List[dict]]) -> List[Dict[str, float]]:
        """Flatten a twin's ``robot_utilization`` metric stream into observation rows."""
        rows: List[Dict[str, float]] = []
        for m in metrics.get("robot_utilization", []):
            rows.append({
                "utilization": float(m.get("utilization", 0.0)),
                "total_picks": float(m.get("total_picks", 0.0)),
                "avg_battery": float(m.get("avg_battery", 0.0)),
                "num_failures": float(m.get("num_failures", 0.0)),
                "queue_length": float(m.get("queue_length", 0.0)),
            })
        return rows

    def learn(self, observations: List[Dict[str, float]]) -> nx.DiGraph:
        """Discover the causal structure from a batch of observations."""
        self.observations = observations
        self.causal_graph = discover_causal_structure(observations, self.alpha)
        return self.causal_graph

    def learn_from_twin(self, metrics: Dict[str, List[dict]]) -> nx.DiGraph:
        """Convenience wrapper: learn directly from a twin's metrics dict."""
        return self.learn(self.observations_from_metrics(metrics))

    # -- inference ------------------------------------------------------- #
    def predict_intervention(self, intervention: Dict[str, float]) -> Dict[str, float]:
        """Estimate downstream KPI changes if we force variables to change.

        The effect on a descendant is the product of edge weights along the
        strongest path from the intervened variable, scaled by the intervention
        magnitude — a first-order structural-causal-model approximation of
        ``P(Y | do(X))``.
        """
        if self.causal_graph is None:
            raise RuntimeError("Call learn(...) before predict_intervention(...).")

        effects: Dict[str, float] = {}
        for var, magnitude in intervention.items():
            if var not in self.causal_graph:
                continue
            for target in nx.descendants(self.causal_graph, var):
                try:
                    path = nx.shortest_path(self.causal_graph, var, target)
                except nx.NetworkXNoPath:
                    continue
                gain = 1.0
                for a, b in zip(path, path[1:]):
                    edge = self.causal_graph[a][b]
                    # Prefer the signed effect; fall back to unsigned weight.
                    gain *= edge.get("effect", edge.get("weight", 0.0))
                effects[target] = effects.get(target, 0.0) + magnitude * gain

        confidence = 0.0
        if self.observations:
            confidence = min(1.0, len(self.observations) / 100.0)
        effects["confidence"] = confidence
        return effects

    def rank_root_causes(self) -> List[str]:
        """Return variables ordered by out-degree weight — the biggest 'drivers'."""
        if self.causal_graph is None:
            return []
        scores = {
            node: sum(d.get("weight", 0.0) for _, _, d in self.causal_graph.out_edges(node, data=True))
            for node in self.causal_graph.nodes
        }
        return sorted(scores, key=scores.get, reverse=True)


# --------------------------------------------------------------------------- #
# Optional Graph-Attention-Network refinement (only if torch is installed)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when torch is available
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalGraphNetwork(nn.Module):
        """Graph Attention Network that refines causal-edge strengths.

        Optional. Learns node embeddings whose pairwise MLP score estimates the
        magnitude of a causal effect between two variables.
        """

        def __init__(self, node_features: int, hidden_dim: int = 64):
            super().__init__()
            self.enc1 = nn.Linear(node_features, hidden_dim)
            self.enc2 = nn.Linear(hidden_dim, hidden_dim)
            self.causal_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor"):
            h = F.elu(self.enc1(x))
            h = F.elu(self.enc2(h))
            num_nodes = x.size(0)
            effects = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        effects.append(self.causal_mlp(torch.cat([h[i], h[j]], dim=-1)))
            return h, (torch.stack(effects) if effects else torch.empty(0))

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    CausalGraphNetwork = None  # type: ignore
    _TORCH_AVAILABLE = False


def torch_available() -> bool:
    """True when the optional PyTorch-based GNN refinement is available."""
    return _TORCH_AVAILABLE
