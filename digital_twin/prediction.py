"""Brain 3 — the Prediction Engine (foresight).

Given the recent history of a KPI (e.g. queue length or utilisation), forecast how
it will evolve and estimate the risk of a *cascading* second-order failure.

The default forecaster is a dependency-light exponential-smoothing model with a
linear trend (Holt's method), so it runs anywhere NumPy is installed.  When PyTorch
is available an :class:`LSTMForecaster` can be used instead for sequence learning;
the public :class:`CascadePredictor` API is identical either way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class PredictionResult:
    """Output of a forecast."""

    forecast: List[float]
    cascade_risk: float                        # 0..1 probability of a cascade
    trend: float                               # slope of the fitted trend
    horizon: int
    explanation: str = ""
    contributing_factors: Dict[str, float] = field(default_factory=dict)


def holt_forecast(series: Sequence[float], horizon: int,
                  alpha: float = 0.5, beta: float = 0.3):
    """Holt's linear-trend exponential smoothing.

    Returns ``(forecast, level, trend)`` where ``forecast`` has length ``horizon``.
    """
    values = list(map(float, series))
    if not values:
        return [0.0] * horizon, 0.0, 0.0
    if len(values) == 1:
        return [values[0]] * horizon, values[0], 0.0

    level = values[0]
    trend = values[1] - values[0]
    for y in values[1:]:
        prev_level = level
        level = alpha * y + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend

    forecast = [level + (h + 1) * trend for h in range(horizon)]
    return forecast, level, trend


class CascadePredictor:
    """Forecast a KPI series and estimate cascading-failure risk.

    Cascade risk combines three signals:

    * **Trend** — a rising queue / falling utilisation trend raises risk.
    * **Volatility** — an unstable system is closer to tipping over.
    * **Proximity to a critical threshold** — how close the forecast gets to a level
      known to trigger secondary failures.
    """

    def __init__(self, critical_threshold: Optional[float] = None,
                 higher_is_worse: bool = True, use_lstm: bool = False):
        self.critical_threshold = critical_threshold
        self.higher_is_worse = higher_is_worse
        self.use_lstm = use_lstm and _TORCH_AVAILABLE
        self._lstm: Optional["LSTMForecaster"] = None

    # -- forecasting ----------------------------------------------------- #
    def predict(self, series: Sequence[float], horizon: int = 10) -> PredictionResult:
        """Forecast ``horizon`` steps ahead and estimate cascade risk."""
        values = list(map(float, series))
        if self.use_lstm and len(values) >= 12:
            forecast, trend = self._predict_lstm(values, horizon)
        else:
            forecast, _level, trend = holt_forecast(values, horizon)

        cascade_risk, factors = self._cascade_risk(values, forecast, trend)
        explanation = self._explain(trend, cascade_risk, factors)
        return PredictionResult(
            forecast=forecast,
            cascade_risk=cascade_risk,
            trend=trend,
            horizon=horizon,
            explanation=explanation,
            contributing_factors=factors,
        )

    def _predict_lstm(self, values, horizon):  # pragma: no cover - needs torch
        if self._lstm is None:
            self._lstm = LSTMForecaster()
            self._lstm.fit(values)
        forecast = self._lstm.forecast(values, horizon)
        trend = (forecast[-1] - forecast[0]) / max(1, horizon - 1) if len(forecast) > 1 else 0.0
        return forecast, trend

    # -- risk model ------------------------------------------------------ #
    def _cascade_risk(self, history: List[float], forecast: List[float], trend: float):
        if not history:
            return 0.0, {}

        scale = (np.std(history) + abs(np.mean(history))) or 1.0
        # 1. Directional trend signal.
        signed_trend = trend if self.higher_is_worse else -trend
        trend_signal = _sigmoid(signed_trend / scale * 3.0)

        # 2. Volatility signal.
        volatility = float(np.std(history)) / scale
        volatility_signal = _sigmoid((volatility - 0.3) * 4.0)

        # 3. Threshold-proximity signal.
        threshold_signal = 0.0
        if self.critical_threshold is not None and forecast:
            peak = max(forecast) if self.higher_is_worse else min(forecast)
            if self.higher_is_worse:
                threshold_signal = _sigmoid((peak - self.critical_threshold) / scale * 3.0)
            else:
                threshold_signal = _sigmoid((self.critical_threshold - peak) / scale * 3.0)

        factors = {
            "trend": round(float(trend_signal), 4),
            "volatility": round(float(volatility_signal), 4),
            "threshold_proximity": round(float(threshold_signal), 4),
        }
        if self.critical_threshold is not None:
            weights = (0.4, 0.25, 0.35)
        else:
            weights = (0.6, 0.4, 0.0)
        risk = (
            weights[0] * trend_signal
            + weights[1] * volatility_signal
            + weights[2] * threshold_signal
        )
        return float(np.clip(risk, 0.0, 1.0)), factors

    @staticmethod
    def _explain(trend: float, risk: float, factors: Dict[str, float]) -> str:
        direction = "rising" if trend > 0 else "falling" if trend < 0 else "flat"
        level = "HIGH" if risk > 0.66 else "MODERATE" if risk > 0.33 else "LOW"
        dominant = max(factors, key=factors.get) if factors else "n/a"
        return (
            f"{level} cascade risk ({risk:.0%}). Series is {direction} "
            f"(trend={trend:+.3f}); dominant driver: {dominant}."
        )


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# --------------------------------------------------------------------------- #
# Optional LSTM forecaster (only if torch is installed)
# --------------------------------------------------------------------------- #
try:  # pragma: no cover - exercised only when torch is available
    import torch
    import torch.nn as nn

    class LSTMForecaster(nn.Module):
        """Minimal LSTM that learns to predict the next value in a series."""

        def __init__(self, hidden_dim: int = 32, window: int = 10):
            super().__init__()
            self.window = window
            self.lstm = nn.LSTM(1, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, 1)
            self._mean = 0.0
            self._std = 1.0

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

        def fit(self, series, epochs: int = 60, lr: float = 0.01):
            values = np.asarray(series, dtype=float)
            self._mean, self._std = float(values.mean()), float(values.std() or 1.0)
            norm = (values - self._mean) / self._std
            if len(norm) <= self.window:
                return self
            xs, ys = [], []
            for i in range(len(norm) - self.window):
                xs.append(norm[i:i + self.window])
                ys.append(norm[i + self.window])
            x = torch.tensor(np.array(xs), dtype=torch.float32).unsqueeze(-1)
            y = torch.tensor(np.array(ys), dtype=torch.float32).unsqueeze(-1)
            opt = torch.optim.Adam(self.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            self.train()
            for _ in range(epochs):
                opt.zero_grad()
                loss = loss_fn(self(x), y)
                loss.backward()
                opt.step()
            return self

        @torch.no_grad()
        def forecast(self, series, horizon: int):
            self.eval()
            values = np.asarray(series, dtype=float)
            norm = list((values - self._mean) / self._std)
            preds = []
            for _ in range(horizon):
                window = norm[-self.window:]
                if len(window) < self.window:
                    window = [window[0]] * (self.window - len(window)) + window
                x = torch.tensor(window, dtype=torch.float32).reshape(1, self.window, 1)
                nxt = float(self(x).item())
                norm.append(nxt)
                preds.append(nxt * self._std + self._mean)
            return preds

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    LSTMForecaster = None  # type: ignore
    _TORCH_AVAILABLE = False


def torch_available() -> bool:
    """True when the optional PyTorch LSTM forecaster is available."""
    return _TORCH_AVAILABLE
