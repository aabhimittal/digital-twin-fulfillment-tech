"""Tests for the prediction engine (Brain 3)."""

from digital_twin.prediction import CascadePredictor, holt_forecast


def test_holt_forecast_length_and_trend():
    forecast, level, trend = holt_forecast([1, 2, 3, 4, 5], horizon=3)
    assert len(forecast) == 3
    assert trend > 0
    assert forecast[0] > 5           # continues the upward trend


def test_rising_series_is_high_risk():
    rising = list(range(0, 40, 2))
    predictor = CascadePredictor(higher_is_worse=True)
    result = predictor.predict(rising, horizon=10)
    assert result.cascade_risk > 0.5
    assert result.trend > 0
    assert "risk" in result.explanation.lower()


def test_flat_series_is_low_risk():
    flat = [5.0] * 30
    predictor = CascadePredictor(higher_is_worse=True)
    result = predictor.predict(flat, horizon=10)
    assert result.cascade_risk < 0.5


def test_threshold_proximity_raises_risk():
    series = [float(x) for x in range(0, 20)]
    near = CascadePredictor(critical_threshold=25, higher_is_worse=True).predict(series)
    far = CascadePredictor(critical_threshold=1000, higher_is_worse=True).predict(series)
    assert near.cascade_risk > far.cascade_risk


def test_empty_series_is_safe():
    result = CascadePredictor().predict([], horizon=5)
    assert result.cascade_risk == 0.0
    assert len(result.forecast) == 5
