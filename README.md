# Digital Twin of a Fulfillment Center — Predictive Chaos-Engineering Platform

A digital twin that doesn't just *replay* known scenarios — it **learns the causal
"physics" of your warehouse** and **predicts unknown, second-order cascading
failures** before they hit production.

> **Analogy.** Imagine a chess engine that plays millions of games in parallel
> universes and then tells you *"if you move your knight here, there's a 73% chance
> your opponent sacrifices their queen in 8 moves."* This does that for warehouse
> operations: *"if you pull 5 robots for maintenance at 2pm, there's a high chance
> the pack stations back up and orders miss their cut-off by 4pm."*

It combines four ideas that are usually kept apart:

**Digital Twin** · **Chaos Engineering** · **Causal AI** · **Multi-Agent RL**

---

## The three brains

```
┌─────────────────────────────────────────────────────────────┐
│                     DIGITAL TWIN CORE                        │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐    │
│  │  Physics    │──▶│   Causal     │──▶│   Prediction    │    │
│  │  Engine     │   │ Graph (partial│  │   Engine        │    │
│  │  (SimPy)    │   │   -ρ / GAT)  │   │  (Holt / LSTM)  │    │
│  └─────────────┘   └──────────────┘   └─────────────────┘    │
│        │                   │                   │             │
└────────┼───────────────────┼───────────────────┼────────────┘
         ▼                   ▼                   ▼
   ┌──────────┐        ┌──────────┐        ┌──────────┐
   │  Robot   │        │ Incident │        │ Decision │
   │  Agents  │        │ Injector │        │Optimizer │
   └──────────┘        └──────────┘        └──────────┘
```

| Brain | Module | Role | What it answers |
|------|--------|------|-----------------|
| **1 — Physics Engine** | [`core.py`](digital_twin/core.py) | Discrete-event SimPy model of robots, stations & orders (ground truth) | *What happens?* |
| **2 — Causal Graph** | [`causal.py`](digital_twin/causal.py) | Constraint-based causal discovery (partial correlation) + optional GAT | *Why does it happen?* |
| **3 — Prediction Engine** | [`prediction.py`](digital_twin/prediction.py) | KPI forecasting + cascade-risk estimation (Holt's method, optional LSTM) | *What happens next?* |
| **Chaos Injector** | [`chaos.py`](digital_twin/chaos.py) | UCB-bandit-driven incident injection | *Where does it break first?* |

### What makes it different

* **Traditional simulations** replay *known* scenarios you script by hand.
* **This system** *discovers* the failure modes that matter (via a bandit), *learns*
  the cause→effect structure from telemetry (`P(Y | do(X))`, not just `P(Y | X)`),
  and *predicts* whether a wobble today will cascade into a collapse tomorrow.

---

## Install

```bash
git clone https://github.com/aabhimittal/digital-twin-fulfillment-tech.git
cd digital-twin-fulfillment-tech
pip install -r requirements.txt          # core deps: simpy, numpy, scipy, sklearn, networkx
pip install -e .                          # optional: installs the `digital-twin` CLI
```

The core has **no deep-learning dependency**. The optional Graph-Attention-Network
(causal refinement) and LSTM forecaster activate automatically if PyTorch is present:

```bash
pip install -e ".[dl]"                    # adds torch
```

---

## Quick start

### 1. Run the full demo

```bash
python examples/run_demo.py
```

```
BRAIN 1 — Physics Engine (SimPy)
  total_picks           : 2385
  processed_orders      : 946
  avg_utilization       : 0.606

CHAOS ENGINEERING — 'robot plague' (30% of the fleet fails)
  recovery time (min)   : 14.0
  cascade detected      : True

BRAIN 2 — Causal Graph (why things happen)
  num_failures   -> total_picks     (strength 0.473)
  queue_length   -> utilization     (strength 0.461)
  ranked root causes  : ['num_failures', 'queue_length', 'avg_battery']

BRAIN 3 — Prediction Engine (foresight)
  HIGH cascade risk (68%). Series is rising; dominant driver: volatility.
```

### 2. Use the CLI

```bash
# Plain simulation
digital-twin simulate --robots 50 --duration 3600

# Chaos experiment (baseline vs. injected failure)
digital-twin chaos --scenario robot_plague --robots 50 --duration 3600

# Learn the causal graph + forecast cascade risk
digital-twin analyze --robots 30 --duration 3600
```

(Without `pip install -e .`, use `python -m digital_twin.cli ...`.)

### 3. Use it as a library

```python
from digital_twin import (
    DigitalTwin, WarehouseConfig,
    ChaosInjector, ChaosScenario,
    CausalInferenceEngine, CascadePredictor,
)

# Brain 1 — run the twin
twin = DigitalTwin(WarehouseConfig(num_robots=40, seed=1))
metrics = twin.run_simulation(duration=3600, num_orders=1000)

# Chaos — inject a "robot plague" and measure the blast radius
injector = ChaosInjector(twin)
scenario = ChaosScenario.robot_plague(twin, fraction=0.3, rng=twin.rng)
impact = injector.run_chaos_experiment(scenario, duration=3600)["impact"]

# Brain 2 — learn cause→effect and answer an interventional question
engine = CausalInferenceEngine()
engine.learn_from_twin(metrics)
print(engine.predict_intervention({"num_failures": 1.0}))   # P(Y | do(X))

# Brain 3 — will the queue cascade?
queue = [m["queue_length"] for m in metrics["robot_utilization"]]
print(CascadePredictor(higher_is_worse=True).predict(queue).explanation)
```

---

## How each brain works

### Brain 1 · Physics Engine (`core.py`)

A `SimPy` discrete-event simulation. Robots are agents with a battery, a location, a
probabilistic failure model, and a tabular **Q-learning** policy for path selection:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · maxₐ' Q(s', a') − Q(s, a) ]
```

Orders arrive via a **Poisson process**; every run is fully **seeded** for
reproducibility (the whole point of a twin is deterministic replay).

### Brain 2 · Causal Graph (`causal.py`)

Correlation ≠ causation (ice-cream sales correlate with drownings — *temperature*
causes both). We learn a causal DAG with a constraint-based (PC-style) algorithm:

1. Start fully connected.
2. Drop edges between variables that are **conditionally independent** given the rest,
   tested via **partial correlation**:
   `ρ_XY|Z = (ρ_XY − ρ_XZ·ρ_YZ) / √((1−ρ_XZ²)(1−ρ_YZ²))`.
3. Orient survivors using domain-knowledge temporal precedence.

`predict_intervention()` then approximates `P(Y | do(X))` by propagating an
intervention along the strongest path — Pearl's do-calculus, first-order.

### Brain 3 · Prediction Engine (`prediction.py`)

Forecasts a KPI series (Holt's linear-trend exponential smoothing by default; an LSTM
when PyTorch is available) and scores **cascade risk** from three signals: directional
**trend**, **volatility**, and **proximity to a critical threshold**.

### Chaos Injector (`chaos.py`)

Doesn't break things at random — it uses an **Upper-Confidence-Bound bandit** to spend
its "break budget" on the incident types that historically cause the most damage:

```
UCB(a) = mean_reward(a) + c · √( ln N / N(a) )
```

Ships with `black_friday` (10× load), `robot_plague` (fleet-wide failure) and
`network_partition` scenarios.

---

## Project layout

```
digital_twin/
├── core.py         # Brain 1 — SimPy physics engine
├── causal.py       # Brain 2 — causal discovery + interventions
├── prediction.py   # Brain 3 — forecasting + cascade risk
├── chaos.py        # chaos injector (UCB bandit) + scenarios
└── cli.py          # `digital-twin` command-line entry point
examples/run_demo.py
tests/               # pytest suite (core, chaos, causal, prediction)
.github/workflows/ci.yml
```

## Development

```bash
pip install -e ".[dev]"
pytest                       # run the suite
pytest --cov=digital_twin    # with coverage
```

CI runs the suite on Python 3.9 / 3.11 / 3.12 and smoke-tests the demo.

## Roadmap

- [x] Brain 1 — physics engine, robots, stations, Poisson workload
- [x] Chaos injector with a UCB bandit + named scenarios
- [x] Brain 2 — causal discovery + interventional queries
- [x] Brain 3 — cascade-risk forecasting
- [ ] Decision-optimization layer (RL agent that recommends optimal interventions)
- [ ] Real-time dashboard for live twin telemetry

## License

MIT — see [LICENSE](LICENSE).
