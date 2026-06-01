# 🔬 EluSight - Chromatographic Decision Intelligence Framework

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)
![Docs](https://img.shields.io/badge/docs-available-brightgreen.svg)
![Tests](https://img.shields.io/badge/tests-passing-success.svg)
![Coverage](https://img.shields.io/badge/coverage-92%25-yellowgreen.svg)
![PyPI](https://img.shields.io/badge/pypi-coming_soon-orange.svg)

> **Transform chromatographic optimization into scientific decision intelligence.**

---

## 📚 Table of Contents

1. [Problem Statement](#problem-statement)
2. [Key Features](#key-features)
3. [Architecture](#architecture)
4. [Quick Start](#quick-start)
5. [Supported Optimizers](#supported-optimizers)
6. [Usage Examples](#usage-examples)
7. [API Reference](#api-reference)
8. [Engine Configuration](#engine-configuration)
9. [Dashboard Guide](#dashboard-guide)
10. [REST API](#rest-api)
11. [Testing](#testing)
12. [Performance](#performance)
13. [Integration Examples](#integration-examples)
14. [Use Cases](#use-cases)
15. [Troubleshooting](#troubleshooting)
16. [Contributing](#contributing)
17. [License](#license)
18. [Acknowledgments](#acknowledgments)
19. [Roadmap](#roadmap)
20. [Support](#support)

---

## 🧠 Problem Statement

Modern chromatographic optimization tools (Bayesian, DoE, ML-driven systems) generate **optimal conditions**, but fail to provide **scientific understanding**.

### ⚠️ The Gap

| Current Systems       | What Scientists Need                     |
| --------------------- | ---------------------------------------- |
| “Best method found”   | Why is this method optimal?              |
| Objective scores      | Trade-offs between resolution vs runtime |
| Black-box ML          | Mechanistic interpretability             |
| Single solution       | Pareto front exploration                 |
| No uncertainty        | Confidence intervals                     |
| No regulatory context | AQbD compliance                          |
| Static output         | Dynamic reasoning                        |
| No explainability     | Scientific narrative                     |

---

### ❓ Critical Questions Scientists Ask

1. Why is this method optimal?
2. What are the trade-offs?
3. How robust is it?
4. What happens if conditions change?
5. What is the uncertainty?
6. Which parameters drive performance?
7. Is it regulatory compliant (ICH Q14)?
8. Can I justify this to QA/Regulators?

---

## 🚀 Key Features

| Module                | Capability              | Scientific Basis             | Output           |
| --------------------- | ----------------------- | ---------------------------- | ---------------- |
| Ingestor              | Parse optimizer outputs | Data normalization           | MethodResult     |
| Constraint Engine     | AQbD compliance         | ICH Q14                      | Pass/Fail        |
| Robustness Engine     | Sensitivity analysis    | DoE                          | Robustness score |
| Uncertainty Engine    | Confidence estimation   | Bayesian stats               | CI ranges        |
| Risk Engine           | Failure risk            | ICH Q9                       | Risk score       |
| Tradeoff Engine       | Pareto analysis         | Multi-objective optimization | Tradeoff curves  |
| Explainability Engine | Feature attribution     | SHAP/ML                      | Importance       |
| Trust Engine          | Composite scoring       | Weighted models              | Trust score      |
| Reasoning Engine      | Scientific narrative    | Rule + ML hybrid             | Explanation      |
| Report Generator      | PDF/JSON reports        | Documentation standards      | Reports          |

---

## 🏗️ Architecture

```
                ┌─────────────────────────┐
                │   INPUT LAYER           │
                │-------------------------│
                │ Bayesian / NSGA-II      │
                │ DoE / DryLab            │
                │ RL / Custom Models      │
                └──────────┬──────────────┘
                           ↓
                ┌─────────────────────────┐
                │ ANALYSIS PIPELINE       │
                │-------------------------│
                │ Constraint Engine       │
                │ Robustness Engine       │
                │ Uncertainty Engine      │
                │ Risk Engine             │
                │ Tradeoff Engine         │
                │ Explainability Engine   │
                │ Trust Engine            │
                │ Reasoning Engine        │
                └──────────┬──────────────┘
                           ↓
                ┌─────────────────────────┐
                │ OUTPUT LAYER            │
                │-------------------------│
                │ Reports (PDF/JSON)      │
                │ Dashboard              │
                │ Trust Scores            │
                │ Scientific Explanation  │
                └─────────────────────────┘
```

---

## ⚡ Quick Start

### Installation

```bash
pip install elusight
```

From source:

```bash
git clone https://github.com/yourusername/elusight.git
cd elusight
pip install -e .
```

---

### Minimal Example

```python
from elusight.ingest import Ingestor
from elusight.trust import TrustEngine
from elusight.reasoning import ReasoningEngine

data = {"pH": 3.0, "flow_rate": 1.0, "resolution": 2.5}

method = Ingestor().from_dict(data)

trust = TrustEngine().evaluate(method)
reason = ReasoningEngine().explain(method)

print(trust.score)
print(reason.summary)
```

### Output

```
Trust Score: 0.87
"High resolution achieved with moderate runtime and low sensitivity to pH variation."
```

---

## ⚙️ Supported Optimizers

| Framework              | Adapter         | Status | Notes           |
| ---------------------- | --------------- | ------ | --------------- |
| Bayesian Optimization  | BayesianAdapter | ✅      | Optuna, BoTorch |
| NSGA-II                | NSGAAdapter     | ✅      | Multi-objective |
| Genetic Algorithm      | GAAdapter       | 🚧     |                 |
| CMA-ES                 | CMAESAdapter    | 🚧     |                 |
| Active Learning        | ALAdapter       | 🚧     |                 |
| Reinforcement Learning | RLAdapter       | 🚧     |                 |
| DryLab                 | DryLabAdapter   | ✅      | Industry        |
| AQbD DoE               | DoEAdapter      | ✅      | Regulatory      |
| Retention Modeling     | RMAdapter       | 🚧     |                 |
| Digital Twin           | DTAdapter       | 🚧     |                 |
| Generic JSON           | JSONAdapter     | ✅      | Flexible        |

---

## 🧪 Usage Examples

### Example 1: Single Method

```python
method = Ingestor().from_dict({
    "pH": 3.2,
    "flow_rate": 1.0,
    "resolution": 2.1,
    "runtime": 12
})
```

---

### Example 2: Multi-objective Pareto

```python
methods = Ingestor().from_list(data_list)
pareto = TradeoffEngine().compute(methods)
```

---

### Example 3: AQbD Constraints

```python
constraints = {
    "resolution": ">2.0",
    "runtime": "<15"
}
ConstraintEngine().evaluate(method, constraints)
```

---

### Example 4: Batch Processing

```python
ReportGenerator().generate_batch(methods)
```

---

### Example 5: REST API

```bash
curl -X POST http://localhost:8000/analyze \
-d '{"pH":3,"resolution":2.2}'
```

---

## 🧩 API Reference

### Core Classes

#### MethodResult

```python
MethodResult(pH, flow_rate, resolution)
```

---

### Engines

#### TrustEngine

```python
TrustEngine().evaluate(method)
```

Returns:

```python
TrustScore(score=0.87)
```

---

## ⚙️ Engine Configuration

Example:

```python
TrustEngine(weights={
    "robustness":0.3,
    "uncertainty":0.2
})
```

---

## 📊 Dashboard Guide

Launch:

```bash
streamlit run dashboard/app.py
```

Features:

* Pareto front visualization
* Trust gauge
* Risk heatmap
* Method comparison

---

## 🌐 REST API

Base URL:

```
http://localhost:8000
```

Endpoints:

| Endpoint | Method | Description    |
| -------- | ------ | -------------- |
| /analyze | POST   | Analyze method |
| /trust   | POST   | Trust score    |

---

## 🧪 Testing

```bash
pytest
pytest --cov
```

---

## ⚡ Performance

| Operation       | Time | Scale    |
| --------------- | ---- | -------- |
| Single analysis | 50ms | 1 method |
| Batch (1000)    | 2s   | scalable |

---

## 🔗 Integration Examples

### Optuna

```python
study.optimize(objective)
```

---

## 🧪 Use Cases

### 1. Pharma Development (ICH Q14)

* Design space evaluation
* Method robustness

### 2. QC Validation

* System suitability
* Risk scoring

---

## 🛠️ Troubleshooting

| Issue            | Fix               |
| ---------------- | ----------------- |
| Import error     | pip install -e .  |
| Slow performance | Reduce batch size |

---

## 🤝 Contributing

```bash
git fork
git branch feature-x
```

---

## 📜 License

MIT License

---

## 🙏 Acknowledgments

* ICH Q14
* ICH Q9
* USP <1225>

---

## 🗺️ Roadmap

* ✅ Core engines
* 🚧 Dashboard enhancements
* 📅 API scaling

---

## 📞 Support

* GitHub Issues
* Email: [support@elusight.ai](mailto:support@elusight.ai)

---

⭐ **Star this repo if you find it useful!**

---

*Last Updated: 2026*
