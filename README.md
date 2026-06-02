# 🔬 EluSight – Chromatographic Decision Intelligence Framework

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-ready-brightgreen.svg)](docs/)

**EluSight** transforms chromatographic optimization outputs into **scientifically explainable decisions** — answering not just *what works*, but *why it works*.

---

## 🎯 What Problem Does It Solve?

Traditional optimizers tell you:

> “This is the best method.”

EluSight tells you:

* ✅ Why it is optimal
* ✅ Whether you should trust it
* ✅ What risks exist
* ✅ What tradeoffs were made
* ✅ What alternatives were rejected

---

## ⚙️ Core Capabilities

| Engine                | Function                     |
| --------------------- | ---------------------------- |
| Constraint Engine     | AQbD constraint evaluation   |
| Uncertainty Engine    | Confidence estimation        |
| Robustness Engine     | Sensitivity & stability      |
| Risk Engine           | Failure probability (ICH Q9) |
| Tradeoff Engine       | Pareto optimization insights |
| Explainability Engine | Feature importance (SHAP)    |
| Trust Engine          | Final score (0–100)          |
| Reasoning Engine      | Scientific explanation       |

---

## 🔌 Supported Inputs

Works with any optimizer:

* Bayesian Optimization
* NSGA-II / Genetic Algorithms
* Optuna / Scikit-Optimize
* DryLab
* AQbD DoE
* Custom ML pipelines with unlimited numerical & categorical parameters 

---

## 🚀 Installation

```bash
pip install elusight
```

Or:

```bash
pip install git+https://github.com/SubhraC-prog/EluSight.git
```

---

## 📦 Quick Example

```python
from elusight import Ingestor, TrustEngine
from elusight.constraints import ConstraintEngine

# Load results
methods = Ingestor.from_json("results.json")

# Define constraints
constraints = {
    "resolution": {"type": "minimum", "value": 2.0, "critical": True},
    "runtime": {"type": "maximum", "value": 20}
}

# Evaluate
constraint_engine = ConstraintEngine(constraints)
report = constraint_engine.evaluate(methods[0])

# Compute trust
trust = TrustEngine().compute_trust(
    method_id=methods[0].method_id,
    constraint_report=report
)

print(trust.overall_score)
```

---

## 📊 Outputs

### 🔹 Trust Score

* 0–100 reliability score
* Recommendation + rationale

### 🔹 Scientific Reasoning

* Constraint margins
* Risk & robustness insights
* Expert-style explanation

### 🔹 Reports

* Markdown
* JSON
* HTML

---

## 💡 Use Cases

* Pharmaceutical method development (ICH Q14)
* QC method validation
* Method transfer between labs
* Regulatory submissions
* High-throughput screening

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📚 Scientific Foundation

* ICH Q14 – Method Development
* ICH Q9 – Risk Management
* AQbD Principles
* Multi-objective Optimization

---

## 📞 Support

* GitHub: https://github.com/SubhraC-prog/EluSight
* Issues: Feature requests & bugs
* Docs: `/docs`

---

## 📄 License

MIT License

---

### 🔬 EluSight Philosophy

> Scientists don’t just need results —
> they need **confidence, reasoning, and evidence**.
