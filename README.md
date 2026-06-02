# 🔬 EluSight – Chromatographic Decision Intelligence Framework

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

**EluSight** converts chromatographic optimization outputs into **explainable, trustworthy, and visual decision intelligence**.

> Not just *what works* — but **why it works, how reliable it is, and what risks exist**.

---

## 🎯 What EluSight Does

Traditional tools → *“Best method = X”*
EluSight → *“X is best because…”*

* ✅ Scientific reasoning
* ✅ Trust score (0–100)
* ✅ Risk & robustness analysis
* ✅ Tradeoff insights (Pareto)
* ✅ Feature importance (SHAP)
* ✅ Automatic visualizations

---

## 📊 Key Outputs

### 🔹 Trust Score

```json
{
  "overall_score": 87.3,
  "recommendation": "Strongly Recommend"
}
```

### 🔹 Scientific Reasoning

* Constraint satisfaction with margins
* Robustness probability
* Risk (co-elution, failure)
* Expert alignment

### 🔹 Visualizations (Auto-generated)

* Correlation heatmap
* Feature importance
* SHAP summary
* Partial dependence
* Parallel coordinates
* Radar chart

---

## ⚙️ Core Engines

| Engine                | Purpose              |
| --------------------- | -------------------- |
| Constraint Engine     | AQbD evaluation      |
| Robustness Engine     | Sensitivity analysis |
| Risk Engine           | Failure probability  |
| Tradeoff Engine       | Pareto analysis      |
| Explainability Engine | SHAP insights        |
| Trust Engine          | Final scoring        |
| Visualization Engine  | Auto plots           |

---

## 🔌 Supported Inputs

* Bayesian Optimization
* NSGA-II / Genetic Algorithms
* Optuna / ML pipelines
* DryLab / AQbD DoE
* Any CSV / JSON dataset

---

## 🚀 Installation

```bash
pip install elusight
```

---

## 📦 Quick Example

```python
from elusight import Ingestor, TrustEngine
from elusight.constraints import ConstraintEngine

methods = Ingestor.from_csv("results.csv")

constraints = {
    "resolution": {"type": "minimum", "value": 2.0}
}

report = ConstraintEngine(constraints).evaluate(methods[0])
trust = TrustEngine().compute_trust(
    method_id=methods[0].method_id,
    constraint_report=report
)

print(trust.overall_score)
```

---

## 📁 Output Structure

```
elusight_outputs/
├── correlation_heatmap.png
├── feature_importance.png
├── shap_summary.png
├── parallel_coordinates.png
├── radar_chart.png
├── report.md
├── report.html
└── report.json
```

---

## 💡 Use Cases

* Pharmaceutical method development (ICH Q14)
* QC validation & transfer
* Regulatory submissions
* High-throughput screening

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📚 Scientific Basis

* ICH Q14 – Method Development
* ICH Q9 – Risk Management
* AQbD principles
* Multi-objective optimization

---

## 📞 Support

* GitHub: https://github.com/SubhraC-prog/EluSight
* Issues: Bug reports & features

---

## 📄 License

MIT License

---

### 🔬 Philosophy

> Scientists need **confidence, evidence, and reasoning** — not just results.
