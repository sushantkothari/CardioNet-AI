<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-Hybrid%20Transformer-darkred?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Healthcare-AI-critical?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Cardiovascular-Risk%20Prediction-crimson?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

# CardioNet-AI

## Overview

**CardioNet-AI** is a production-oriented deep learning framework for cardiovascular disease prediction using hybrid Transformer-based ensemble architectures on structured clinical data.

The project combines multiple neural modeling strategies — including Transformer-inspired attention learning, gated feature interaction mechanisms, and convolution-enhanced deep representations — to capture complex nonlinear relationships between cardiovascular risk indicators.

Unlike conventional machine learning pipelines that rely on shallow statistical classifiers, CardioNet-AI leverages deep representation learning and ensemble optimization to improve:

- Predictive robustness
- Clinical feature interaction learning
- Generalization performance
- Model stability
- Healthcare AI scalability

The repository demonstrates a complete end-to-end applied AI workflow covering:

- Clinical data preprocessing
- Feature engineering
- Hybrid neural architecture design
- Transformer-inspired attention learning
- Ensemble training workflows
- Cross-validation evaluation
- Performance diagnostics
- Model persistence and reproducibility

The implementation reflects modern engineering practices commonly used in real-world healthcare AI systems and applied deep learning research environments.

---

## Why This Project Matters

Cardiovascular disease remains one of the leading causes of global mortality. Early-stage AI-assisted prediction systems can significantly improve preventive healthcare workflows and clinical decision-making processes.

CardioNet-AI demonstrates how advanced deep learning architectures can be adapted for structured medical tabular data to build scalable and deployment-oriented healthcare intelligence systems.

Potential real-world applications include:

- Clinical decision support systems
- Automated cardiovascular risk assessment
- Preventive healthcare analytics
- AI-assisted patient triage
- Hospital screening systems
- Medical research and predictive analytics

---

## Key Highlights

- Hybrid Transformer-inspired ensemble learning architecture
- Attention-driven clinical feature interaction modeling
- CNN-enhanced deep representation extraction
- Gated MLP feature fusion workflow
- Stratified cross-validation evaluation pipeline
- End-to-end preprocessing and normalization workflow
- Modular deep learning experimentation framework
- TensorFlow/Keras-based implementation
- Reproducible training and evaluation pipeline
- Persistent model checkpointing
- Prediction export and evaluation logging
- Production-oriented repository structure

---

## Repository Structure

```text
CardioNet-AI/
│
├── models/
│   ├── best_cnn_transformer.keras
│   ├── best_ft_transformer.keras
│   └── best_gated_mlp.keras
│
├── results/
│   ├── heart_fold_metrics.csv
│   └── heart_test_predictions.csv
│
├── Hybrid_Transformer_Ensemble_for_Heart_Disease_Prediction.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Saved Models

The repository includes multiple trained deep learning architectures:

| Model | Description |
|---|---|
| `best_cnn_transformer.keras` | CNN-enhanced Transformer ensemble model |
| `best_ft_transformer.keras` | Fine-tuned Transformer-based architecture |
| `best_gated_mlp.keras` | Gated multilayer perceptron model |

These checkpoints allow direct inference, evaluation, and future transfer learning workflows without retraining from scratch.

---

## Dataset

The project utilizes structured cardiovascular healthcare data containing demographic, physiological, and diagnostic attributes used for binary heart disease prediction.

### Input Features

| Category | Examples |
|---|---|
| Demographic Features | Age, Sex |
| Clinical Measurements | Resting Blood Pressure, Cholesterol |
| Cardiovascular Indicators | Chest Pain Type, ECG Results |
| Exercise Metrics | Maximum Heart Rate, Exercise-Induced Angina |
| Diagnostic Parameters | ST Depression, Slope, Vessel Count |

### Target Variable

| Label | Meaning |
|---|---|
| `0` | Absence of Heart Disease |
| `1` | Presence of Heart Disease |

---

## System Architecture

### Pipeline Overview

1. Clinical dataset loading and validation
2. Data preprocessing and cleaning
3. Missing value inspection and normalization
4. Exploratory data analysis and visualization
5. Stratified train-validation splitting
6. Deep feature embedding generation
7. Transformer-inspired attention learning
8. CNN-enhanced feature extraction
9. Gated neural feature fusion
10. Ensemble model training
11. Cross-validation evaluation
12. Prediction inference and diagnostics
13. Model checkpoint persistence
14. Final prediction export

---

## Deep Learning Methodology

### Data Preprocessing Pipeline

The preprocessing workflow is designed to maximize training stability and generalization capability.

Core preprocessing operations include:

- Data cleaning and validation
- Missing value inspection
- Numerical feature scaling
- Structured tensor preparation
- Statistical distribution analysis
- Deep learning-ready feature transformation

### Hybrid Transformer Ensemble Architecture

CardioNet-AI integrates multiple deep learning paradigms into a unified cardiovascular prediction framework.

Core architectural components include:

- Dense neural embedding layers
- Transformer-inspired attention blocks
- CNN-enhanced feature extraction
- Gated MLP interaction layers
- Ensemble representation fusion
- Regularized classification heads

The architecture is specifically designed to improve:

- Clinical feature interaction learning
- Nonlinear cardiovascular pattern extraction
- Representation quality
- Predictive robustness
- Generalization performance

### Optimization Strategy

The training workflow incorporates modern deep learning optimization principles:

- Adaptive optimization algorithms
- Validation-aware checkpointing
- Regularization-aware training
- Controlled convergence strategies
- Stable gradient optimization
- Cross-validation performance evaluation

---

## Evaluation Framework

The models are evaluated using multiple healthcare-focused classification metrics to ensure balanced performance analysis across cardiovascular risk classes.

### Metrics Used

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision | Reliability of positive predictions |
| Recall | Sensitivity to disease detection |
| F1 Score | Harmonic balance of precision and recall |
| ROC-AUC | Classification separability capability |
| Confusion Matrix | Prediction distribution analysis |

### Evaluation Outputs

The repository includes exported evaluation artifacts:

| File | Description |
|---|---|
| `heart_fold_metrics.csv` | Cross-validation fold performance metrics |
| `heart_test_predictions.csv` | Final test-set predictions |

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantkothari/CardioNet-AI.git

cd CardioNet-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Launch Jupyter Notebook

```bash
jupyter notebook
```

### Execution Workflow

1. Open `Hybrid_Transformer_Ensemble_for_Heart_Disease_Prediction.ipynb`
2. Run notebook cells sequentially
3. The pipeline performs:
   - Data preprocessing
   - Model training
   - Cross-validation evaluation
   - Prediction inference
   - Visualization generation
   - Model persistence

Saved model checkpoints are automatically stored inside the `models/` directory.

Evaluation outputs and predictions are exported to the `results/` directory.

---

## Technology Stack

- Python
- TensorFlow
- Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Engineering Principles

- Transformer-inspired attention learning for structured healthcare data
- CNN-enhanced deep representation extraction
- Ensemble-driven predictive robustness
- Modular and scalable experimentation pipeline
- Reproducible deep learning workflows
- Production-oriented repository organization
- Evaluation-focused healthcare AI design
- Persistent checkpoint management
- Cross-validation-based performance assessment

---

## Potential Extensions

- Explainable AI integration using SHAP or LIME
- FastAPI or Flask inference deployment
- TensorFlow Lite or ONNX optimization
- Real-time cardiovascular risk prediction API
- Federated healthcare learning systems
- Electronic Health Record (EHR) integration
- Multi-class cardiovascular severity prediction
- Clinical dashboard deployment architecture

---

## Environment Requirements

Recommended environment configuration:

- Python 3.10+
- TensorFlow 2.15+
- GPU acceleration recommended for faster training

All required dependencies are provided in `requirements.txt`.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Author

**Sushant Kothari**

AI/ML Engineer | Data Scientist | Deep Learning Researcher

GitHub: https://github.com/sushantkothari
