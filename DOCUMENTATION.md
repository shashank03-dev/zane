# AI Drug Discovery Platform - Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Components](#components)
6. [API Reference](#api-reference)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Advanced Usage](#advanced-usage)
10. [Contributing](#contributing)

## Overview

The AI Drug Discovery Platform is a state-of-the-art machine learning system for pharmaceutical research. It combines:

- **Graph Neural Networks (GNN)** for molecular structure analysis
- **Transformer models** for sequence-based learning
- **Ensemble methods** for robust predictions
- **Self-learning capabilities** for continuous improvement
- **Multi-source data integration** from PubChem, ChEMBL, and more

### Key Features

- ✅ Autonomous data collection from public databases
- ✅ Multi-model architecture (GNN, Transformer, Ensemble)
- ✅ ADMET property prediction (Absorption, Distribution, Metabolism, Excretion, Toxicity)
- ✅ Continuous learning pipeline
- ✅ Drug-likeness assessment (Lipinski's Rule, QED)
- ✅ Synthetic accessibility estimation
- ✅ Toxicity screening

## Architecture

```
drug_discovery/
├── data/
│   ├── collector.py       # Data collection from public sources
│   ├── dataset.py         # PyTorch datasets and featurization
│   └── __init__.py
├── models/
│   ├── gnn.py            # Graph Neural Networks
│   ├── transformer.py    # Transformer models
│   ├── ensemble.py       # Ensemble methods
│   └── __init__.py
├── training/
│   ├── trainer.py        # Training loop and self-learning
│   └── __init__.py
├── evaluation/
│   ├── predictor.py      # Property prediction and ADMET
│   └── __init__.py
├── utils/
│   └── __init__.py       # Utility functions
└── pipeline.py           # Main pipeline orchestrator
```

## Installation

### Requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM

### Install from source

```bash
git clone https://github.com/cosmic-hydra/zane.git
cd zane
pip install -r requirements.txt
pip install -e .
```

### Install dependencies

```bash
pip install torch torch-geometric rdkit scikit-learn transformers
```

## Quick Start

### Basic Usage

```python
from drug_discovery import DrugDiscoveryPipeline

# Initialize
pipeline = DrugDiscoveryPipeline(model_type='gnn')

# Collect data
data = pipeline.collect_data(sources=['pubchem', 'chembl'], limit_per_source=1000)

# Prepare datasets
train_loader, test_loader = pipeline.prepare_datasets(data)

# Train
history = pipeline.train(train_loader, test_loader, num_epochs=50)

# Predict
properties = pipeline.predict_properties("CC(=O)OC1=CC=CC=C1C(=O)O")
print(properties)
```

## Components

### 1. Data Collection

The `DataCollector` class fetches molecular data from:

- **PubChem**: General chemical database
- **ChEMBL**: Bioactivity database
- **DrugBank**: Approved and experimental drugs

```python
from drug_discovery.data import DataCollector

collector = DataCollector()
pubchem_data = collector.collect_from_pubchem(query='kinase inhibitor', limit=1000)
chembl_data = collector.collect_from_chembl(organism='Homo sapiens', limit=1000)
approved_drugs = collector.collect_approved_drugs()
```

### 2. Molecular Featurization

Convert SMILES to machine learning features:

```python
from drug_discovery.data import MolecularFeaturizer

featurizer = MolecularFeaturizer()

# Graph representation
graph = featurizer.smiles_to_graph("CC(=O)OC1=CC=CC=C1C(=O)O")

# Molecular fingerprint
fingerprint = featurizer.smiles_to_fingerprint("CC(=O)OC1=CC=CC=C1C(=O)O")

# Molecular descriptors
descriptors = featurizer.compute_molecular_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")
```

### 3. Models

#### Graph Neural Network (GNN)

Uses Graph Attention Networks for molecular property prediction:

```python
from drug_discovery.models import MolecularGNN

model = MolecularGNN(
    node_features=8,
    edge_features=3,
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    pooling='attention'
)
```

#### Transformer

Sequence-based learning from molecular fingerprints:

```python
from drug_discovery.models import MolecularTransformer

model = MolecularTransformer(
    input_dim=2048,
    hidden_dim=512,
    num_layers=6,
    num_heads=8
)
```

#### Ensemble

Combines multiple models:

```python
from drug_discovery.models import EnsembleModel

ensemble = EnsembleModel([gnn_model, transformer_model])
```

### 4. Training

Self-learning trainer with automatic hyperparameter tuning:

```python
from drug_discovery.training import SelfLearningTrainer

trainer = SelfLearningTrainer(
    model=model,
    learning_rate=1e-4,
    patience=10
)

history = trainer.train(train_loader, val_loader, num_epochs=100)
```

### 5. ADMET Prediction

Predict drug-like properties:

```python
from drug_discovery.evaluation import ADMETPredictor

admet = ADMETPredictor()

# Lipinski's Rule of Five
lipinski = admet.check_lipinski_rule(smiles)

# Drug-likeness (QED)
qed = admet.calculate_qed(smiles)

# Synthetic accessibility
sa_score = admet.calculate_synthetic_accessibility(smiles)

# Toxicity flags
toxicity = admet.predict_toxicity_flags(smiles)
```

## API Reference

### DrugDiscoveryPipeline

Main orchestrator class.

**Methods:**
- `collect_data(sources, limit_per_source)` - Collect molecular data
- `prepare_datasets(data, smiles_col, target_col)` - Prepare train/test datasets
- `build_model(**kwargs)` - Build the model
- `train(train_loader, val_loader, num_epochs)` - Train the model
- `predict_properties(smiles, include_admet)` - Predict molecular properties
- `generate_candidates(target_protein, num_candidates)` - Generate drug candidates
- `evaluate(test_loader)` - Evaluate model performance
- `save(filepath)` - Save pipeline
- `load(filepath)` - Load pipeline

## Training

### Self-Learning

The platform supports continuous learning:

```python
from drug_discovery.training import ContinuousLearner

learner = ContinuousLearner(
    trainer=trainer,
    data_collector=collector,
    retrain_threshold=500
)

# Add new samples
learner.add_samples(100)

# Retrain when threshold reached
if learner.add_samples(400):
    learner.retrain(train_loader, val_loader)
```

### Hyperparameter Tuning

The trainer automatically:
- Adjusts learning rate (ReduceLROnPlateau)
- Applies early stopping
- Tracks best model
- Clips gradients

## Evaluation

### Metrics

For regression tasks:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- R² (R-squared)
- Pearson correlation

For classification:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Drug-Likeness

- **Lipinski's Rule of Five**: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
- **QED**: Quantitative Estimate of Drug-likeness (0-1)
- **SA Score**: Synthetic Accessibility (1-10, lower is easier)

## Advanced Usage

### Custom Model

```python
import torch.nn as nn
from drug_discovery.training import SelfLearningTrainer

class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Your architecture

    def forward(self, x):
        # Your forward pass
        return x

model = CustomModel()
trainer = SelfLearningTrainer(model)
```

### Multi-Task Learning

```python
from drug_discovery.models import MultiTaskModel

multi_task = MultiTaskModel(
    base_model=gnn_model,
    num_tasks=5,  # Predict 5 different properties
)
```

### Custom Data Source

```python
class CustomDataCollector(DataCollector):
    def collect_from_custom_source(self):
        # Your custom data collection logic
        return dataframe

collector = CustomDataCollector()
```

## Performance Optimization

### GPU Acceleration

```python
pipeline = DrugDiscoveryPipeline(device='cuda')
```

### Batch Size Tuning

```python
train_loader, test_loader = pipeline.prepare_datasets(
    data,
    batch_size=64  # Increase for better GPU utilization
)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop
with autocast():
    output = model(data)
    loss = criterion(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

## Best Practices

1. **Data Quality**: Always validate SMILES strings
2. **Feature Engineering**: Use appropriate featurization for your task
3. **Model Selection**: GNN for structure, Transformer for sequences
4. **Ensemble**: Combine models for robust predictions
5. **Validation**: Use separate test set, not seen during training
6. **ADMET Early**: Filter candidates early with ADMET predictions

## Troubleshooting

### Common Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or model hidden dimension

**Issue**: RDKit errors with SMILES
**Solution**: Validate and sanitize SMILES strings

**Issue**: Slow data collection
**Solution**: Use cached data or reduce limit_per_source

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

CC0 1.0 Universal - Public Domain Dedication

## Citation

If you use this platform in your research, please cite:

```bibtex
@software{ai_drug_discovery_2024,
  title = {AI Drug Discovery Platform},
  author = {AI Drug Discovery Team},
  year = {2024},
  url = {https://github.com/cosmic-hydra/zane}
}
```

## Acknowledgments

Based on concepts from:
- AIAgents4Pharma
- DeepChem
- PyTorch Geometric
- RDKit

## Support

For issues and questions:
- GitHub Issues: https://github.com/cosmic-hydra/zane/issues
- Documentation: https://github.com/cosmic-hydra/zane/wiki
