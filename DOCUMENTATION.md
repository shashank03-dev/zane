# ZANE Technical Documentation

Comprehensive technical reference for the ZANE autonomous AI drug discovery platform.

## Document Control

- Product: ZANE
- Repository: cosmic-hydra/zane
- Audience: ML engineers, computational chemists, platform engineers, researchers
- Scope: architecture, module behavior, operations, workflows, quality gates, and troubleshooting

## Table of Contents

1. Introduction
2. Platform Goals and Non-Goals
3. Conceptual Architecture
4. Runtime Data Flow
5. Package and Module Reference
6. Core Pipeline API
7. Data Layer
8. Modeling Layer
9. Training Layer
10. Evaluation Layer
11. Optimization Layer
12. Physics Layer
13. Synthesis Layer
14. Multi-Agent Orchestration
15. Command-Line Interface
16. Dashboard Operations
17. AI Support Integration (Meta Llama)
18. Experiment Design and Runbook
19. Artifact and Output Management
20. Configuration Strategy
21. Validation, Testing, and Quality Controls
22. CI/CD Practices
23. Security and Responsible Research Use
24. Performance and Scaling Guidance
25. Deployment Considerations
26. Failure Modes and Troubleshooting
27. Extension Guide
28. Contribution Workflow
29. FAQ
30. License

## 1. Introduction

ZANE is an integrated computational drug discovery platform that combines molecular data ingestion, machine learning prediction, simulation-informed triage, and operational observability.

Unlike isolated scripts and disconnected tools, ZANE provides a coherent workflow where each phase can produce reproducible artifacts and feed downstream decision logic.

### Intended Usage

- Rapid prototyping of molecular screening workflows
- Model-centric candidate ranking and analysis
- Team-based research operations with shared conventions
- AI-assisted planning and interpretation support

### Out of Scope

- Clinical decision support in regulated production settings
- Direct replacement for wet-lab validation
- Regulatory-grade safety claims without additional validation systems

## 2. Platform Goals and Non-Goals

### Goals

- Modular, extensible architecture for scientific iteration
- Reproducible workflows with explicit outputs
- Practical CLI-first operations
- Strong documentation and quality controls
- Support for both baseline and advanced model experimentation

### Non-Goals

- Monolithic, hard-to-extend architecture
- Hidden side effects in core workflow execution
- Over-optimized assumptions for one domain only

## 3. Conceptual Architecture

ZANE follows a layered architecture to separate concerns and reduce coupling.

- Interface Layer
  - Command-line interface
  - Terminal dashboard
- Orchestration Layer
  - Main pipeline orchestration
  - Multi-agent flow control
- Intelligence Layer
  - Property prediction models
  - ADMET and scoring evaluators
  - Optimization engines
- Scientific Layer
  - Docking and molecular dynamics
  - Retrosynthesis and synthesis feasibility
- Data Layer
  - Data collection adapters
  - Featurization and dataset preparation
- Platform Layer
  - Tests, linting, CI, package metadata

## 4. Runtime Data Flow

Standard execution pattern:

1. Acquire molecules from configured sources.
2. Normalize and deduplicate records.
3. Convert molecular records into model-ready features.
4. Split data into train and validation subsets.
5. Train selected architecture.
6. Evaluate predictive behavior and reliability.
7. Score candidate molecules with ADMET and optional simulation signals.
8. Rank and export outputs for human review.
9. Observe progress and health via terminal dashboard.

## 5. Package and Module Reference

Primary package: `drug_discovery`

### Top-Level Modules

- `pipeline.py`: orchestration of end-to-end discovery lifecycle
- `cli.py`: command-line entrypoint and task routing
- `dashboard.py`: terminal UI for operational monitoring
- `ai_support.py`: Meta Llama integration for assistant tasks

### Subpackages

- `data`: collection, featurization, dataset management
- `models`: GNN, transformer, ensemble, equivariant models
- `training`: trainer and learning loops
- `evaluation`: predictor and scoring logic
- `optimization`: Bayesian and multi-objective optimization
- `physics`: docking and MD simulation components
- `synthesis`: retrosynthesis utilities
- `knowledge_graph`: graph data abstractions
- `agents`: orchestration agents for discovery cycles
- `web_scraping`: domain data scraping helpers

## 6. Core Pipeline API

Main orchestrator: `DrugDiscoveryPipeline`

### Constructor Inputs

- `model_type`: `gnn`, `transformer`, or `ensemble`
- `device`: compute target, typically `cpu` or `cuda`
- `cache_dir`: cache location for collected datasets
- `checkpoint_dir`: model checkpoint output path

### Core Methods

- `collect_data(sources, limit_per_source)`
  - Retrieves data from selected sources and merges results.
- `prepare_datasets(data, smiles_col, target_col, test_size, batch_size)`
  - Builds train/test loaders based on featurization mode.
- `build_model(**model_kwargs)`
  - Instantiates model architecture according to model type.
- `train(train_loader, val_loader, num_epochs, learning_rate, **trainer_kwargs)`
  - Trains and returns training history.
- `predict_properties(smiles, include_admet)`
  - Produces property and optional ADMET outputs for one molecule.
- `generate_candidates(target_protein, num_candidates, filter_criteria)`
  - Produces candidate list with attached predictions.
- `evaluate(test_loader, is_graph)`
  - Runs evaluator metrics from model predictions.
- `save(filepath)` and `load(filepath)`
  - Persists and restores pipeline state.

## 7. Data Layer

### Data Sources

Current collection pathways include:

- PubChem
- ChEMBL
- Approved-drug collections

### Data Responsibilities

- Query external sources
- Normalize schema
- Deduplicate molecular records
- Persist cached outputs for repeatability

### Featurization Pathways

- Graph-based featurization for GNN workflows
- Fingerprint/descriptor pathway for transformer or non-graph pipelines

### Dataset Construction

`MolecularDataset` provides sample retrieval behavior compatible with pipeline loaders.

## 8. Modeling Layer

### Supported Model Types

- Graph Neural Network (`MolecularGNN`)
- Transformer (`MolecularTransformer`)
- Ensemble (`EnsembleModel`)

### Model Selection Guidance

- Use GNN when molecular structural topology is central.
- Use transformer for sequence/fingerprint-heavy workflows.
- Use ensemble when improving robustness across biases.

### Extensibility Notes

To add a new architecture:

1. Implement model class in `models`.
2. Add factory logic in pipeline model builder.
3. Ensure trainer compatibility for batch format.
4. Add tests for forward pass and integration path.

## 9. Training Layer

Training relies on `SelfLearningTrainer` for fit/predict loop behavior.

### Typical Training Controls

- Number of epochs
- Learning rate
- Device placement
- Checkpoint save directory

### Expected Outputs

- Train and validation loss history
- Best checkpoint artifact
- Trained model attached to pipeline state

### Best Practices

- Track fixed splits for fair model comparisons.
- Run short smoke epochs before long jobs.
- Validate model outputs on a stable holdout set.

## 10. Evaluation Layer

Evaluation utilities provide both pure model metrics and drug-centric indicators.

### Model Metrics

- Regression metrics (for property prediction tasks)
- Consistency checks between predicted and observed values

### ADMET and Drug-Likeness

Via `ADMETPredictor`, common checks include:

- Lipinski rule assessment
- QED estimation
- Synthetic accessibility scoring
- Toxicity-flag heuristics

## 11. Optimization Layer

Optimization modules support candidate improvement and trade-off management.

- Bayesian optimization primitives
- Multi-objective optimization support

Use this layer when balancing competing objectives such as potency, synthetic feasibility, and safety proxies.

## 12. Physics Layer

Physics modules provide complementary evidence for model-driven ranking.

### Components

- Docking interfaces and scoring utilities
- Molecular dynamics simulation functions

### Use Cases

- Prioritization refinement after ML ranking
- Stability sanity checks
- Secondary evidence generation for shortlist review

## 13. Synthesis Layer

Synthesis modules provide feasibility-aware signals.

### Responsibilities

- Retrosynthesis planning scaffolding
- Route feasibility support
- Chemistry-aware candidate triage

## 14. Multi-Agent Orchestration

`agents/orchestrator.py` implements role-based agent flow.

### Agent Roles

- Generator agent
- Evaluator agent
- Planner agent
- Optimizer agent

### Workflow Pattern

1. Generate candidates
2. Evaluate by selected criteria
3. Plan experiment subset
4. Optimize final shortlist

## 15. Command-Line Interface

CLI entry module: `python -m drug_discovery.cli`

### `train`

Train a model from collected data.

Example:

```bash
python -m drug_discovery.cli train --model transformer --epochs 20 --batch-size 32
```

### `predict`

Predict properties for one molecule using a checkpoint.

Example:

```bash
python -m drug_discovery.cli predict "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --model gnn \
  --checkpoint ./checkpoints/gnn_model.pt
```

### `admet`

Run ADMET-focused analysis for one molecule.

Example:

```bash
python -m drug_discovery.cli admet "CC(=O)OC1=CC=CC=C1C(=O)O"
```

### `collect`

Collect molecules from selected sources.

Example:

```bash
python -m drug_discovery.cli collect --sources pubchem chembl --limit 500
```

### `dashboard`

Display terminal dashboard in static or live mode.

Examples:

```bash
python -m drug_discovery.cli dashboard --static
python -m drug_discovery.cli dashboard --refresh 1.0 --iterations 60
```

### `assist`

Use Llama-based AI support.

Example:

```bash
python -m drug_discovery.cli assist "Propose next validation experiments"
```

## 16. Dashboard Operations

Dashboard module provides a professional terminal control surface.

### Typical Panels

- Run metadata and mode
- KPI summary
- Training monitor
- Candidate queue
- Alerts/status panel

### Practical Usage

- Use static mode for logs and CI snapshots.
- Use live mode during active experiments.
- Capture snapshots to compare run behavior over time.

## 17. AI Support Integration (Meta Llama)

AI support module: `ai_support.py`

### Defaults

- Default model id: `meta-llama/Llama-3.2-1B-Instruct`
- Prompt format: system guidance plus optional context and user request

### Advanced Invocation

```bash
python -m drug_discovery.cli assist "Draft a short assay plan" \
  --model-id meta-llama/Llama-3.2-1B-Instruct \
  --context "Top candidates: Caffeine, Warfarin" \
  --max-new-tokens 300 \
  --temperature 0.7 \
  --top-p 0.9
```

### Access Requirements

- Account access to gated checkpoint if required
- Authentication token in environment (for example `HF_TOKEN`)
- Network access to model hub

## 18. Experiment Design and Runbook

### Recommended Baseline Runbook

1. Collect 200 to 1000 molecules.
2. Train transformer baseline and save checkpoint.
3. Run evaluation and ADMET scoring.
4. Produce top candidate shortlist.
5. Review in dashboard and export artifacts.

### Comparative Runbook

1. Train GNN and transformer with aligned split.
2. Compare losses and ranking agreement.
3. Build ensemble candidate ranking.
4. Re-score top molecules with simulation evidence.

### Human Review Runbook

1. Export final shortlist.
2. Use AI assist to draft validation strategy.
3. Review with domain experts before downstream action.

## 19. Artifact and Output Management

Artifacts should be organized by run identifiers and timestamps.

Suggested contents per run folder:

- Input dataset snapshot
- Training logs and metrics
- Model checkpoint
- Candidate scoring output CSV/JSON
- Run summary metadata

This enables repeatability and post-hoc auditability.

## 20. Configuration Strategy

Configuration may be managed through:

- CLI flags for routine workflows
- Python constructor arguments for programmatic workflows
- Optional project config files for environment-specific defaults

Recommended policy:

- Keep defaults conservative.
- Promote explicit override of experiment-critical parameters.
- Record effective config in run artifacts.

## 21. Validation, Testing, and Quality Controls

### Test Coverage

The repository includes tests for:

- Data and dataset behavior
- Model components
- Pipeline orchestration
- Evaluation logic

### Recommended Local Checks

```bash
python -m pytest -q
python -m ruff check .
python -m black --check .
```

### Review Criteria

- No regression in existing tests
- Lint and style cleanliness
- Documented behavior for user-visible changes

## 22. CI/CD Practices

Core CI expectations:

- Run tests on push and PR
- Enforce static checks
- Surface actionable logs for failures

Operational recommendations:

- Keep CI deterministic where possible
- Isolate slow external-dependency tests
- Preserve artifact logs for failed runs

## 23. Security and Responsible Research Use

This platform is for research support.

Mandatory guardrails:

- Do not interpret outputs as clinical directives.
- Keep expert review in the loop for critical decisions.
- Validate key conclusions with experimental evidence.
- Apply access controls for sensitive data and artifacts.

## 24. Performance and Scaling Guidance

### Training Throughput

- Increase batch size gradually based on memory limits.
- Use GPU acceleration for large runs.
- Profile data loading bottlenecks before model optimization.

### Operational Scaling

- Separate data collection from training in larger workflows.
- Use cached datasets to improve reproducibility and speed.
- Parallelize independent experiments where feasible.

## 25. Deployment Considerations

For internal platformization:

- Standardize run directories and naming conventions.
- Provide pinned dependency environments.
- Integrate CLI flows into schedulers or pipeline runners.
- Implement monitoring around long-running tasks.

## 26. Failure Modes and Troubleshooting

### Llama Model Cannot Load

Likely causes:

- Missing/invalid auth token
- Access not granted for model id
- Blocked network route to model hub

### Training Divergence

Likely causes and actions:

- Learning rate too high: reduce by 2x to 10x
- Unstable batch composition: reduce batch size
- Data quality issues: inspect invalid or noisy records

### CLI Argument Errors

Actions:

- Review command help and required flags
- Confirm checkpoint path exists
- Activate expected Python environment

### Slow Data Collection

Actions:

- Lower source limits for quick iteration
- Use cached datasets for repeat runs
- Decouple external collection from training jobs

## 27. Extension Guide

### Add a New Data Source

1. Extend data collector with new method.
2. Normalize schema to required columns.
3. Add source switch logic in pipeline collection path.
4. Add test coverage for parsing and merge behavior.

### Add a New Model

1. Create model module in `models`.
2. Add selection branch in pipeline model builder.
3. Ensure trainer compatibility for batch format.
4. Add unit and integration tests.

### Add a New CLI Command

1. Register parser and options in CLI module.
2. Implement command handler function.
3. Add docs and examples.
4. Add smoke tests where appropriate.

## 28. Contribution Workflow

Recommended flow:

1. Create a focused branch.
2. Keep changes scoped and reviewable.
3. Run local checks before PR.
4. Update documentation with behavior changes.
5. Include validation summary in PR description.

## 29. FAQ

### Is ZANE production-ready for clinical decisions?

No. It is a research and decision-support platform requiring expert oversight and experimental validation.

### Can I replace built-in models?

Yes. The architecture is modular and designed for extension.

### Can I run this without a GPU?

Yes. CPU execution is supported, but larger training jobs will be slower.

### Is dashboard usage required?

No. Dashboard is optional but recommended for operational visibility.

## 30. License

CC0 1.0 Universal
