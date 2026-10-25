# Research methods and reproduction

Start with the **[report](../report.pdf)** or **[executed notebook](../research.ipynb)**. This directory supplies their scientific computations and numerical results.

From the repository root, after installing `requirements.txt`, run:

```bash
.venv/bin/python reproducibility/reproduce.py
```

The entry point regenerates the model trajectories, scalar analyses, ablations, Monte Carlo statistics and summary tables. It does not require test suites, verifiers or document-production tools. Rerun the notebook afterwards to display the regenerated results.

## Methods

| Research component | Source |
| --- | --- |
| HR equations, integration and scalar estimators | `src/cnm.py`, `run.py` |
| Embedding, duration, sampling and ordinal sensitivity | `sensitivity_analysis.py` |
| Independent equilibrium, periodic-orbit and exponent calculations | `independent_hr_checks.py` |
| Scalar reconstruction refinement, dense sampling and frozen IC2 experiment | `reconstruction_refinement.py` |
| Matched-scale and input-calibration ablations | `ablation_checks.py` |
| Paired measurement-noise ensembles and bootstrap intervals | `noise_statistics.py` |
| ordpy ordinal statistics and JiTCODE model exponents | `library_validation.py` |
| HH/Izh equations, calibration, forcing and scalar analysis | `optional/models.py`, `optional/calibrate.py`, `optional/run.py`, `optional/scalar.py` |
| Sustained-firing rules, thresholds, parameter and sampling sensitivity | `optional/firing.py`, `optional/check_calibration.py`, `optional/threshold_checks.py`, `optional/reliability_checks.py`, `optional/sampling_check.py` |
| Hybrid tangent dynamics with event saltation | `optional/forced_lyapunov.py` |
| Scientific summary tables for parts D and 5 | `synthesis.py` |
| Figure generation and theoretical H–C bounds | `publication_figures.py`, `src/publication.py`, `src/hc_bounds.py`, `optional/plot_reliability.py`, `optional/supplement.py` |

## Numerical results

- `data/`, `optional/data/`: raw physical states, events, numerical parameters and initial random-generator states.
- `tables/`, `optional/tables/`: computed statistics, sensitivity ranges and explicit reliability labels.
- `figures/`, `optional/figures/`: the report figures as PNG and vector PDF.

HR time and variables are in model units; HH/Izh time is in ms. Observation intervals differ from internal integration steps. Discarded transients can be regenerated from their initial states and durations.

The snapshot `data/reconstruction_refinement/heldout_frozen_settings_source.py` records the exact method fixed before the IC2 experiment. Together with its settings, it establishes that the new recording did not retune the reconstruction.

Standard ordinal calculations use ordpy, model-based HR reference exponents use JiTCODE, and SciPy supplies adaptive integration, root finding, spatial trees and linear algebra. Custom numerical layers implement the specified Theiler exclusion, scaling diagnostics and hybrid reset saltation.
