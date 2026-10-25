# Neuronal Dynamics

A computational study of the Hindmarsh–Rose model, with the Hodgkin–Huxley/Izhikevich comparison. All required parts A–D and optional section 5 are included, in English.

- **[Report](report.pdf)** — problem, mathematical formulation, methods, results and conclusions; detailed numerical tables are in the appendix.
- **[Research notebook](research.ipynb)** — executed calculations, equations, figures, ablations, statistics and interpretation. Outputs are visible without rerunning.
- [Assignment](reproducibility/assignment.md).

## Run the notebook

Use Python 3.11. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m jupyterlab research.ipynb
```

**Run All** performs the notebook analyses using the supplied raw recordings. Set `RECOMPUTE_ALL = True` in the setup cell to regenerate the simulations first. Keep the folder structure intact.

To regenerate the numerical study directly:

```bash
.venv/bin/python reproducibility/reproduce.py
```

This runs the scientific calculations and writes data, tables and figures. Then rerun the notebook to display the current results. Full regeneration takes substantially longer than reading the saved outputs; JiTCODE requires a C compiler.

## Research code and data

[reproducibility/README.md](reproducibility/README.md) maps the research methods to their source files. The directory contains the models, estimators, sensitivity experiments, ablations and statistical analysis, together with the raw recordings and computed results. Raw data retain all model variables, numerical settings and initial random-generator states required by the assignment.

Finite-time exponents and finite-scale dimensions are labeled accordingly. The conclusions concern the selected synthetic examples; a new-record classification error rate is not estimated.
