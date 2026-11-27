# Optimal detector memory under stochastic resetting

[Read the manuscript](paper/manuscript.pdf) | [Gain calculations](numerics/gain.csv) | [Monte Carlo results](numerics/checks.csv)

A particle makes one pass at a gated detector per launch. When a Poisson reset returns it to the source, should the detector retain its state, redraw it from stationarity, or sometimes do both?

For a homogeneous driven three-state ring with independent exponential flight stages, eight stages or fewer exclude a strict interior optimum; nine can suffice. The theoretical advantage of mixing relative to the better endpoint is less than 0.01014% for this model, including unequal stage rates. The same bound holds for a deterministic flight. This is a certified ceiling from an analytical argument, not a measured acceleration or a sharp maximum. The parameter grid in gain.csv gives attained examples. The larger improvements for a twelve-state detector compare full retention with full redraw.

## Reproducing the calculations

Use Python 3 with NumPy. ReportLab is needed only for the figures.

    python -m pip install -r requirements.txt
    python numerics/reproduce.py
    python numerics/stages.py
    python numerics/gain.py
    python numerics/layer.py
    python numerics/figures.py
    python numerics/crossover.py

The first command after installation regenerates the six ensembles in checks.csv (150,000 trajectories per setting). stages.py solves the three-state renewal system at 60-digit precision. gain.py recomputes the grid-selected examples and compares the finite-stage winners with direct high-precision solutions. layer.py and crossover.py check the finite-size results. All results concern the specified mathematical model. No experimental data were collected.

To compile the paper, run tectonic manuscript.tex in the paper directory. To rebuild the figure inputs, copy transport.pdf and crossover.pdf from numerics after running both plotting scripts.

This research draft has not undergone external peer review. OpenAI Codex assisted with derivations, numerical software and drafting, as disclosed in the manuscript.
