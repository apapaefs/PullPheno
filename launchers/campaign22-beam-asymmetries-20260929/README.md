# Campaign 22 beam-oriented pull asymmetries

This launcher uses the existing campaign 22 Herwig H/Z events, campaign 14
continuum diphoton background and older campaign 20 Pythia dipole-recoil events.
No event files are copied and no old result run is reused: all scenarios need the
new asymmetry moments. The particle threshold is pT > 0.1 GeV; there is no
pull-magnitude cut. Both tagging jets contribute half the event weight.

The distributions follow the signed-density construction of Eqs. (5.1)–(5.2)
of [arXiv:1911.05090](https://arxiv.org/pdf/1911.05090), using the existing beam
and transverse projections. There are 60 equal bins in |x| on [0,0.03), plus
full and [0,0.008] plot views. Zero projections and overflow events remain in
the selected denominator. Finite-MC and 300/3000 fb⁻¹ projected-data errors
retain the within-event and selected-normalization covariance.
Individual processes have a finite-MC plot set and projected-data plot sets at
each luminosity; channel comparisons include both uncertainty models and
ratios to nominal Herwig.

## Preparation

Synchronize `/home/apapaefs/Projects/PullPheno` with the committed source before
deployment. Place this directory at
`/micron/Projects/PullPheno/analysis-launchers/launch-campaign22-beam-asymmetries-20260929`.
The existing ROOT setup, `.venv-odysseus`, input manifests and thermal guard are
required. Preparation reads ROOT headers and three representative entries in
each cached file; it does not run the event analysis.

```bash
bash run.sh --check
bash run.sh prepare
```

`--check` is a read-only preflight. `prepare` freezes `git archive HEAD`, input
manifests, common nominal-Herwig normalization, source/input hashes, batch paths
and thermal-guard configuration. It is safe to repeat with the same source and
inputs; it refuses to replace conflicting preparation. `PULLPHENO_REPOSITORY`
and `PULLPHENO_ASYMMETRY_OUTPUT` may override the repository and output roots
before preparation.

## Run manually on Odysseus

```bash
cd /micron/Projects/PullPheno/analysis-launchers/launch-campaign22-beam-asymmetries-20260929
tmux new-session -s c22-beam-asym
bash run.sh smoke
bash run.sh analysis
bash run.sh plots
bash run.sh report
```

Run each next command after the preceding phase succeeds. `smoke` processes at
most 5,000 events per process for the three scenarios at |ηj| < 4.5, then builds
the three comparison groups. `analysis` runs nine fresh analyses: three
scenarios at |ηj| < 2, 3 and 4.5. Histogram/moment filling and individual process
plots are part of this phase. It permits at most two concurrent analyses, each
using 190 workers and 384 event shards. `plots` constructs the nine three-way
and two-way comparisons from the completed archives without reading ROOT.
`report` validates pairwise closure, signed-stack closure, luminosity scaling
and embedded downloads, then builds 12 summary pages and dedicated galleries.

The thermal guard runs every 15 seconds throughout the smoke and production
ROOT phases. It monitors the controller and descendant groups using an isolated
state directory. Do not bypass a latched guard; inspect its report and use the
existing manual-resume procedure after the temperature condition is resolved.

Completed phases and runs are preserved and skipped on repeated commands.
Exclusive locks prevent simultaneous phases. Failed or interrupted jobs remain
recoverable, and the launcher refuses to restart them automatically. Inspect the
job log and the immutable `.incomplete` run/checkpoint before deciding on
recovery; the analysis CLI's `--resume-incomplete` uses a complete ROOT-pass
checkpoint without reopening events. The batch controller state must be
reconciled with any recovered run before resuming the phase.

## Logs and results

The output root is
`/micron/Projects/PullPheno/analysis-results/campaign22-beam-asymmetries-20260929`.

- Report: `report/index.html`.
- Launcher log: `launch.log`.
- Batch status, phase logs and per-job logs:
  `batches/campaign22-beam-asymmetries-20260929/`.
- Frozen code, manifests, normalization and commit/source hashes: the same batch
  directory, in `code/`, `manifests/`, `sample-normalization.json` and
  `source-provenance.json`.
- Immutable ROOT-analysis and comparison archives: `runs/`.

The `analysis` phase checks existing uncut observables/cutflows against the
previous campaign-22 results where its configured baseline state is available.
The report links the numerical asymmetry JSON/NPZ exports and records the exact
commit and all source-run IDs. The original campaign-22 report is preserved.
