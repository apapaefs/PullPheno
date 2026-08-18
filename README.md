# Jet-pull phenomenology in VBF Higgs production

## Aim and scope

The pull vector is designed to resolve colour-flow information that is not contained in the four-momenta of reconstructed jets. In vector-boson-fusion (VBF) Higgs production, the two tagging jets are associated with two quark lines that exchange no colour. Each outgoing quark is therefore expected to remain colour connected predominantly to its corresponding beam remnant. In gluon-fusion (ggF) Higgs production with two jets, colour connections can instead extend across the event and between the tagging jets. This leads to the qualitative expectation that the pull vectors of the VBF tagging jets point preferentially outwards, towards the beam directions, whereas ggF configurations exhibit a larger inward component.

The purpose of this project is to determine whether this difference remains observable once a physical Higgs decay, realistic event selection, detector acceptance, pileup, irreducible backgrounds, event yields and systematic uncertainties are included. The central result should therefore not be a truth-level ROC curve. It should be the change in the expected VBF signal sensitivity or precision when pull information is added to an otherwise realistic analysis.

The concrete nominal event-generation setup used to start this programme is recorded in [PILOT_GENERATION.md](PILOT_GENERATION.md).

## Running the particle-level analyses

`HwSimPythonAnalysis.py` reads the merged HwSim ROOT event records, applies the
Higgs or Z object selection, reclusters anti-$k_T$ $R=0.4$ jets, applies the
cut-based VBF selection and calculates the beam-signed pull vector of both
tagging jets.  It can also train separate Higgs and Z XGBoost classifiers that
replace the three VBF cuts while sharing exactly the same common selection.
ROOT, the FastJet Python bindings, NumPy and Matplotlib must be available.  The
preliminary files on Timur are configured in `samples_preliminary_timur.json`.

Cut-based runs remain the backward-compatible default:

```bash
python3 HwSimPythonAnalysis.py \
  --samples samples_preliminary_timur.json \
  --output-root results \
  --run-name nominal \
  --workers 5
```

For a combined cut-based and XGBoost run on Timur, first create the isolated
runtime once and then request both branches:

```bash
python3 -m venv --system-site-packages .venv-xgboost
.venv-xgboost/bin/python -m pip install -r requirements-xgboost.txt
.venv-xgboost/bin/python HwSimPythonAnalysis.py \
  --samples samples_preliminary_timur.json \
  --analyses cutbased xgboost \
  --output-root results \
  --run-name nominal-cutbased-xgboost \
  --workers 5
```

### Odysseus campaign 3

The complete `summer2026` campaign-3 samples on Odysseus are configured in
three manifests, keeping each generator setup separate:

- `samples_odysseus_campaign00003_herwig.json`: nominal Herwig;
- `samples_odysseus_campaign00003_herwig_preco.json`: Herwig with
  `pReco=0.25`;
- `samples_odysseus_campaign00003_mg5_pythia.json`: MG5_aMC plus Pythia 8.

The event disk is mounted as `/micron` (not `/micro`).  The manifests point to
the immutable merged files under
`/micron/Projects/PullPheno/campaigns/campaign-00003/merged/`.  Run only after
the campaign dashboard reports the selected job as complete; never analyze a
hidden `.partial` merge product.

Create the Odysseus runtime once:

```bash
source /home/apapaefs/Projects/root/root_v6.40.02/bin/thisroot.sh
python3 -m venv --system-site-packages .venv-odysseus
.venv-odysseus/bin/python -m pip install -r requirements-odysseus.txt
```

Source ROOT in each new shell, then run a low-statistics smoke test before the
full sample:

```bash
source /home/apapaefs/Projects/root/root_v6.40.02/bin/thisroot.sh
.venv-odysseus/bin/python HwSimPythonAnalysis.py \
  --samples samples_odysseus_campaign00003_herwig.json \
  --analyses cutbased xgboost \
  --output-root /micron/Projects/PullPheno/analysis-results \
  --run-name campaign00003-herwig-smoke \
  --workers 5 \
  --max-events 1000
```

Remove `--max-events 1000` and change the run name only after the smoke output
has been checked.  Each manifest records both the generator cross section and
the final-state cross section consumed by the analysis.  The latter includes
one factor of `BR(H to gamma gamma)=0.002270` for `VBFH` and `QCDHjj`; no extra
branching factor is applied to `aajj` or the decay-inclusive Z samples.

### Odysseus campaign 4: independent CR scenarios

Campaign 4 provides two complete, statistically independent five-process
Herwig sets with one million events per process:

- `samples_odysseus_campaign00004_herwig.json`: default Herwig;
- `samples_odysseus_campaign00004_herwig_preco025.json`: all five processes
  generated with `ReconnectionProbability=0.25`.

Both manifests deliberately use the campaign-4 nominal final-state cross
sections.  The pReco manifest separately records its own generator cross-section
estimates and uncertainties as provenance; those estimates do not alter its
normalisation.  Before production, run both scenario commands with the same
`--max-events 1000` and `-smoke` run-name suffix, using the partial nominal run
as the pReco smoke-test model.  Never reuse a partial model run for the full
production sample.  The production nominal command, which trains and freezes
the five-fold XGBoost ensemble, is:

```bash
source /home/apapaefs/Projects/root/root_v6.40.02/bin/thisroot.sh
.venv-odysseus/bin/python HwSimPythonAnalysis.py \
  --samples samples_odysseus_campaign00004_herwig.json \
  --analyses cutbased xgboost \
  --output-root /micron/Projects/PullPheno/analysis-results \
  --run-name campaign00004-herwig \
  --workers 5
```

Use the completed path printed by that command as `NOMINAL_RUN`, and apply the
frozen ensemble to the independent pReco set:

```bash
.venv-odysseus/bin/python HwSimPythonAnalysis.py \
  --samples samples_odysseus_campaign00004_herwig_preco025.json \
  --analyses cutbased xgboost \
  --xgb-model-run NOMINAL_RUN \
  --output-root /micron/Projects/PullPheno/analysis-results \
  --run-name campaign00004-herwig-preco025 \
  --workers 5
```

With `PRECO_RUN` set to the second completed path, construct the comparison
without reopening a ROOT file:

```bash
.venv-odysseus/bin/python HwSimPythonAnalysis.py \
  --compare-runs NOMINAL_RUN PRECO_RUN \
  --analyses cutbased xgboost \
  --output-root /micron/Projects/PullPheno/analysis-results \
  --run-name campaign00004-cr-comparison
```

The comparison run sums all processes within each scenario and produces
non-stacked error-bar plots for the five pull observables, separately for
projected counting and finite-MC uncertainties.  Its lower panels show each
scenario relative to the first run.  It also stores full covariance matrices,
$R_i$, $f_{\rm beam}$, differences, directional $f_{\rm beam}$ sensitivities
and six-bin Mahalanobis $D^2$.  It also performs pairwise reference-versus-CR
hypothesis tests for every pull histogram.  The primary ranking uses unit-area
shape differences with the full projected-data plus independent-scenario MC
covariance; a separate rate-and-shape result corresponds to the absolute-yield
plots.  Each $D^2$ is converted to a local expected upper-tail chi-square
$p$-value using the supported covariance rank.  These statistical-only values
do not include experimental or modelling systematics, and selecting the
smallest value across correlated plots incurs a look-elsewhere effect.
To add another complete CR scenario, analyze it
with the same `NOMINAL_RUN` models and append its completed path to
`--compare-runs`; the existing source runs remain untouched.

For the absolute signed pull angle, the comparison additionally reproduces the
single-run expected-yield presentation: the reference Herwig processes remain
stacked in their configured order, while every independent CR variation is
overlaid only as its complete total prediction with event-level MC-statistical
error bars.  Variation processes are never stacked and are neither added to
nor substituted inside the nominal stack.  A separate companion plot adds a
lower variation/reference ratio panel.  Its CR-point error bars propagate the
finite-MC uncertainties of the statistically independent variation and
reference samples; bins with zero nominal yield are masked.

Analysis version 2.3 additionally stores an event-level joint observable with
ten frozen XGBoost-score quantiles and the six absolute signed-pull-angle bins.  The
quantile boundaries are fixed from the nominal total prediction after common
selection and copied unchanged into every independent CR run.  The comparison
therefore produces score-by-pull maps, $\Delta f_{\rm beam}$ in every score
quantile, and exhaustive scans of all contiguous two- and three-category score
partitions.  Each category's pull distribution is normalized independently,
so the scan measures conditional pull-shape sensitivity rather than a change in
the XGBoost category rate.  For multiple CR variations, one common partition is
chosen by maximizing the minimum data-plus-independent-MC $D^2$ across the
variations.

The resulting boundaries are an exploratory design choice, since the same CR
samples are used to propose them.  They must be frozen and confirmed with
statistically independent simulations before quoting a sensitivity.  Runs made
before version 2.3 do not contain the joint event moments and cannot be used to
construct this diagnostic with `--from-run`; nominal and variation analyses
must be rerun once from their ROOT events.  After that one-time pass, all maps,
partition scans and later multi-scenario comparisons are generated without
reopening ROOT files.

The manifest assigns each process an XGBoost `role` of `signal` or
`background`.  The classifiers use only $m_{jj}$,
$|\Delta y_{jj}|$, the two tagging-jet transverse momenta, boson $p_T$ and
$z^*$.  Nominal training uses five rotating, deterministic pipelines.  For
each pipeline, three folds train the classifier, one separate fold fixes that
pipeline's score cut by maximising $S/\sqrt{S+B}$, and the remaining fold is
used only for physics application.  Across all five pipelines every event is
physics-tested exactly once, validation-tested exactly once, and used for
training three times.  The post-XGBoost physics plots therefore use the full
common-selected Monte Carlo sample with unit event weights; no event is scored
by a model or threshold that used its own fold, and no inverse 20% sampling
correction is applied.

Each channel saves five model/threshold pairs beneath
`models/xgboost/<channel>/`.  Application to an independent CR manifest routes
every event deterministically through one frozen pair, so the new sample is
also used exactly once and no model or threshold is retrained.

To apply already frozen models and score cuts to independent CR samples, use:

```bash
.venv-xgboost/bin/python HwSimPythonAnalysis.py \
  --samples <variation-manifest.json> \
  --analyses xgboost \
  --xgb-model-run results/runs/<nominal-combined-run-id> \
  --output-root results \
  --run-name <variation-name>
```

For a non-physical smoke test, use `--max-events N`.  Such output is visibly
marked as partial and retains the full generated sum of weights in its
normalisation metadata.  The default luminosities are $300$ and
$3000\,\mathrm{fb}^{-1}$; they may be replaced with
`--luminosities L1 L2 ...`.  `--workers N` sets the maximum number of ROOT
event-range processes; one worker remains the conservative default.

Large samples can be divided into deterministic contiguous entry ranges with
`--event-shards S`.  Shards retain the original manifest file/entry identity
and are merged in global event order, so generated `sumw`, weighted cutflows,
histogram `sumw2`, same-event pull covariances and XGBoost fold assignments are
unchanged apart from floating-point reduction order.  For example:

```bash
.venv-odysseus/bin/python HwSimPythonAnalysis.py \
  --samples <manifest.json> \
  --analyses cutbased xgboost \
  --event-shards 8 \
  --workers 8 \
  --output-root /data/Projects/PullPheno/analysis-results \
  --run-name <name>
```

Every sample receives up to eight shards in this example.  Jobs are scheduled
largest-first, so the ranges of a dominant sample such as QCDZjj occupy the
worker pool before small samples.  Reading many ranges from one rotating disk
can limit the speedup; staging the input ROOT files on NVMe is recommended for
production sharded runs.  The complete shard plan and effective worker counts
are recorded in `run.json`.

Every invocation creates a new immutable directory beneath `results/runs/`.
The run identifier contains a UTC timestamp, the optional run name and a hash
of the resolved analysis configuration.  Existing runs and plots are never
overwritten.  A completed run contains:

- expected-yield and unit-area PNG/PDF plots;
- differential $R_i$ plots with separate projected-data and MC uncertainties;
- saved Higgs and Z models, frozen thresholds and classifier diagnostics;
- long-form CSV and Markdown cutflows;
- JSON summaries and compressed NumPy histogram arrays;
- a persistent `run.log` recording parent-process diagnostics;
- a self-contained `index.html` with plot filters, cutflows and numerical
  summaries.

The top-level `results/index.html` is an atomically regenerated catalogue of
all completed runs.  Directories prefixed with `.incomplete-` record failed or
interrupted runs and are deliberately excluded from that catalogue.  After
the ROOT event pass, the analysis writes an atomic recovery checkpoint before
starting XGBoost.  If a later Python exception occurs, the incomplete run also
contains `failure.json`; the checkpoint is retained while successful runs
remove it to avoid permanently duplicating the compact event table.

Resume a failed run that reports
`"root_pass_checkpoint_available": true` without rereading ROOT events:

```bash
python3 HwSimPythonAnalysis.py \
  --resume-incomplete results/runs/.incomplete-<failed-run-id> \
  --analyses cutbased xgboost \
  --output-root results \
  --run-name <name>-resume
```

The resumed invocation creates another immutable run.  Its analysis list must
exactly match the checkpoint, and its frozen-model source (if any) is restored
from the checkpoint rather than accepted from the command line.

Plot-only revisions can reuse the stored histogram arrays without reopening
the ROOT events.  The command still creates a new immutable run and records
the parent run explicitly:

```bash
python3 HwSimPythonAnalysis.py \
  --from-run results/runs/<completed-run-id> \
  --output-root results \
  --run-name folded-angle
```

The manifest uses final-state cross sections in pb.  No Higgs or Z branching
fraction is applied in the analysis code.  For every sample and luminosity the
normalisation is

```text
1000 * luminosity_fb * cross_section_pb / sum_generated_evweight.
```

Each selected event fills the signed-pull histograms twice, once per tagging
jet, with half of the event weight.  Consequently the signed-angle histogram
integral is checked against the expected selected-event yield.  The signed
angle is defined through

```text
t_beam       = sign(y_jet) * t_y
signed_angle = atan2(t_phi, t_beam)
```

and uses twelve equal bins on $[-\pi,+\pi]$.

The analysis also stores a folded presentation,

```text
folded_angle = abs(signed_angle),
```

using six equal bins on $[0,\pi]$.  Each negative-angle bin is summed with
its positive-angle partner, including the corresponding `sumw2`, so the
folded histogram has exactly the same expected-yield integral as the signed
histogram.  The signed distribution is retained to expose possible
left--right implementation asymmetries.

The normalized differential observable is

```text
R_i = N_i / sum_j N_j,
f_beam = R_1 + R_2 + R_3.
```

Thus all six $R_i$ values sum to one; their average is identically $1/6$ and
is not $f_{\rm beam}$.

For each selected event the code forms a six-component vector from its two
half-weight tagging-jet entries.  Event-level first and second moments retain
the within-event jet correlation.  The HTML, JSON, CSV and NPZ artifacts store
the full six-bin covariance for both the expected counting uncertainty at each
luminosity and the current weighted-MC uncertainty.  The latter is fixed by the
generated samples and does not scale with displayed luminosity.  Detector and
physics-modelling systematic uncertainties are not yet included.

## Executive recommendation

We propose a two-tier analysis programme:

1. **Higgs application:**
   $$
   pp\to Hjj\to\gamma\gamma jj,
   $$
   with VBF Higgs production as the process of interest, ggF $Hjj$ as an irreducible resonant background or second signal component, and continuum diphoton production as the principal non-resonant background.

2. **High-statistics validation:**
   $$
   pp\to Z(\ell^+\ell^-)jj,\qquad \ell=e,\mu,
   $$
   comparing electroweak $Zjj$ production with QCD-induced Drell--Yan $Zjj$ production.

If a single Higgs final state is required, $H\to\gamma\gamma$ is the preferred starting point. If the immediate objective is instead to establish that tagging-jet pull can be measured and modelled reliably, $Z\to\ell\ell$ provides the stronger first test.

## Choice of Higgs final state

### Baseline: $H\to\gamma\gamma$

The diphoton channel provides the cleanest environment in which to isolate the incremental information carried by the tagging-jet pull vectors:

- the Higgs decay products are colourless, so the colour-flow observable probes the production process rather than the decay;
- the narrow $m_{\gamma\gamma}$ peak separates resonant Higgs production from continuum backgrounds;
- the diphoton sidebands constrain the non-resonant background directly, reducing dependence on an absolute Monte Carlo prediction for prompt $\gamma\gamma jj$, $\gamma j$ and multijet events;
- VBF and ggF contributions can be extracted simultaneously in categories defined using tagging-jet kinematics and pull information;
- the channel has well-developed triggers, photon identification and experimental categorisation strategies.

The full Run-2 CMS diphoton analysis used VBF-like categories containing approximately 200 expected Higgs events at $137\,\mathrm{fb}^{-1}$, of which approximately 110 were expected to arise from VBF production. The quoted $S/(S+B)$ values in a $\pm1\sigma_{\mathrm{eff}}$ diphoton-mass window range from approximately 0.08 to 0.56 across these categories. This establishes a realistic event-count scale against which a pull-based analysis can be tested [1].

A preliminary truth-level study has already examined tagging-jet pull in VBF and ggF $H\to\gamma\gamma$ events. It found the expected outward/inward tendency, but also found that the difference is small and identified reconstructed-level validation as the necessary next step [2]. The principal contribution of the present project should therefore be the inclusion of reconstruction, physical yields, backgrounds, luminosity projections and modelling uncertainties.

### Complementary Higgs channels

| Decay channel | Assessment |
| --- | --- |
| $H\to\gamma\gamma$ | Preferred baseline: narrow mass peak, clean trigger and data sidebands. |
| $H\to\tau^+\tau^-$ | Preferred follow-up: approximately 28 times larger branching fraction, but missing neutrinos, $Z\to\tau\tau$, top and fake-$\tau$ backgrounds complicate the interpretation. |
| $H\to WW\to e\mu\nu\nu$ | Larger raw rate than the diphoton channel, but no narrow Higgs mass peak and substantial $t\bar t$ and continuum $WW$ backgrounds. |
| $H\to ZZ\to4\ell$ | Very clean, but too statistically limited for an effect expected to be small. |
| $H\to b\bar b$ | Large branching fraction, but the multijet background, four-jet combinatorics and tagging-jet misassignment obscure the production colour flow. |
| Invisible Higgs decay | Useful for developing particle-level shapes, but it is not a Standard Model Higgs final state and therefore does not provide the desired phenomenological validation. |

The VBF $H\to\tau\tau$ production rate has already been measured at the LHC, demonstrating that this channel is viable for a subsequent combination [3]. It is nevertheless less suitable for the first pull study because its background extraction and event reconstruction introduce additional correlations that are absent from the diphoton mass fit.

## Event-rate benchmarks

At $\sqrt{s}=13.6$ TeV and $m_H=125$ GeV, the inclusive Standard Model cross sections are approximately

$$
\sigma_{\mathrm{ggF}}=52.23\ \mathrm{pb},\qquad
\sigma_{\mathrm{VBF}}=4.078\ \mathrm{pb},
$$

and the branching fraction is

$$
\mathrm{BR}(H\to\gamma\gamma)=2.270\times10^{-3}.
$$

These values are taken from the LHC Higgs Cross Section Working Group recommendations [4,5].

The current `akcolor` particle-level samples use anti-$k_T$ jets with $R=0.4$, $p_{T,j}>30$ GeV and $|\eta_j|<3$. Selecting the two leading jets and requiring $m_{jj}>400$ GeV gives the weighted cross sections

$$
\sigma_{\mathrm{VBF}}^{\mathrm{current\ selection}}=0.460349\ \mathrm{pb},\qquad
\sigma_{\mathrm{ggF}\ Hjj}^{\mathrm{current\ selection}}=0.170727\ \mathrm{pb}.
$$

Multiplying these rates by the diphoton branching fraction gives the following raw Higgs yields:

| Process | Cross section after dijet selection and $H\to\gamma\gamma$ branching fraction | $140\,\mathrm{fb}^{-1}$ | $300\,\mathrm{fb}^{-1}$ | $3\,\mathrm{ab}^{-1}$ |
| --- | ---: | ---: | ---: | ---: |
| VBF $H\to\gamma\gamma$ | 1.045 fb | 146 | 313 | 3,135 |
| ggF $Hjj\to\gamma\gamma jj$ | 0.388 fb | 54 | 116 | 1,163 |

These numbers are ceilings rather than selected experimental yields. They do not include photon acceptance and identification, detector response, pileup or continuum backgrounds. They nevertheless indicate that a Run-2 or Run-3 analysis can provide a proof of method, while a statistically useful improvement in the Higgs measurement is more naturally expected at the HL-LHC.

## Physical signal and background definition

The analysis should not be formulated as a binary classification of combined VBF+ggF Higgs production against a generic background. Such a definition would dilute the colour-flow effect that motivates the study. Instead, the statistical model should contain at least three components:

1. **VBF $H\to\gamma\gamma$:** the principal parameter of interest, $\mu_{\mathrm{VBF}}$;
2. **ggF $Hjj\to\gamma\gamma jj$:** an irreducible resonant background, with $\mu_{\mathrm{ggF}}$ either floated or constrained as a second production-mode parameter;
3. **continuum background:** prompt $\gamma\gamma jj$, electroweak diphoton production, $\gamma j$ and multijet events, with the mass spectrum constrained by diphoton sidebands.

Other Higgs production modes, including hadronic $VH$, $t\bar tH$, $tH$ and $b\bar bH$, should be included at their physical rates. They are small but resonant and can populate VBF-like categories.

The analysis should compare the expected uncertainty on $\mu_{\mathrm{VBF}}$, or the expected VBF discovery significance, for four nested inputs:

1. standard VBF kinematics;
2. standard kinematics plus central-jet or gap-activity information;
3. standard kinematics plus pull observables;
4. all inputs combined.

This comparison quantifies the information supplied specifically by pull and its complementarity with established colour-flow-sensitive selections. ROC curves and AUC values are useful diagnostics, but they should remain secondary to a profile-likelihood result using physical event weights.

## Baseline object and event selection

A suitable starting selection is:

- two isolated photons satisfying, for example,
  $$
  p_T^{\gamma_1}/m_{\gamma\gamma}>0.35,\qquad
  p_T^{\gamma_2}/m_{\gamma\gamma}>0.25;
  $$
- photon acceptance $|\eta_\gamma|<2.5$, with detector transition regions removed in the experiment-specific implementation;
- a broad diphoton mass range, e.g. $105<m_{\gamma\gamma}<160$ GeV, retained for the signal-plus-sideband fit;
- anti-$k_T$ jets with $R=0.4$, $p_{T,j}>30$ GeV and detector-appropriate rapidity acceptance;
- at least two tagging-jet candidates with $m_{jj}>400$ GeV and $|\Delta\eta_{jj}|>2.5$;
- categories in $m_{jj}$, $|\Delta\eta_{jj}|$, Higgs centrality and, where useful, $p_T^{\gamma\gamma jj}$.

The preselection should not be made unnecessarily tight. Very restrictive VBF cuts would eliminate most ggF contamination before pull is allowed to contribute. Binned $m_{jj}$ and $|\Delta\eta_{jj}|$ categories retain more information and permit a direct test of where the pull observable is useful.

Two tagging-jet definitions should be compared:

- the two highest-$p_T$ jets, matching the current prototype;
- the jet pair with the largest $m_{jj}$, subject to the VBF preselection.

Incorrect identification of the underlying tagging jets is expected to dilute the pull correlation and should be quantified explicitly.

## Pull-observable definition

For a jet $J$, the pull vector is

$$
\vec t_J=\sum_{i\in J}\frac{p_{T,i}}{p_{T,J}}\,|\vec r_i|\,\vec r_i,
\qquad
\vec r_i=(y_i-y_J,\,\phi_i-\phi_J).
$$

For the two tagging jets, define unit vectors $\hat n_{12}$ and $\hat n_{21}$ that point from each jet towards the other. A useful signed projection is then

$$
T_{\parallel}=\vec t_1\cdot\hat n_{12}
              +\vec t_2\cdot\hat n_{21}.
$$

With this convention, positive values correspond to an inward pull component, while negative values correspond to an outward component. VBF events are expected to favour the latter relative to ggF events. The individual parallel and perpendicular projections, pull magnitudes and symmetrised asymmetries should also be retained.

The pull angle can be shown as an intuitive auxiliary observable, but it should not be the sole theoretical object because the angle is not infrared-and-collinear safe. Suitable projections and asymmetry distributions retain colour-flow sensitivity while improving theoretical control [6].

At least two constituent definitions should be investigated:

- **charged-track pull**, which offers strong pileup rejection but restricts the tagging jets to tracking acceptance;
- **particle-flow or calorimeter pull**, which retains forward VBF jets but is more sensitive to pileup, calorimeter granularity and constituent subtraction.

This distinction is particularly important because the existing prototype uses all stable constituents with no constituent $p_T$ threshold and $|\eta|<10$. That definition is appropriate for a particle-level benchmark, but not for a realistic Run-2/3 projection. The HL-LHC tracker extension towards $|\eta|\simeq4$ can significantly improve the acceptance for track-based pull in VBF topologies [7].

## The $Z(\ell\ell)jj$ validation analysis

Electroweak $Zjj$ production contains the same essential colour structure as VBF Higgs production: the two quark lines exchange an electroweak boson and no net colour. QCD-induced $Zjj$ production provides a high-rate colour-exchange comparison. The leptonic $Z$-boson decay supplies a clean trigger and a narrow reconstructed mass peak without introducing coloured decay products.

CMS measured the electroweak $Z(\ell\ell)jj$ cross section at 13 TeV using only $35.9\,\mathrm{fb}^{-1}$, obtaining

$$
\sigma_{\mathrm{EW}}(\ell\ell jj)
=552\pm19\ \mathrm{(stat)}\pm55\ \mathrm{(syst)}\ \mathrm{fb}
$$

in a region with $m_{\ell\ell}>50$ GeV, $m_{jj}>120$ GeV and $p_{T,j}>25$ GeV. The same analysis studied soft gap activity and compared parton-shower descriptions [8]. This process therefore supplies enough data to validate the pull response, tagging-jet assignment, pileup treatment and generator dependence before interpreting the Higgs result.

The control analysis should apply the same jet, pull and VBF-category definitions as the diphoton analysis, replacing the photons with an opposite-sign, same-flavour dilepton pair near the $Z$-boson mass. Electroweak $Zjj$ is then treated as signal and QCD $Zjj$ as the dominant background.

## Simulation strategy

The following generator setup provides a suitable starting point, subject to explicit version and configuration checks before production:

- **VBF Higgs:** NLO+parton-shower simulation of electroweak $Hjj$ production, retaining the tagging-jet kinematics and colour structure;
- **ggF Higgs:** an NLO+parton-shower $H+2j$ calculation where available, or a merged $H+0,1,2j$ calculation in the heavy-top effective theory supplemented by finite-top-mass corrections in the hard region;
- **continuum diphoton:** merged prompt $\gamma\gamma+$jets samples covering the sideband and VBF phase space;
- **electroweak and QCD $Zjj$:** NLO+parton-shower predictions with consistent dilepton cuts;
- **other Higgs modes and smaller backgrounds:** generated or normalised at their best available perturbative accuracy.

Jet-to-photon fake rates should not be claimed as first-principles Monte Carlo predictions. In a projection, their shapes and normalisations should be anchored to published diphoton analyses or represented through data-driven templates and nuisance parameters.

Because pull is constructed from soft, wide-angle radiation, the parton shower, hadronisation and underlying event are part of the physics prediction rather than technical afterthoughts. At minimum, the analysis should compare:

- the Herwig angular-ordered and dipole showers where applicable;
- a Pythia 8 shower and hadronisation prediction;
- tune, MPI and colour-reconnection variations;
- shower starting-scale and recoil-scheme variations;
- renormalisation and factorisation scales, PDFs and, where relevant, matching or merging scales.

ATLAS has measured jet-pull observables in $t\bar t$ events and found that no tested generator prediction described all measured distributions simultaneously [9]. Generator dependence must therefore be propagated to the expected VBF sensitivity rather than quoted only as a shape comparison.

## Detector and pileup programme

The study should proceed through three clearly separated levels:

1. **particle level:** stable-particle jets, ideal photon/lepton identification and no pileup, used to reproduce the qualitative colour-flow pattern;
2. **particle level with realistic constituent definitions:** charged-only and particle-flow-like pull, finite constituent thresholds and acceptance restrictions;
3. **fast detector simulation:** photon and jet reconstruction, overlap removal, pileup mitigation, tracking acceptance and HL-LHC pileup scenarios.

For each level, record the efficiency of the event selection, the fraction of events with correctly assigned tagging jets, the migration of pull projections, and the change in expected $\mu_{\mathrm{VBF}}$ precision. A detector-level result that preserves only a small part of the particle-level separation can still be useful if that residual information is robust against generator and pileup variations.

## Principal uncertainties

The likelihood model should eventually contain nuisance parameters for:

- hard-process scale and PDF uncertainties;
- ggF jet-multiplicity and finite-top-mass modelling;
- parton-shower, hadronisation, MPI and colour-reconnection modelling;
- jet energy scale and resolution;
- constituent response, tracking efficiency and charged/neutral energy fractions;
- pileup mitigation and pileup-jet rejection;
- photon energy scale, identification and isolation;
- continuum-background shape and fake-photon normalisation;
- limited Monte Carlo statistics in finely binned pull categories.

Correlations between the pull observable, central-jet activity and the standard VBF classifier should be retained in the statistical model.

## Staged work plan

### Stage 0: normalisation and object-definition audit

- verify the generator cuts and cross-section normalisations of the existing VBF and QCD samples;
- resolve the current QCD $Zjj$ normalisation of `0.48297 pb`, which may represent either a missing factor of $10^3$ or undocumented generator cuts;
- introduce explicit photon and lepton objects, isolation and jet overlap removal, since the current stable-Higgs event builder would otherwise cluster Higgs-decay photons into jets;
- store both signed pull-vector components and the existing magnitude/angle representation.

### Stage 1: limited particle-level pilot

- generate modest VBF and ggF $H\to\gamma\gamma$ samples and electroweak/QCD $Z\to\ell\ell$ samples;
- reproduce the existing stable-Higgs pull separation;
- compare leading-$p_T$ and maximum-$m_{jj}$ tagging-jet assignments;
- establish physical weighted cutflows and the three benchmark luminosities.

### Stage 2: shower and constituent validation

- compare at least two shower/hadronisation descriptions;
- repeat the analysis with all-particle, charged-only and particle-flow-like pull;
- identify the regions in $m_{jj}$, $|\Delta\eta_{jj}|$ and jet rapidity where pull provides stable incremental information.

### Stage 3: detector-level projection

- add photon and lepton efficiencies, detector acceptance and pileup;
- construct diphoton mass fits in categories based on kinematics alone and on kinematics plus pull;
- report expected uncertainties on $\mu_{\mathrm{VBF}}$ at $140$, $300$ and $3000\,\mathrm{fb}^{-1}$.

### Stage 4: robustness and interpretation

- propagate generator and detector variations through the profile likelihood;
- use $Z(\ell\ell)jj$ to validate the pull response and constrain modelling uncertainties;
- quantify whether pull improves the VBF measurement after central-jet and gap-activity information is already included.

A useful go/no-go criterion for a full production campaign is a visible improvement in the expected VBF precision that survives changes of shower, hadronisation and constituent definition. A large truth-level AUC that disappears after these variations would not justify a high-statistics detector-level campaign.

## Reproducibility record

For each campaign, record the generator and analysis versions, process cards, model files, PDF set, tune, hard-process scales, matching or merging settings, generation cuts, random seeds, event counts, sums of weights, cross-section source, shower and hadronisation configuration, pileup scenario, jet and constituent definitions, object efficiencies, cutflow and validation target.

## References

1. [CMS Collaboration, *Measurements of Higgs boson production cross sections and couplings in the diphoton decay channel at 13 TeV*, JHEP 07 (2021) 027](https://cms-results.web.cern.ch/cms-results/public-results/publications/HIG-19-015/index.html).
2. [A. Vaillancourt and D. Gillberg, *Identifying VBF Higgs boson production using the jet pull vector observable*](https://depot-e.uqtr.ca/id/eprint/8001/1/11_Vaillancourt_Audrey_Affiche.pdf), preliminary truth-level poster.
3. [ATLAS Collaboration, *Measurements of Higgs boson production cross-sections in the $H\to\tau^+\tau^-$ decay channel*, JHEP 08 (2022) 175](https://arxiv.org/abs/2201.08269).
4. [LHC Higgs Cross Section Working Group, ad-interim 13.6 TeV Higgs-production cross sections](https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap).
5. [LHC Higgs Cross Section Working Group, Standard Model Higgs branching ratios](https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR).
6. [A. J. Larkoski, S. Marzani and C. Wu, *Safe Use of Jet Pull*, JHEP 01 (2020) 104](https://arxiv.org/abs/1911.05090).
7. [ATLAS Collaboration, *A new ATLAS for the high-luminosity era*](https://atlas.cern/Updates/Feature/High-Luminosity-ATLAS).
8. [CMS Collaboration, *Electroweak production of two jets in association with a Z boson in proton-proton collisions at 13 TeV*, Eur. Phys. J. C 78 (2018) 589](https://cms-results.web.cern.ch/cms-results/public-results/publications/SMP-16-018/index.html).
9. [ATLAS Collaboration, *Measurement of colour flow using jet-pull observables in $t\bar t$ events at 13 TeV*, Eur. Phys. J. C 78 (2018) 847](https://arxiv.org/abs/1805.02935).
