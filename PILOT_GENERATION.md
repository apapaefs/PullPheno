# Nominal pilot event generation

This document fixes the first small event sample used to develop the PullPheno
analysis.  It is deliberately a generator-validation sample, not yet the
statistics target for the final phenomenology.

## Samples

| Component | Hard process | Decay/final state | Stored cross section |
| --- | --- | --- | --- |
| VBF Hjj | electroweak Hjj with the common Herwig/MG5 VBF approximation | H forced to gamma gamma | inclusive Hjj production |
| ggF Hjj | full five-flavour loop-induced pp to Hjj, including gg, qg and qq channels | H forced to gamma gamma | inclusive Hjj production |
| EW Zjj | electroweak resonant Zjj with the common VBF approximation | Z to e+e- and mu+mu- combined | decay-inclusive |
| QCD Zjj | QCD-induced resonant Zjj | Z to e+e- and mu+mu- combined | decay-inclusive |

For the Higgs samples, the forced decay is not accompanied by a generator
branching-ratio reweighting.  Event-yield predictions must multiply both
generator chains by one common external physical BR(H to gamma gamma).  The
Z samples already include both requested dilepton branching fractions and
must not be multiplied by another Z branching fraction.

The pilot contains two nominal chains:

- Herwig 7.3 nominal with OpenLoops for the loop-induced Hjj process;
- MG5_aMC 3.5.15 plus Pythia 8.317 nominal.

No Herwig pReco override is used, and the Pythia settings contain no colour-
reconnection override.  Future Herwig pReco and Pythia colour-reconnection
tests will be separate, explicitly labelled variations.

## Harmonization choices

The VBF samples retain the approximation already selected by the Herwig cards:
`VBFDiagramsOnly.in` in Herwig and `$$ w+ w- z a` in MG5.  Changing only one
generator to the full electroweak process would create a process-definition
mismatch.

The previous loop-induced Hjj comparison was restricted to gg initial states.
That is useful as a matrix-element diagnostic but incomplete for jet-pull
phenomenology, because qg and qq subprocesses carry distinct jet flavour and
colour structures.  The pilot therefore uses the full pp process in both
generators.  A tiresias probe confirmed that the installed OpenLoops `pphjj2`
library constructs all 176 subprocesses.

The Z samples are resonant on-shell-Z samples.  MG5 uses spin-correlated decay
chains; Herwig uses direct dilepton matrix-element final states with
`OnShellZProduction.in`.  Electroweak and QCD Zjj orders are generated
separately, and their interference is deferred beyond the first pilot.

## Common inputs and generation cuts

- sqrt(s) = 13.6 TeV;
- CT14lo, LHAPDF ID 13200, alpha_s(MZ) = 0.118;
- fixed scales: 125 GeV for Hjj and 91.1876 GeV for Zjj;
- five-flavour, massless-b hard-process scheme;
- parton-level protection cuts: pT > 20 GeV, |y| < 5 and mjj > 20 GeV.

The MG5 parameter cards are aligned to the Herwig G_mu scheme: GF =
1.16637e-5 GeV^-2, mZ = 91.1876 GeV, GammaZ = 2.4952 GeV, mW = 80.377 GeV,
GammaW = 2.085 GeV and the derived 1/alpha_EW(MZ) = 132.168828224.  The
campaign conventions mH = 125 GeV and mt = 173 GeV are applied to both.

These loose cuts are not the analysis selection.  The phenomenological code
will reconstruct anti-kT R=0.4 jets and impose the photon/lepton acceptance,
tagging-jet, mjj and rapidity-gap cuts downstream.

## Production gate

The first smoke campaign should contain 100 events for each of the four
processes and each nominal chain.  Before scale-up, verify:

1. the requested photons or opposite-sign same-flavour leptons are present;
2. ROOT event counts and required hwsim branches are complete;
3. Higgs and Z weight/branching-ratio conventions are respected;
4. cross sections have the expected order of magnitude;
5. basic tagging-jet spectra agree sufficiently between providers to proceed.

After that gate, 20,000 events per process and chain is an appropriate small
code-development sample.  The full loop-induced Hjj integration has a large
fixed setup cost, so it should not be scaled up merely because the requested
event count is small.  The earlier Herwig Hjj sampling/shape discrepancy must
also be retested before any pull-distribution difference is interpreted as
physics.
