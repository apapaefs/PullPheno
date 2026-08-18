#!/usr/bin/env python3
"""Particle-level cut-based, XGBoost and CR-comparison analysis for HwSim events.

The module is deliberately import-safe: ROOT, FastJet and Matplotlib are loaded
only by the command-line execution path.  This keeps the object selection,
normalisation, pull-vector and run-management helpers independently testable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import itertools
import json
import logging
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

import xgboost_root_varfiles_module as xgbtools


ANALYSIS_VERSION = "2.6.0"
ROOT_PASS_CHECKPOINT_VERSION = 1
ROOT_PASS_CHECKPOINT_RELATIVE = Path("checkpoints") / "root-pass"
RUN_LOG_NAME = "run.log"
MZ_GEV = 91.1876
NEUTRINO_IDS = frozenset((12, 14, 16))
ANALYSIS_STRATEGIES = ("cutbased", "xgboost")
PULL_BIN_COUNT = 6
PULL_BIN_EDGES = np.linspace(0.0, math.pi, PULL_BIN_COUNT + 1, dtype=np.float64)
PULL_VALUE_NAMES = ("t_beam", "t_phi", "magnitude", "signed_angle", "zero_magnitude")
PULL_OBSERVABLE_KEYS = (
    "pull_t_beam",
    "pull_t_phi",
    "pull_magnitude",
    "signed_pull_angle",
    "folded_pull_angle",
)
PULL_MOMENT_MODEL = "event_level_two_tagging_jet_all_pull_observables"
SCORE_PULL_BIN_COUNT = 10
SCORE_PULL_MOMENT_MODEL = "event_level_xgboost_score_quantile_by_folded_pull"
COMPARISON_PINV_RCOND = 1.0e-12
COMPARISON_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
COMPARISON_COLORS = (
    "#275DAD",
    "#D05A47",
    "#0D8274",
    "#8A5FBF",
    "#D18B20",
    "#4F6D7A",
    "#B64E75",
    "#4E8A3A",
)

CUTS: Dict[str, float] = {
    "photon_pt_lead_min_gev": 40.0,
    "photon_pt_sublead_min_gev": 30.0,
    "photon_abs_eta_max": 2.5,
    "photon_mass_min_gev": 120.0,
    "photon_mass_max_gev": 130.0,
    "photon_isolation_dr": 0.4,
    "photon_relative_isolation_max": 0.1,
    "lepton_pt_lead_min_gev": 25.0,
    "lepton_pt_sublead_min_gev": 20.0,
    "lepton_abs_eta_max": 2.5,
    "lepton_mass_min_gev": 80.0,
    "lepton_mass_max_gev": 100.0,
    "jet_radius": 0.4,
    "jet_pt_min_gev": 30.0,
    "jet_abs_y_max": 4.5,
    "mjj_min_gev": 400.0,
    "abs_delta_y_min": 2.5,
}

HIGGS_CUTFLOW = (
    "all_events",
    "at_least_two_photons",
    "photon_acceptance",
    "leading_photon_pt",
    "subleading_photon_pt",
    "photon_isolation",
    "diphoton_mass_window",
    "at_least_two_jets",
    "opposite_hemispheres",
    "mjj",
    "delta_yjj",
    "boson_centrality",
)

Z_CUTFLOW = (
    "all_events",
    "at_least_two_leptons",
    "lepton_acceptance",
    "ossf_pair",
    "leading_lepton_pt",
    "subleading_lepton_pt",
    "dilepton_mass_window",
    "at_least_two_jets",
    "opposite_hemispheres",
    "mjj",
    "delta_yjj",
    "boson_centrality",
)

XGBOOST_HIGGS_CUTFLOW = HIGGS_CUTFLOW[:-3] + (
    "xgboost_application_sample",
    "xgboost_score",
)
XGBOOST_Z_CUTFLOW = Z_CUTFLOW[:-3] + (
    "xgboost_application_sample",
    "xgboost_score",
)

CUT_LABELS = {
    "all_events": "All generated events",
    "at_least_two_photons": "At least two photons",
    "photon_acceptance": r"Two photons with |eta| < 2.5",
    "leading_photon_pt": r"Leading photon pT > 40 GeV",
    "subleading_photon_pt": r"Subleading photon pT > 30 GeV",
    "photon_isolation": r"Two isolated photons (Irel < 0.1)",
    "diphoton_mass_window": r"120 < m(gamma gamma) < 130 GeV",
    "at_least_two_leptons": "At least two electrons or muons",
    "lepton_acceptance": r"Two leptons with |eta| < 2.5",
    "ossf_pair": "Opposite-sign, same-flavour pair",
    "leading_lepton_pt": r"Leading lepton pT > 25 GeV",
    "subleading_lepton_pt": r"Subleading lepton pT > 20 GeV",
    "dilepton_mass_window": r"80 < m(ll) < 100 GeV",
    "at_least_two_jets": r"At least two jets (pT > 30 GeV, |y| < 4.5)",
    "opposite_hemispheres": r"Opposite hemispheres (y1 y2 < 0)",
    "mjj": r"mjj > 400 GeV",
    "delta_yjj": r"|Delta yjj| > 2.5",
    "boson_centrality": "Boson between tagging jets",
    "xgboost_application_sample": "XGBoost application sample",
    "xgboost_score": "Frozen XGBoost score requirement",
}


@dataclass(frozen=True)
class FourVector:
    energy: float
    px: float
    py: float
    pz: float

    @property
    def pt(self) -> float:
        return math.hypot(self.px, self.py)

    @property
    def phi(self) -> float:
        return math.atan2(self.py, self.px)

    @property
    def mass2(self) -> float:
        return self.energy * self.energy - self.px * self.px - self.py * self.py - self.pz * self.pz

    @property
    def mass(self) -> float:
        return math.sqrt(max(0.0, self.mass2))

    @property
    def rapidity(self) -> float:
        plus = self.energy + self.pz
        minus = self.energy - self.pz
        if plus <= 0.0 or minus <= 0.0:
            if self.pz > 0.0:
                return math.inf
            if self.pz < 0.0:
                return -math.inf
            return 0.0
        return 0.5 * math.log(plus / minus)

    def __add__(self, other: "FourVector") -> "FourVector":
        return FourVector(
            self.energy + other.energy,
            self.px + other.px,
            self.py + other.py,
            self.pz + other.pz,
        )


@dataclass
class EventParticles:
    energy: np.ndarray
    px: np.ndarray
    py: np.ndarray
    pz: np.ndarray
    pid: np.ndarray
    pt: np.ndarray = field(init=False)
    phi: np.ndarray = field(init=False)
    eta: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        lengths = {len(self.energy), len(self.px), len(self.py), len(self.pz), len(self.pid)}
        if len(lengths) != 1:
            raise ValueError("Particle arrays have inconsistent lengths")
        self.energy = np.asarray(self.energy, dtype=np.float64)
        self.px = np.asarray(self.px, dtype=np.float64)
        self.py = np.asarray(self.py, dtype=np.float64)
        self.pz = np.asarray(self.pz, dtype=np.float64)
        self.pid = np.asarray(self.pid, dtype=np.int64)
        self.pt = np.hypot(self.px, self.py)
        self.phi = np.arctan2(self.py, self.px)
        momentum = np.sqrt(self.pt * self.pt + self.pz * self.pz)
        plus = momentum + self.pz
        minus = momentum - self.pz
        self.eta = np.zeros_like(momentum)
        finite = (plus > 0.0) & (minus > 0.0)
        self.eta[finite] = 0.5 * np.log(plus[finite] / minus[finite])
        self.eta[(~finite) & (self.pz > 0.0)] = math.inf
        self.eta[(~finite) & (self.pz < 0.0)] = -math.inf

    def __len__(self) -> int:
        return len(self.energy)

    def p4(self, index: int) -> FourVector:
        return FourVector(
            float(self.energy[index]),
            float(self.px[index]),
            float(self.py[index]),
            float(self.pz[index]),
        )

    def relative_photon_isolation(self, photon_index: int, radius: float = 0.4) -> float:
        photon_pt = float(self.pt[photon_index])
        if photon_pt <= 0.0:
            return math.inf
        visible = ~np.isin(np.abs(self.pid), tuple(NEUTRINO_IDS))
        visible[photon_index] = False
        delta_eta = self.eta - self.eta[photon_index]
        delta_phi = wrap_delta_phi_array(self.phi - self.phi[photon_index])
        cone = visible & ((delta_eta * delta_eta + delta_phi * delta_phi) < radius * radius)
        return float(np.sum(self.pt[cone], dtype=np.float64) / photon_pt)


@dataclass(frozen=True)
class BosonCandidate:
    leading_index: int
    subleading_index: int
    p4: FourVector
    leading_isolation: Optional[float] = None
    subleading_isolation: Optional[float] = None
    flavour: Optional[str] = None


@dataclass(frozen=True)
class CandidateDecision:
    passed_steps: Tuple[str, ...]
    candidate: Optional[BosonCandidate]


@dataclass(frozen=True)
class VBFDecision:
    passed_steps: Tuple[str, ...]
    mjj: float
    abs_delta_y: float
    zstar: float
    boson_central: bool


@dataclass(frozen=True)
class PullVector:
    t_y: float
    t_phi: float
    t_beam: float
    magnitude: float
    signed_angle: float
    zero_magnitude: bool


@dataclass(frozen=True)
class SampleSpec:
    name: str
    channel: str
    files: Tuple[str, ...]
    cross_section_pb: float
    label: str
    color: str
    stack_order: int
    role: str = "background"
    generator_cross_section_pb: Optional[float] = None
    generator_cross_section_unc_pb: Optional[float] = None
    cross_section_unc_pb: Optional[float] = None
    cross_section_source: Optional[str] = None


@dataclass(frozen=True)
class ScenarioSpec:
    identifier: str
    label: str
    color: Optional[str] = None
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisConfig:
    tree_name: str
    luminosities_fb: Tuple[float, ...]
    samples: Tuple[SampleSpec, ...]
    source_manifest: str
    scenario: Optional[ScenarioSpec] = None


@dataclass(frozen=True)
class PlotSpec:
    key: str
    channel: str
    stage: str
    title: str
    xlabel: str
    edges: np.ndarray


@dataclass
class WeightedHistogram:
    edges: np.ndarray
    sumw: np.ndarray = field(init=False)
    sumw2: np.ndarray = field(init=False)
    entries: int = 0

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=np.float64)
        if self.edges.ndim != 1 or len(self.edges) < 2 or np.any(np.diff(self.edges) <= 0.0):
            raise ValueError("Histogram edges must be a strictly increasing one-dimensional array")
        self.sumw = np.zeros(len(self.edges) - 1, dtype=np.float64)
        self.sumw2 = np.zeros(len(self.edges) - 1, dtype=np.float64)

    def fill(self, value: float, weight: float) -> None:
        value = float(value)
        weight = float(weight)
        if not math.isfinite(value) or not math.isfinite(weight):
            raise ValueError(f"Non-finite histogram fill: value={value}, weight={weight}")
        index = int(np.searchsorted(self.edges, value, side="right") - 1)
        index = min(max(index, 0), len(self.sumw) - 1)
        self.sumw[index] += weight
        self.sumw2[index] += weight * weight
        self.entries += 1

    @property
    def integral(self) -> float:
        return float(np.sum(self.sumw, dtype=np.float64))


@dataclass
class CutStat:
    raw_count: int = 0
    sumw: float = 0.0
    sumw2: float = 0.0

    def fill(self, weight: float) -> None:
        self.raw_count += 1
        self.sumw += float(weight)
        self.sumw2 += float(weight) * float(weight)


@dataclass
class PullObservableMoments:
    edges: np.ndarray
    bin_sumw: np.ndarray = field(init=False)
    event_second_sumw: np.ndarray = field(init=False)
    mc_second_sumw2: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=np.float64)
        if self.edges.ndim != 1 or len(self.edges) < 2 or np.any(np.diff(self.edges) <= 0.0):
            raise ValueError("Pull-observable moment edges must be strictly increasing")
        bins = len(self.edges) - 1
        self.bin_sumw = np.zeros(bins, dtype=np.float64)
        self.event_second_sumw = np.zeros((bins, bins), dtype=np.float64)
        self.mc_second_sumw2 = np.zeros((bins, bins), dtype=np.float64)

    def fill(self, values: Sequence[float], event_weight: float) -> None:
        q_vector = event_histogram_bin_vector(self.edges, values)
        weight = float(event_weight)
        self.bin_sumw += weight * q_vector
        outer = np.outer(q_vector, q_vector)
        self.event_second_sumw += weight * outer
        self.mc_second_sumw2 += weight * weight * outer


@dataclass
class ScorePullMoments:
    """Joint XGBoost-score and folded-pull moments at common selection.

    The score belongs to the event and both half-weight tagging-jet entries
    therefore occupy the same score bin.  The flattened second moments retain
    their correlation for projected-data and finite-MC covariance estimates.
    """

    score_edges: np.ndarray
    pull_edges: np.ndarray = field(default_factory=lambda: PULL_BIN_EDGES.copy())
    bin_sumw: np.ndarray = field(init=False)
    event_second_sumw: np.ndarray = field(init=False)
    mc_second_sumw2: np.ndarray = field(init=False)
    event_count: int = 0

    def __post_init__(self) -> None:
        self.score_edges = np.asarray(self.score_edges, dtype=np.float64)
        self.pull_edges = np.asarray(self.pull_edges, dtype=np.float64)
        for name, edges in (("score", self.score_edges), ("pull", self.pull_edges)):
            if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0.0):
                raise ValueError(f"Joint {name} edges must be strictly increasing")
        shape = (len(self.score_edges) - 1, len(self.pull_edges) - 1)
        flat_bins = shape[0] * shape[1]
        self.bin_sumw = np.zeros(shape, dtype=np.float64)
        self.event_second_sumw = np.zeros((flat_bins, flat_bins), dtype=np.float64)
        self.mc_second_sumw2 = np.zeros((flat_bins, flat_bins), dtype=np.float64)

    def fill_batch(
        self,
        scores: np.ndarray,
        signed_angles: np.ndarray,
        event_weights: np.ndarray,
    ) -> None:
        score_values = np.asarray(scores, dtype=np.float64)
        angle_values = np.asarray(signed_angles, dtype=np.float64)
        weights = np.asarray(event_weights, dtype=np.float64)
        if score_values.ndim != 1 or weights.shape != score_values.shape:
            raise ValueError("Joint score-pull scores and weights must be one-dimensional")
        if angle_values.shape != (len(score_values), 2):
            raise ValueError("Joint score-pull angles must have shape (events, 2)")
        if not (
            np.all(np.isfinite(score_values))
            and np.all(np.isfinite(angle_values))
            and np.all(np.isfinite(weights))
        ):
            raise ValueError("Joint score-pull inputs must be finite")
        if np.any((score_values < 0.0) | (score_values > 1.0)):
            raise ValueError("XGBoost scores must lie in [0, 1]")

        score_bins = np.searchsorted(self.score_edges, score_values, side="right") - 1
        score_bins = np.clip(score_bins, 0, len(self.score_edges) - 2)
        folded = np.abs((angle_values + math.pi) % (2.0 * math.pi) - math.pi)
        pull_bins = np.searchsorted(self.pull_edges, folded, side="right") - 1
        pull_bins = np.clip(pull_bins, 0, len(self.pull_edges) - 2)
        pull_bin_count = len(self.pull_edges) - 1
        flat = score_bins[:, None] * pull_bin_count + pull_bins
        half_weights = 0.5 * weights
        flat_sum = self.bin_sumw.reshape(-1)
        np.add.at(flat_sum, flat[:, 0], half_weights)
        np.add.at(flat_sum, flat[:, 1], half_weights)

        for first, second in ((0, 0), (0, 1), (1, 0), (1, 1)):
            np.add.at(
                self.event_second_sumw,
                (flat[:, first], flat[:, second]),
                0.25 * weights,
            )
            np.add.at(
                self.mc_second_sumw2,
                (flat[:, first], flat[:, second]),
                0.25 * np.square(weights),
            )
        self.event_count += len(score_values)


@dataclass
class SampleResult:
    spec: SampleSpec
    total_entries: int
    generated_sumw: float
    strategy: str = "cutbased"
    application_scope: str = "all_events"
    processed_entries: int = 0
    processed_sumw: float = 0.0
    cutflow: MutableMapping[str, CutStat] = field(default_factory=OrderedDict)
    histograms: MutableMapping[str, WeightedHistogram] = field(default_factory=dict)
    pull_total_sumw: float = 0.0
    pull_beam_sumw: float = 0.0
    pull_left_sumw: float = 0.0
    pull_right_sumw: float = 0.0
    zero_pull_jets: int = 0
    zero_pull_sumw: float = 0.0
    pull_bin_sumw: np.ndarray = field(
        default_factory=lambda: np.zeros(PULL_BIN_COUNT, dtype=np.float64)
    )
    pull_event_second_sumw: np.ndarray = field(
        default_factory=lambda: np.zeros((PULL_BIN_COUNT, PULL_BIN_COUNT), dtype=np.float64)
    )
    pull_mc_second_sumw2: np.ndarray = field(
        default_factory=lambda: np.zeros((PULL_BIN_COUNT, PULL_BIN_COUNT), dtype=np.float64)
    )
    pull_observable_moments: MutableMapping[str, PullObservableMoments] = field(
        default_factory=dict
    )
    pull_moment_model: str = PULL_MOMENT_MODEL
    score_pull_moments: Optional[ScorePullMoments] = None
    score_pull_moment_model: Optional[str] = None
    invalid_events: int = 0
    files: List[Dict[str, Any]] = field(default_factory=list)
    common_events: Optional["CommonEventTable"] = None


@dataclass(frozen=True)
class EventRange:
    """Half-open entry interval in one original manifest ROOT file."""

    source_file_index: int
    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class SampleShard:
    """One deterministic, contiguous interval of a sample's global event order."""

    index: int
    shard_count: int
    global_start: int
    global_stop: int
    ranges: Tuple[EventRange, ...]

    @property
    def event_count(self) -> int:
        return self.global_stop - self.global_start


@dataclass(frozen=True)
class CommonEventTable:
    observable_keys: Tuple[str, ...]
    weights: np.ndarray
    observables: np.ndarray
    pulls: np.ndarray
    source_file_indices: np.ndarray
    source_entries: np.ndarray

    def __len__(self) -> int:
        return len(self.weights)

    def feature_matrix(self) -> np.ndarray:
        from xgboost_root_varfiles_module import FEATURE_NAMES

        indices = [self.observable_keys.index(name) for name in FEATURE_NAMES]
        matrix = self.observables[:, indices]
        if matrix.shape != (len(self), len(FEATURE_NAMES)) or not np.all(np.isfinite(matrix)):
            raise ValueError("Common-event XGBoost feature matrix is invalid")
        return matrix


class CommonEventBuffer:
    """Geometrically growing, compact buffer used during the ROOT stream."""

    def __init__(self, observable_keys: Sequence[str], initial_capacity: int = 4096) -> None:
        self.observable_keys = tuple(observable_keys)
        self.size = 0
        self.capacity = max(1, int(initial_capacity))
        self.weights = np.empty(self.capacity, dtype=np.float64)
        self.observables = np.empty((self.capacity, len(self.observable_keys)), dtype=np.float64)
        self.pulls = np.empty((self.capacity, 2, len(PULL_VALUE_NAMES)), dtype=np.float64)
        self.source_file_indices = np.empty(self.capacity, dtype=np.int32)
        self.source_entries = np.empty(self.capacity, dtype=np.int64)

    def _grow(self) -> None:
        new_capacity = 2 * self.capacity
        self.weights = _grow_array(self.weights, new_capacity)
        self.observables = _grow_array(self.observables, new_capacity)
        self.pulls = _grow_array(self.pulls, new_capacity)
        self.source_file_indices = _grow_array(self.source_file_indices, new_capacity)
        self.source_entries = _grow_array(self.source_entries, new_capacity)
        self.capacity = new_capacity

    def append(
        self,
        weight: float,
        observable_values: Mapping[str, float],
        pulls: Sequence[PullVector],
        source_file_index: int,
        source_entry: int,
    ) -> None:
        if len(pulls) != 2:
            raise ValueError("A common-selected event must contain two tagging-jet pulls")
        if self.size == self.capacity:
            self._grow()
        row = self.size
        values = np.asarray([observable_values[key] for key in self.observable_keys], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Non-finite common-event observable")
        self.weights[row] = float(weight)
        self.observables[row] = values
        for jet_index, pull in enumerate(pulls):
            self.pulls[row, jet_index] = (
                pull.t_beam,
                pull.t_phi,
                pull.magnitude,
                pull.signed_angle,
                float(pull.zero_magnitude),
            )
        self.source_file_indices[row] = int(source_file_index)
        self.source_entries[row] = int(source_entry)
        self.size += 1

    def finalize(self) -> CommonEventTable:
        return CommonEventTable(
            observable_keys=self.observable_keys,
            weights=self.weights[: self.size].copy(),
            observables=self.observables[: self.size].copy(),
            pulls=self.pulls[: self.size].copy(),
            source_file_indices=self.source_file_indices[: self.size].copy(),
            source_entries=self.source_entries[: self.size].copy(),
        )


@dataclass(frozen=True)
class RunReservation:
    run_id: str
    output_root: Path
    runs_root: Path
    incomplete_dir: Path
    final_dir: Path


@dataclass(frozen=True)
class ComparisonSource:
    run_dir: Path
    config: AnalysisConfig
    results: Tuple[SampleResult, ...]
    metadata: Mapping[str, Any]
    scenario: ScenarioSpec
    xgboost_metadata: Optional[Mapping[str, Any]] = None


def _grow_array(array: np.ndarray, new_capacity: int) -> np.ndarray:
    shape = (int(new_capacity),) + array.shape[1:]
    grown = np.empty(shape, dtype=array.dtype)
    grown[: len(array)] = array
    return grown


def histogram_bin_index(edges: np.ndarray, value: float) -> int:
    index = int(np.searchsorted(edges, float(value), side="right") - 1)
    return min(max(index, 0), len(edges) - 2)


def event_histogram_bin_vector(edges: np.ndarray, values: Sequence[float]) -> np.ndarray:
    """Represent two tagging-jet entries as one half-weight event vector."""
    if len(values) != 2:
        raise ValueError("Exactly two tagging-jet observable values are required")
    bin_edges = np.asarray(edges, dtype=np.float64)
    vector = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    for value in values:
        vector[histogram_bin_index(bin_edges, float(value))] += 0.5
    return vector


def pull_event_bin_vector(signed_angles: Sequence[float]) -> np.ndarray:
    if len(signed_angles) != 2:
        raise ValueError("Exactly two tagging-jet pull angles are required")
    return event_histogram_bin_vector(
        PULL_BIN_EDGES,
        [fold_signed_pull_angle(float(angle)) for angle in signed_angles],
    )


def weighted_score_quantile_edges(
    scores: np.ndarray,
    physical_weights: np.ndarray,
    bins: int = SCORE_PULL_BIN_COUNT,
) -> Tuple[np.ndarray, str]:
    """Return frozen score edges with approximately equal nominal yield.

    Tree classifiers can occasionally return too few distinct probabilities
    for strict quantile edges, particularly in tiny tests.  In that case the
    deterministic equal-width fallback keeps the joint observable well-defined
    and records the fallback in metadata.
    """
    values = np.asarray(scores, dtype=np.float64)
    weights = np.asarray(physical_weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape or len(values) == 0:
        raise ValueError("Score-quantile inputs must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("Score-quantile values must be finite and lie in [0, 1]")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Score-quantile weights must be finite and non-negative")
    total = float(np.sum(weights, dtype=np.float64))
    if total <= 0.0:
        raise ValueError("Score-quantile weights must have a positive sum")
    if bins < 2:
        raise ValueError("At least two score bins are required")

    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order], dtype=np.float64)
    targets = total * np.arange(1, bins, dtype=np.float64) / float(bins)
    positions = np.searchsorted(cumulative, targets, side="left")
    positions = np.clip(positions, 0, len(ordered_values) - 1)
    edges = np.concatenate(([0.0], ordered_values[positions], [1.0]))
    if np.all(np.diff(edges) > 0.0):
        return edges, "nominal_total_weighted_quantiles"
    return np.linspace(0.0, 1.0, bins + 1, dtype=np.float64), "equal_width_fallback"


def normalized_fraction_covariance(
    bin_sums: np.ndarray,
    unnormalized_covariance: np.ndarray,
) -> np.ndarray:
    values = np.asarray(bin_sums, dtype=np.float64)
    covariance = np.asarray(unnormalized_covariance, dtype=np.float64)
    if values.shape != (PULL_BIN_COUNT,) or covariance.shape != (PULL_BIN_COUNT, PULL_BIN_COUNT):
        raise ValueError("Unexpected R_i moment dimensions")
    _, result = normalized_binned_prediction(values, covariance)
    return result


def normalized_binned_prediction(
    bin_sums: np.ndarray,
    unnormalized_covariance: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return unit-area bin fractions and their delta-method covariance."""
    values = np.asarray(bin_sums, dtype=np.float64)
    covariance = np.asarray(unnormalized_covariance, dtype=np.float64)
    if values.ndim != 1 or covariance.shape != (len(values), len(values)):
        raise ValueError("Histogram values and covariance dimensions differ")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(covariance)):
        raise ValueError("Histogram normalization inputs must be finite")
    total = float(np.sum(values, dtype=np.float64))
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("Histogram normalization must be finite and positive")
    fractions = values / total
    jacobian = (np.eye(len(values), dtype=np.float64) - fractions[:, None]) / total
    result = jacobian @ covariance @ jacobian.T
    return fractions, 0.5 * (result + result.T)


def differential_pull_statistics(
    bin_cross_sections_pb: np.ndarray,
    event_second_moment_pb: np.ndarray,
    mc_second_moment_pb2: np.ndarray,
    luminosities_fb: Sequence[float],
) -> Dict[str, Any]:
    """Calculate R_i and event-level projected-data/MC covariances."""
    bins = np.asarray(bin_cross_sections_pb, dtype=np.float64)
    event_second = np.asarray(event_second_moment_pb, dtype=np.float64)
    mc_second = np.asarray(mc_second_moment_pb2, dtype=np.float64)
    total_pb = float(np.sum(bins, dtype=np.float64))
    if total_pb <= 0.0:
        return {
            "bin_edges": PULL_BIN_EDGES.tolist(),
            "R": None,
            "f_beam": None,
            "expected_statistical_covariance": {},
            "mc_statistical_covariance": None,
            "f_beam_statistical_error": {},
            "f_beam_mc_statistical_error": None,
        }
    fractions = bins / total_pb
    mc_covariance = normalized_fraction_covariance(bins, mc_second)
    mean_outer = event_second / total_pb
    single_event_covariance = 0.5 * (
        mean_outer - np.outer(fractions, fractions)
        + (mean_outer - np.outer(fractions, fractions)).T
    )
    selector = np.zeros(PULL_BIN_COUNT, dtype=np.float64)
    selector[: PULL_BIN_COUNT // 2] = 1.0
    expected_covariances: Dict[str, List[List[float]]] = {}
    expected_fbeam_errors: Dict[str, float] = {}
    for luminosity in luminosities_fb:
        expected_events = 1000.0 * float(luminosity) * total_pb
        covariance = single_event_covariance / expected_events
        covariance = 0.5 * (covariance + covariance.T)
        expected_covariances[str(float(luminosity))] = covariance.tolist()
        expected_fbeam_errors[str(float(luminosity))] = math.sqrt(
            max(float(selector @ covariance @ selector), 0.0)
        )
    return {
        "bin_edges": PULL_BIN_EDGES.tolist(),
        "bin_cross_sections_pb": bins.tolist(),
        "R": fractions.tolist(),
        "sum_R": float(np.sum(fractions, dtype=np.float64)),
        "f_beam": float(np.sum(fractions[: PULL_BIN_COUNT // 2], dtype=np.float64)),
        "expected_statistical_covariance": expected_covariances,
        "expected_statistical_errors": {
            key: np.sqrt(np.maximum(np.diag(np.asarray(value)), 0.0)).tolist()
            for key, value in expected_covariances.items()
        },
        "mc_statistical_covariance": mc_covariance.tolist(),
        "mc_statistical_errors": np.sqrt(np.maximum(np.diag(mc_covariance), 0.0)).tolist(),
        "f_beam_statistical_error": expected_fbeam_errors,
        "f_beam_mc_statistical_error": math.sqrt(
            max(float(selector @ mc_covariance @ selector), 0.0)
        ),
        "uncertainty_model": "event_level_two_tagging_jet_covariance",
    }


def delta_phi(phi1: float, phi2: float) -> float:
    return float((phi1 - phi2 + math.pi) % (2.0 * math.pi) - math.pi)


def wrap_delta_phi_array(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values) + math.pi) % (2.0 * math.pi) - math.pi


def fold_signed_pull_angle(angle: float) -> float:
    """Fold a signed angle onto [0, pi], identifying +/- theta.

    The endpoint at +/- pi remains pi, while a numerically zero angle remains
    zero.  This is equivalent to acos(cos(angle)) but avoids the loss of
    precision from evaluating the inverse trigonometric functions.
    """
    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("Signed pull angle must be finite")
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    folded = abs(wrapped)
    if math.isclose(folded, math.pi, rel_tol=0.0, abs_tol=1.0e-15):
        return math.pi
    return folded


def fold_symmetric_histogram(histogram: WeightedHistogram) -> WeightedHistogram:
    """Fold a histogram with symmetric edges about zero onto [0, max]."""
    edges = histogram.edges
    bins = len(histogram.sumw)
    if bins % 2 != 0 or not np.allclose(edges, -edges[::-1], rtol=0.0, atol=1.0e-12):
        raise ValueError("Histogram must have an even number of bins symmetric about zero")
    half = bins // 2
    folded = WeightedHistogram(edges[half:].copy())
    folded.sumw = histogram.sumw[half - 1 :: -1] + histogram.sumw[half:]
    folded.sumw2 = histogram.sumw2[half - 1 :: -1] + histogram.sumw2[half:]
    folded.entries = histogram.entries
    return folded


def select_higgs_candidate(particles: EventParticles) -> CandidateDecision:
    passed: List[str] = []
    photons = np.flatnonzero(particles.pid == 22)
    if len(photons) < 2:
        return CandidateDecision(tuple(passed), None)
    passed.append("at_least_two_photons")

    accepted = photons[np.abs(particles.eta[photons]) < CUTS["photon_abs_eta_max"]]
    if len(accepted) < 2:
        return CandidateDecision(tuple(passed), None)
    passed.append("photon_acceptance")
    accepted = accepted[np.argsort(particles.pt[accepted])[::-1]]

    if particles.pt[accepted[0]] <= CUTS["photon_pt_lead_min_gev"]:
        return CandidateDecision(tuple(passed), None)
    passed.append("leading_photon_pt")
    if particles.pt[accepted[1]] <= CUTS["photon_pt_sublead_min_gev"]:
        return CandidateDecision(tuple(passed), None)
    passed.append("subleading_photon_pt")

    pt_qualified = accepted[particles.pt[accepted] > CUTS["photon_pt_sublead_min_gev"]]
    isolated: List[Tuple[int, float]] = []
    for index in pt_qualified:
        isolation = particles.relative_photon_isolation(
            int(index), radius=CUTS["photon_isolation_dr"]
        )
        if isolation < CUTS["photon_relative_isolation_max"]:
            isolated.append((int(index), isolation))
    isolated.sort(key=lambda item: particles.pt[item[0]], reverse=True)
    if (
        len(isolated) < 2
        or particles.pt[isolated[0][0]] <= CUTS["photon_pt_lead_min_gev"]
        or particles.pt[isolated[1][0]] <= CUTS["photon_pt_sublead_min_gev"]
    ):
        return CandidateDecision(tuple(passed), None)
    passed.append("photon_isolation")

    lead_index, lead_iso = isolated[0]
    sublead_index, sublead_iso = isolated[1]
    boson = particles.p4(lead_index) + particles.p4(sublead_index)
    if not (CUTS["photon_mass_min_gev"] < boson.mass < CUTS["photon_mass_max_gev"]):
        return CandidateDecision(tuple(passed), None)
    passed.append("diphoton_mass_window")
    return CandidateDecision(
        tuple(passed),
        BosonCandidate(
            lead_index,
            sublead_index,
            boson,
            leading_isolation=lead_iso,
            subleading_isolation=sublead_iso,
            flavour="gamma gamma",
        ),
    )


def select_z_candidate(particles: EventParticles) -> CandidateDecision:
    passed: List[str] = []
    leptons = np.flatnonzero(np.isin(np.abs(particles.pid), (11, 13)))
    if len(leptons) < 2:
        return CandidateDecision(tuple(passed), None)
    passed.append("at_least_two_leptons")

    accepted = leptons[np.abs(particles.eta[leptons]) < CUTS["lepton_abs_eta_max"]]
    if len(accepted) < 2:
        return CandidateDecision(tuple(passed), None)
    passed.append("lepton_acceptance")

    pairs: List[Tuple[float, int, int, FourVector]] = []
    for position, first in enumerate(accepted[:-1]):
        for second in accepted[position + 1 :]:
            if particles.pid[first] != -particles.pid[second]:
                continue
            pair = particles.p4(int(first)) + particles.p4(int(second))
            pairs.append((abs(pair.mass - MZ_GEV), int(first), int(second), pair))
    if not pairs:
        return CandidateDecision(tuple(passed), None)
    passed.append("ossf_pair")
    _, first, second, boson = min(pairs, key=lambda item: item[0])
    lead_index, sublead_index = sorted((first, second), key=lambda idx: particles.pt[idx], reverse=True)

    if particles.pt[lead_index] <= CUTS["lepton_pt_lead_min_gev"]:
        return CandidateDecision(tuple(passed), None)
    passed.append("leading_lepton_pt")
    if particles.pt[sublead_index] <= CUTS["lepton_pt_sublead_min_gev"]:
        return CandidateDecision(tuple(passed), None)
    passed.append("subleading_lepton_pt")
    if not (CUTS["lepton_mass_min_gev"] < boson.mass < CUTS["lepton_mass_max_gev"]):
        return CandidateDecision(tuple(passed), None)
    passed.append("dilepton_mass_window")
    flavour = "ee" if abs(int(particles.pid[lead_index])) == 11 else "mu mu"
    return CandidateDecision(
        tuple(passed),
        BosonCandidate(lead_index, sublead_index, boson, flavour=flavour),
    )


def evaluate_vbf_selection(jet1: FourVector, jet2: FourVector, boson: FourVector) -> VBFDecision:
    dijet = jet1 + jet2
    delta_y = abs(jet1.rapidity - jet2.rapidity)
    midpoint = 0.5 * (jet1.rapidity + jet2.rapidity)
    zstar = abs(boson.rapidity - midpoint) / delta_y if delta_y > 0.0 else math.inf
    central = min(jet1.rapidity, jet2.rapidity) < boson.rapidity < max(
        jet1.rapidity, jet2.rapidity
    )
    passed: List[str] = []
    if dijet.mass <= CUTS["mjj_min_gev"]:
        return VBFDecision(tuple(passed), dijet.mass, delta_y, zstar, central)
    passed.append("mjj")
    if delta_y <= CUTS["abs_delta_y_min"]:
        return VBFDecision(tuple(passed), dijet.mass, delta_y, zstar, central)
    passed.append("delta_yjj")
    if not central:
        return VBFDecision(tuple(passed), dijet.mass, delta_y, zstar, central)
    passed.append("boson_centrality")
    return VBFDecision(tuple(passed), dijet.mass, delta_y, zstar, central)


def _pseudojet_value(obj: Any, name: str) -> float:
    value = getattr(obj, name)
    return float(value() if callable(value) else value)


def pseudojet_p4(jet: Any) -> FourVector:
    return FourVector(
        _pseudojet_value(jet, "e"),
        _pseudojet_value(jet, "px"),
        _pseudojet_value(jet, "py"),
        _pseudojet_value(jet, "pz"),
    )


def pseudojet_pt(jet: Any) -> float:
    if hasattr(jet, "perp"):
        return _pseudojet_value(jet, "perp")
    return _pseudojet_value(jet, "pt")


def calculate_pull_vector(jet: Any) -> PullVector:
    jet_pt = pseudojet_pt(jet)
    jet_y = _pseudojet_value(jet, "rapidity")
    jet_phi = _pseudojet_value(jet, "phi")
    t_y = 0.0
    t_phi = 0.0
    if jet_pt > 0.0:
        for constituent in jet.constituents():
            constituent_pt = pseudojet_pt(constituent)
            if constituent_pt <= 0.0:
                continue
            dy = _pseudojet_value(constituent, "rapidity") - jet_y
            dphi = delta_phi(_pseudojet_value(constituent, "phi"), jet_phi)
            radius = math.hypot(dy, dphi)
            weight = constituent_pt / jet_pt * radius
            t_y += weight * dy
            t_phi += weight * dphi
    t_beam = math.copysign(1.0, jet_y) * t_y
    magnitude = math.hypot(t_y, t_phi)
    zero = magnitude == 0.0
    signed_angle = math.atan2(t_phi, t_beam) if not zero else 0.0
    return PullVector(t_y, t_phi, t_beam, magnitude, signed_angle, zero)


def normalization_factor(luminosity_fb: float, cross_section_pb: float, generated_sumw: float) -> float:
    if not math.isfinite(generated_sumw) or generated_sumw == 0.0:
        raise ValueError("Generated sum of weights must be finite and non-zero")
    return 1000.0 * float(luminosity_fb) * float(cross_section_pb) / float(generated_sumw)


def compensated_add(total: float, correction: float, value: float) -> Tuple[float, float]:
    """Add one value using Kahan compensated summation."""
    adjusted = float(value) - float(correction)
    updated = float(total) + adjusted
    return updated, (updated - float(total)) - adjusted


def projected_fbeam_statistical_error(
    f_beam: float,
    expected_selected_events: float,
    pull_entries_per_event: float = 2.0,
) -> float:
    """Independent-entry binomial estimate for the projected f_beam error."""
    f_beam = float(f_beam)
    expected_selected_events = float(expected_selected_events)
    pull_entries_per_event = float(pull_entries_per_event)
    if not 0.0 <= f_beam <= 1.0:
        raise ValueError("f_beam must lie between zero and one")
    if expected_selected_events <= 0.0 or not math.isfinite(expected_selected_events):
        raise ValueError("Expected selected-event yield must be finite and positive")
    if pull_entries_per_event <= 0.0 or not math.isfinite(pull_entries_per_event):
        raise ValueError("Pull entries per event must be finite and positive")
    return math.sqrt(
        f_beam * (1.0 - f_beam) / (pull_entries_per_event * expected_selected_events)
    )


def weighted_fraction_mc_error(
    numerator_sumw: float,
    denominator_sumw: float,
    numerator_sumw2: float,
    complement_sumw2: float,
) -> float:
    """Propagate weighted MC sumw2 for a fraction B/(B + O)."""
    numerator_sumw = float(numerator_sumw)
    denominator_sumw = float(denominator_sumw)
    numerator_sumw2 = float(numerator_sumw2)
    complement_sumw2 = float(complement_sumw2)
    if denominator_sumw <= 0.0 or not math.isfinite(denominator_sumw):
        raise ValueError("Fraction denominator must be finite and positive")
    if numerator_sumw2 < 0.0 or complement_sumw2 < 0.0:
        raise ValueError("Category sumw2 values must be non-negative")
    fraction = numerator_sumw / denominator_sumw
    variance = (
        (1.0 - fraction) ** 2 * numerator_sumw2
        + fraction * fraction * complement_sumw2
    ) / (denominator_sumw * denominator_sumw)
    return math.sqrt(max(variance, 0.0))


def _linspace(low: float, high: float, bins: int) -> np.ndarray:
    return np.linspace(low, high, bins + 1, dtype=np.float64)


def plot_registry() -> Tuple[PlotSpec, ...]:
    specs: List[PlotSpec] = []
    for channel, object_name, mass_label, mass_range in (
        ("higgs", "photon", r"$m_{\gamma\gamma}$ [GeV]", (120.0, 130.0)),
        ("z", "lepton", r"$m_{\ell\ell}$ [GeV]", (80.0, 100.0)),
    ):
        specs.extend(
            [
                PlotSpec("leading_object_pt", channel, "common", f"Leading {object_name} transverse momentum", rf"Leading {object_name} $p_T$ [GeV]", _linspace(0.0, 500.0, 40)),
                PlotSpec("subleading_object_pt", channel, "common", f"Subleading {object_name} transverse momentum", rf"Subleading {object_name} $p_T$ [GeV]", _linspace(0.0, 400.0, 40)),
                PlotSpec("leading_object_eta", channel, "common", f"Leading {object_name} pseudorapidity", rf"Leading {object_name} $\eta$", _linspace(-2.5, 2.5, 30)),
                PlotSpec("subleading_object_eta", channel, "common", f"Subleading {object_name} pseudorapidity", rf"Subleading {object_name} $\eta$", _linspace(-2.5, 2.5, 30)),
                PlotSpec("boson_mass", channel, "common", "Reconstructed boson mass", mass_label, _linspace(mass_range[0], mass_range[1], 20)),
                PlotSpec("boson_pt", channel, "common", "Reconstructed boson transverse momentum", r"Boson $p_T$ [GeV]", _linspace(0.0, 600.0, 40)),
                PlotSpec("boson_y", channel, "common", "Reconstructed boson rapidity", r"Boson $y$", _linspace(-5.0, 5.0, 40)),
                PlotSpec("n_jets", channel, "common", "Selected jet multiplicity", r"Number of jets", np.arange(1.5, 11.5, 1.0)),
                PlotSpec("leading_jet_pt", channel, "common", "Leading tagging-jet transverse momentum", r"Leading jet $p_T$ [GeV]", _linspace(30.0, 800.0, 40)),
                PlotSpec("subleading_jet_pt", channel, "common", "Subleading tagging-jet transverse momentum", r"Subleading jet $p_T$ [GeV]", _linspace(30.0, 600.0, 40)),
                PlotSpec("leading_jet_y", channel, "common", "Leading tagging-jet rapidity", r"Leading jet $y$", _linspace(-4.5, 4.5, 36)),
                PlotSpec("subleading_jet_y", channel, "common", "Subleading tagging-jet rapidity", r"Subleading jet $y$", _linspace(-4.5, 4.5, 36)),
                PlotSpec("mjj", channel, "common", "Tagging-jet invariant mass", r"$m_{jj}$ [GeV]", _linspace(0.0, 4000.0, 40)),
                PlotSpec("abs_delta_yjj", channel, "common", "Tagging-jet rapidity separation", r"$|\Delta y_{jj}|$", _linspace(0.0, 9.0, 36)),
                PlotSpec("zstar", channel, "common", "Boson centrality", r"$z^*$", _linspace(0.0, 2.0, 40)),
                PlotSpec("pull_t_beam", channel, "vbf", "Beam-oriented pull component", r"$t_{\mathrm{beam}}$", _linspace(-0.03, 0.03, 40)),
                PlotSpec("pull_t_phi", channel, "vbf", "Azimuthal pull component", r"$t_{\phi}$", _linspace(-0.03, 0.03, 40)),
                PlotSpec("pull_magnitude", channel, "vbf", "Pull-vector magnitude", r"$|\vec{t}|$", _linspace(0.0, 0.06, 40)),
                PlotSpec("signed_pull_angle", channel, "vbf", "Signed pull angle", r"Signed pull angle [rad]", _linspace(-math.pi, math.pi, 12)),
                PlotSpec("folded_pull_angle", channel, "vbf", "Absolute signed pull angle", r"$|\theta_s|$ [rad]", _linspace(0.0, math.pi, 6)),
            ]
        )
    specs.append(
        PlotSpec("leading_photon_isolation", "higgs", "common", "Leading-photon relative isolation", r"Leading photon $I_{\mathrm{rel}}$", _linspace(0.0, 0.1, 20))
    )
    specs.append(
        PlotSpec("subleading_photon_isolation", "higgs", "common", "Subleading-photon relative isolation", r"Subleading photon $I_{\mathrm{rel}}$", _linspace(0.0, 0.1, 20))
    )
    keys = [(spec.channel, spec.key) for spec in specs]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate plot keys in registry")
    return tuple(specs)


def cutflow_steps(channel: str, strategy: str = "cutbased") -> Tuple[str, ...]:
    if strategy not in ANALYSIS_STRATEGIES:
        raise ValueError(f"Unknown analysis strategy: {strategy}")
    if channel == "higgs":
        return HIGGS_CUTFLOW if strategy == "cutbased" else XGBOOST_HIGGS_CUTFLOW
    if channel == "z":
        return Z_CUTFLOW if strategy == "cutbased" else XGBOOST_Z_CUTFLOW
    raise ValueError(f"Unknown channel: {channel}")


def read_manifest(path: Path, luminosity_override: Optional[Sequence[float]] = None) -> AnalysisConfig:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    tree_name = str(payload.get("tree_name", "Data"))
    luminosities = tuple(float(value) for value in (luminosity_override or payload.get("luminosities_fb", (300.0, 3000.0))))
    if not luminosities or any(value <= 0.0 or not math.isfinite(value) for value in luminosities):
        raise ValueError("Luminosities must be finite positive values")
    scenario: Optional[ScenarioSpec] = None
    raw_scenario = payload.get("scenario")
    if raw_scenario is not None:
        if not isinstance(raw_scenario, Mapping):
            raise ValueError("Manifest scenario metadata must be an object")
        identifier = str(raw_scenario.get("id", "")).strip()
        label = str(raw_scenario.get("label", "")).strip()
        if not identifier or sanitize_run_name(identifier) != identifier:
            raise ValueError("Scenario id must be a non-empty sanitized identifier")
        if not label:
            raise ValueError("Scenario label must be non-empty")
        parameters = raw_scenario.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise ValueError("Scenario parameters must be an object")
        color_value = raw_scenario.get("color")
        scenario = ScenarioSpec(
            identifier=identifier,
            label=label,
            color=None if color_value is None else str(color_value),
            parameters=dict(parameters),
        )

    def optional_number(raw: Mapping[str, Any], key: str) -> Optional[float]:
        value = raw.get(key)
        if value is None:
            return None
        parsed = float(value)
        if parsed < 0.0 or not math.isfinite(parsed):
            raise ValueError(f"Sample metadata {key} must be finite and non-negative")
        return parsed

    samples: List[SampleSpec] = []
    seen_names = set()
    for raw in payload.get("samples", []):
        name = str(raw["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate sample name: {name}")
        seen_names.add(name)
        channel = str(raw["channel"]).lower()
        if channel not in ("higgs", "z"):
            raise ValueError(f"Sample {name} has unsupported channel {channel}")
        raw_files = raw.get("files")
        if raw_files is None and "path" in raw:
            raw_files = [raw["path"]]
        if not raw_files:
            raise ValueError(f"Sample {name} does not define any ROOT files")
        files = tuple(str(Path(value).expanduser()) for value in raw_files)
        cross_section = float(raw["cross_section_pb"])
        if cross_section <= 0.0 or not math.isfinite(cross_section):
            raise ValueError(f"Sample {name} has invalid cross section")
        role = str(raw.get("role", "background")).lower()
        if role not in ("signal", "background"):
            raise ValueError(f"Sample {name} has unsupported XGBoost role {role}")
        samples.append(
            SampleSpec(
                name=name,
                channel=channel,
                role=role,
                files=files,
                cross_section_pb=cross_section,
                label=str(raw.get("label", name)),
                color=str(raw.get("color", "#4C78A8")),
                stack_order=int(raw.get("stack_order", 0)),
                generator_cross_section_pb=optional_number(raw, "generator_cross_section_pb"),
                generator_cross_section_unc_pb=optional_number(
                    raw, "generator_cross_section_unc_pb"
                ),
                cross_section_unc_pb=optional_number(raw, "cross_section_unc_pb"),
                cross_section_source=(
                    None
                    if raw.get("cross_section_source") is None
                    else str(raw["cross_section_source"])
                ),
            )
        )
    if not samples:
        raise ValueError("Manifest contains no samples")
    for channel in ("higgs", "z"):
        if not any(sample.channel == channel for sample in samples):
            raise ValueError(f"Manifest contains no {channel} samples")
    return AnalysisConfig(tree_name, luminosities, tuple(samples), str(path), scenario)


def analysis_config_from_payload(
    configuration: Mapping[str, Any],
    luminosity_override: Optional[Sequence[float]] = None,
) -> AnalysisConfig:
    """Reconstruct a resolved analysis configuration without reopening a manifest."""
    luminosities = tuple(
        float(value)
        for value in (
            luminosity_override
            or configuration.get("luminosities_fb", (300.0, 3000.0))
        )
    )
    if not luminosities or any(
        value <= 0.0 or not math.isfinite(value) for value in luminosities
    ):
        raise ValueError("Luminosities must be finite positive values")

    samples = tuple(
        SampleSpec(
            name=str(raw["name"]),
            channel=str(raw["channel"]),
            files=tuple(str(value) for value in raw["files"]),
            cross_section_pb=float(raw["cross_section_pb"]),
            label=str(raw["label"]),
            color=str(raw["color"]),
            stack_order=int(raw["stack_order"]),
            role=str(raw.get("role", "background")),
            generator_cross_section_pb=(
                None
                if raw.get("generator_cross_section_pb") is None
                else float(raw["generator_cross_section_pb"])
            ),
            generator_cross_section_unc_pb=(
                None
                if raw.get("generator_cross_section_unc_pb") is None
                else float(raw["generator_cross_section_unc_pb"])
            ),
            cross_section_unc_pb=(
                None
                if raw.get("cross_section_unc_pb") is None
                else float(raw["cross_section_unc_pb"])
            ),
            cross_section_source=(
                None
                if raw.get("cross_section_source") is None
                else str(raw["cross_section_source"])
            ),
        )
        for raw in configuration["samples"]
    )
    if not samples:
        raise ValueError("Resolved configuration contains no samples")
    raw_scenario = configuration.get("scenario")
    scenario = (
        None
        if raw_scenario is None
        else ScenarioSpec(
            identifier=str(
                raw_scenario.get("identifier", raw_scenario.get("id", ""))
            ),
            label=str(raw_scenario["label"]),
            color=(
                None
                if raw_scenario.get("color") is None
                else str(raw_scenario["color"])
            ),
            parameters=dict(raw_scenario.get("parameters", {})),
        )
    )
    return AnalysisConfig(
        tree_name=str(configuration.get("tree_name", "Data")),
        luminosities_fb=luminosities,
        samples=samples,
        source_manifest=str(configuration.get("source_manifest", "resolved-configuration")),
        scenario=scenario,
    )


def resolved_config_payload(
    config: AnalysisConfig,
    max_events: Optional[int],
    analyses: Sequence[str] = ("cutbased",),
    xgb_model_run: Optional[Path] = None,
    event_shards_per_sample: int = 1,
) -> Dict[str, Any]:
    return {
        "analysis_version": ANALYSIS_VERSION,
        "tree_name": config.tree_name,
        "luminosities_fb": list(config.luminosities_fb),
        "samples": [asdict(sample) for sample in config.samples],
        "scenario": None if config.scenario is None else asdict(config.scenario),
        "cuts": dict(CUTS),
        "max_events_per_sample": max_events,
        "source_manifest": config.source_manifest,
        "analyses": list(analyses),
        "event_processing": {
            "event_shards_per_sample": int(event_shards_per_sample),
            "partition": "contiguous_global_entry_ranges",
            "merge_order": "manifest_sample_then_global_event_order",
        },
        "xgboost": {
            "model_run": str(xgb_model_run.resolve()) if xgb_model_run is not None else None,
            "feature_names": list(xgbtools.FEATURE_NAMES),
            "model_parameters": dict(xgbtools.MODEL_PARAMETERS),
            "cross_fitting": {
                "folds": xgbtools.CROSS_FIT_FOLDS,
                "seed": xgbtools.CROSS_FIT_SEED,
                "per_pipeline": {
                    "train_folds": xgbtools.CROSS_FIT_FOLDS - 2,
                    "validation_folds": 1,
                    "test_folds": 1,
                },
                "validation_rotation": "(test_fold + 1) modulo folds",
            },
            "nominal_histogram_scope": "five_fold_out_of_fold_all_events",
        }
        if "xgboost" in analyses
        else None,
    }


def sanitize_run_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned[:48] or None


def config_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:8]


def make_run_id(
    payload: Mapping[str, Any],
    run_name: Optional[str] = None,
    now: Optional[datetime] = None,
) -> str:
    timestamp = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    parts = [timestamp]
    safe_name = sanitize_run_name(run_name)
    if safe_name:
        parts.append(safe_name)
    parts.append(config_digest(payload))
    return "-".join(parts)


def reserve_run_directory(output_root: Path, base_run_id: str) -> RunReservation:
    output_root = output_root.resolve()
    runs_root = output_root / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        run_id = base_run_id if suffix == 0 else f"{base_run_id}-{suffix:02d}"
        final_dir = runs_root / run_id
        incomplete_dir = runs_root / f".incomplete-{run_id}"
        if final_dir.exists() or incomplete_dir.exists():
            suffix += 1
            continue
        try:
            incomplete_dir.mkdir()
        except FileExistsError:
            suffix += 1
            continue
        return RunReservation(run_id, output_root, runs_root, incomplete_dir, final_dir)


def write_text_exclusive(path: Path, text_value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text_value)


def write_json_exclusive(path: Path, payload: Any) -> None:
    write_text_exclusive(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_write_text(path: Path, text_value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text_value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text_exclusive(path: Path, text_value: str) -> None:
    """Atomically publish text while retaining exclusive-artifact semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text_value)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def attach_run_file_logger(run_dir: Path, level_name: str) -> logging.FileHandler:
    """Mirror parent-process diagnostics to a file that survives failed runs."""
    path = run_dir / RUN_LOG_NAME
    handler = logging.FileHandler(path, mode="x", encoding="utf-8")
    handler.setLevel(getattr(logging, level_name))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    return handler


def detach_run_file_logger(handler: logging.FileHandler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.flush()
    handler.close()


def write_failure_record(
    reservation: RunReservation,
    started: datetime,
    phase: str,
    error: BaseException,
) -> Optional[Path]:
    """Best-effort structured failure record next to an incomplete run log."""
    destination = reservation.incomplete_dir / "failure.json"
    checkpoint = reservation.incomplete_dir / ROOT_PASS_CHECKPOINT_RELATIVE / "checkpoint.json"
    payload = {
        "status": "failed",
        "run_id": reservation.run_id,
        "started_utc": started.isoformat(),
        "failed_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "root_pass_checkpoint_available": checkpoint.is_file(),
        "root_pass_checkpoint": (
            str(ROOT_PASS_CHECKPOINT_RELATIVE) if checkpoint.is_file() else None
        ),
        "run_log": RUN_LOG_NAME,
    }
    try:
        atomic_write_text_exclusive(
            destination,
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
    except Exception:
        logging.exception("Could not write structured failure record %s", destination)
        return None
    return destination


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Durably publish an uncompressed NPZ without an overwrite window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _root_pass_result_metadata(result: SampleResult) -> Dict[str, Any]:
    return {
        "sample": result.spec.name,
        "total_entries": result.total_entries,
        "generated_sumw": result.generated_sumw,
        "strategy": result.strategy,
        "application_scope": result.application_scope,
        "processed_entries": result.processed_entries,
        "processed_sumw": result.processed_sumw,
        "cutflow": {step: asdict(stat) for step, stat in result.cutflow.items()},
        "histogram_entries": {
            key: histogram.entries for key, histogram in result.histograms.items()
        },
        "pull_total_sumw": result.pull_total_sumw,
        "pull_beam_sumw": result.pull_beam_sumw,
        "pull_left_sumw": result.pull_left_sumw,
        "pull_right_sumw": result.pull_right_sumw,
        "zero_pull_jets": result.zero_pull_jets,
        "zero_pull_sumw": result.zero_pull_sumw,
        "pull_moment_model": result.pull_moment_model,
        "invalid_events": result.invalid_events,
        "files": result.files,
        "common_event_observable_keys": (
            None
            if result.common_events is None
            else list(result.common_events.observable_keys)
        ),
    }


def _root_pass_result_arrays(result: SampleResult) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for key, histogram in result.histograms.items():
        prefix = f"hist__{key}"
        arrays[f"{prefix}__edges"] = histogram.edges
        arrays[f"{prefix}__sumw"] = histogram.sumw
        arrays[f"{prefix}__sumw2"] = histogram.sumw2
    for key, moments in result.pull_observable_moments.items():
        prefix = f"moment__{key}"
        arrays[f"{prefix}__edges"] = moments.edges
        arrays[f"{prefix}__bin_sumw"] = moments.bin_sumw
        arrays[f"{prefix}__event_second_sumw"] = moments.event_second_sumw
        arrays[f"{prefix}__mc_second_sumw2"] = moments.mc_second_sumw2
    if result.common_events is not None:
        common = result.common_events
        arrays["common__weights"] = common.weights
        arrays["common__observables"] = common.observables
        arrays["common__pulls"] = common.pulls
        arrays["common__source_file_indices"] = common.source_file_indices
        arrays["common__source_entries"] = common.source_entries
    return arrays


def _validate_root_pass_results(
    config: AnalysisConfig,
    results: Sequence[SampleResult],
    analyses: Sequence[str],
) -> None:
    expected_names = [sample.name for sample in config.samples]
    names = [result.spec.name for result in results]
    if names != expected_names or len(names) != len(set(names)):
        raise ValueError(
            f"ROOT-pass result order differs from the resolved manifest: {names}"
        )
    for result in results:
        if result.strategy != "cutbased":
            raise ValueError("ROOT-pass checkpoints may contain only cut-based source results")
        if result.generated_sumw == 0.0 or not math.isfinite(result.generated_sumw):
            raise ValueError(f"Sample {result.spec.name} has invalid generated sum of weights")
        if "xgboost" in analyses and result.common_events is None:
            raise ValueError(
                f"Sample {result.spec.name} lacks common-event records required by XGBoost"
            )


def write_root_pass_checkpoint(
    run_dir: Path,
    configuration: Mapping[str, Any],
    results: Sequence[SampleResult],
    source_run_id: str,
) -> Path:
    """Checkpoint reconstructed events and cut histograms before XGBoost training."""
    config = analysis_config_from_payload(configuration)
    analyses = tuple(str(value) for value in configuration.get("analyses", ("cutbased",)))
    _validate_root_pass_results(config, results, analyses)
    checkpoint_dir = run_dir / ROOT_PASS_CHECKPOINT_RELATIVE
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir()
    samples: List[Dict[str, Any]] = []
    for index, result in enumerate(results):
        archive_name = f"sample-{index:03d}.npz"
        archive_path = checkpoint_dir / archive_name
        logging.info("Checkpointing ROOT-pass result for %s", result.spec.name)
        _write_npz_exclusive(archive_path, _root_pass_result_arrays(result))
        item = _root_pass_result_metadata(result)
        item.update(
            {
                "archive": archive_name,
                "archive_size_bytes": archive_path.stat().st_size,
            }
        )
        samples.append(item)
    payload = {
        "status": "complete",
        "checkpoint_version": ROOT_PASS_CHECKPOINT_VERSION,
        "analysis_version": ANALYSIS_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_id": source_run_id,
        "configuration_hash": config_digest(configuration),
        "configuration": dict(configuration),
        "samples": samples,
    }
    atomic_write_text_exclusive(
        checkpoint_dir / "checkpoint.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    logging.info("Completed ROOT-pass checkpoint at %s", checkpoint_dir)
    return checkpoint_dir


def _restore_root_pass_result(
    spec: SampleSpec,
    metadata: Mapping[str, Any],
    archive_path: Path,
) -> SampleResult:
    if archive_path.stat().st_size != int(metadata["archive_size_bytes"]):
        raise ValueError(f"Checkpoint archive size differs for {spec.name}: {archive_path}")
    strategy = str(metadata.get("strategy", "cutbased"))
    result = initialize_result(
        spec,
        int(metadata["total_entries"]),
        float(metadata["generated_sumw"]),
        strategy=strategy,
    )
    result.application_scope = str(metadata.get("application_scope", "all_events"))
    result.processed_entries = int(metadata["processed_entries"])
    result.processed_sumw = float(metadata["processed_sumw"])
    result.cutflow = OrderedDict(
        (step, CutStat(**metadata["cutflow"][step]))
        for step in cutflow_steps(spec.channel, strategy)
    )
    result.pull_total_sumw = float(metadata["pull_total_sumw"])
    result.pull_beam_sumw = float(metadata["pull_beam_sumw"])
    result.pull_left_sumw = float(metadata["pull_left_sumw"])
    result.pull_right_sumw = float(metadata["pull_right_sumw"])
    result.zero_pull_jets = int(metadata["zero_pull_jets"])
    result.zero_pull_sumw = float(metadata["zero_pull_sumw"])
    result.pull_moment_model = str(metadata["pull_moment_model"])
    result.invalid_events = int(metadata["invalid_events"])
    result.files = list(metadata.get("files", []))

    with np.load(archive_path, allow_pickle=False) as arrays:
        available = set(arrays.files)
        for key, histogram in result.histograms.items():
            prefix = f"hist__{key}"
            required = tuple(
                f"{prefix}__{suffix}" for suffix in ("edges", "sumw", "sumw2")
            )
            if not all(name in available for name in required):
                raise ValueError(f"Checkpoint histogram {key} is missing for {spec.name}")
            edges = np.asarray(arrays[required[0]], dtype=np.float64)
            if not np.array_equal(edges, histogram.edges):
                raise ValueError(f"Checkpoint histogram edges differ for {spec.name} {key}")
            histogram.sumw = np.asarray(arrays[required[1]], dtype=np.float64).copy()
            histogram.sumw2 = np.asarray(arrays[required[2]], dtype=np.float64).copy()
            histogram.entries = int(metadata["histogram_entries"][key])

        restored_moments: Dict[str, PullObservableMoments] = {}
        for key in PULL_OBSERVABLE_KEYS:
            prefix = f"moment__{key}"
            required = {
                suffix: f"{prefix}__{suffix}"
                for suffix in (
                    "edges",
                    "bin_sumw",
                    "event_second_sumw",
                    "mc_second_sumw2",
                )
            }
            if not all(name in available for name in required.values()):
                raise ValueError(f"Checkpoint pull moments {key} are missing for {spec.name}")
            moments = PullObservableMoments(
                np.asarray(arrays[required["edges"]], dtype=np.float64).copy()
            )
            moments.bin_sumw = np.asarray(
                arrays[required["bin_sumw"]], dtype=np.float64
            ).copy()
            moments.event_second_sumw = np.asarray(
                arrays[required["event_second_sumw"]], dtype=np.float64
            ).copy()
            moments.mc_second_sumw2 = np.asarray(
                arrays[required["mc_second_sumw2"]], dtype=np.float64
            ).copy()
            restored_moments[key] = moments
        result.pull_observable_moments = restored_moments
        folded = restored_moments["folded_pull_angle"]
        result.pull_bin_sumw = folded.bin_sumw
        result.pull_event_second_sumw = folded.event_second_sumw
        result.pull_mc_second_sumw2 = folded.mc_second_sumw2

        observable_keys = metadata.get("common_event_observable_keys")
        if observable_keys is not None:
            required = (
                "common__weights",
                "common__observables",
                "common__pulls",
                "common__source_file_indices",
                "common__source_entries",
            )
            if not all(name in available for name in required):
                raise ValueError(f"Checkpoint common-event table is missing for {spec.name}")
            common = CommonEventTable(
                observable_keys=tuple(str(value) for value in observable_keys),
                weights=np.asarray(arrays[required[0]], dtype=np.float64).copy(),
                observables=np.asarray(arrays[required[1]], dtype=np.float64).copy(),
                pulls=np.asarray(arrays[required[2]], dtype=np.float64).copy(),
                source_file_indices=np.asarray(arrays[required[3]], dtype=np.int32).copy(),
                source_entries=np.asarray(arrays[required[4]], dtype=np.int64).copy(),
            )
            if common.observables.shape != (len(common), len(common.observable_keys)):
                raise ValueError(f"Checkpoint observable shape is invalid for {spec.name}")
            if common.pulls.shape != (len(common), 2, len(PULL_VALUE_NAMES)):
                raise ValueError(f"Checkpoint pull shape is invalid for {spec.name}")
            if common.source_file_indices.shape != (len(common),) or common.source_entries.shape != (len(common),):
                raise ValueError(f"Checkpoint source-identity shape is invalid for {spec.name}")
            if not (
                np.all(np.isfinite(common.weights))
                and np.all(np.isfinite(common.observables))
                and np.all(np.isfinite(common.pulls))
            ):
                raise ValueError(f"Checkpoint common-event values are non-finite for {spec.name}")
            result.common_events = common
    return result


def load_root_pass_checkpoint(
    incomplete_run: Path,
    luminosity_override: Optional[Sequence[float]] = None,
) -> Tuple[AnalysisConfig, List[SampleResult], Dict[str, Any]]:
    """Load a complete ROOT-pass checkpoint from a failed immutable run."""
    source = incomplete_run.expanduser().resolve()
    checkpoint_dir = (
        source
        if (source / "checkpoint.json").is_file()
        else source / ROOT_PASS_CHECKPOINT_RELATIVE
    )
    metadata_path = checkpoint_dir / "checkpoint.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Complete ROOT-pass checkpoint does not exist: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"ROOT-pass checkpoint is not complete: {metadata_path}")
    if int(metadata.get("checkpoint_version", -1)) != ROOT_PASS_CHECKPOINT_VERSION:
        raise ValueError(
            f"Unsupported ROOT-pass checkpoint version in {metadata_path}"
        )
    if metadata.get("analysis_version") != ANALYSIS_VERSION:
        raise ValueError(
            "ROOT-pass checkpoint analysis version differs from the running code: "
            f"{metadata.get('analysis_version')} != {ANALYSIS_VERSION}"
        )
    configuration = metadata["configuration"]
    if metadata.get("configuration_hash") != config_digest(configuration):
        raise ValueError(f"ROOT-pass checkpoint configuration hash failed: {metadata_path}")
    config = analysis_config_from_payload(configuration, luminosity_override)
    specs_by_name = {sample.name: sample for sample in config.samples}
    results: List[SampleResult] = []
    for item in metadata["samples"]:
        sample_name = str(item["sample"])
        if sample_name not in specs_by_name:
            raise ValueError(f"Checkpoint references unknown sample {sample_name}")
        archive_path = checkpoint_dir / str(item["archive"])
        if not archive_path.is_file():
            raise FileNotFoundError(f"Checkpoint archive is missing: {archive_path}")
        results.append(_restore_root_pass_result(specs_by_name[sample_name], item, archive_path))
    analyses = tuple(str(value) for value in configuration.get("analyses", ("cutbased",)))
    _validate_root_pass_results(config, results, analyses)
    logging.info(
        "Loaded ROOT-pass checkpoint for %d samples from %s",
        len(results),
        checkpoint_dir,
    )
    return config, results, metadata


def remove_root_pass_checkpoint(run_dir: Path) -> None:
    """Discard the bulky recovery checkpoint only after all final artifacts validate."""
    checkpoint_dir = run_dir / ROOT_PASS_CHECKPOINT_RELATIVE
    if not checkpoint_dir.is_dir():
        return
    shutil.rmtree(checkpoint_dir)
    parent = checkpoint_dir.parent
    if parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def git_provenance(repository: Path) -> Dict[str, Any]:
    def run_git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git",) + arguments,
            cwd=str(repository),
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD") or None,
        "branch": run_git("branch", "--show-current") or None,
        "dirty": bool(status),
        "status": status.splitlines(),
    }


def initialize_result(
    spec: SampleSpec,
    total_entries: int,
    generated_sumw: float,
    strategy: str = "cutbased",
) -> SampleResult:
    relevant_specs = [item for item in plot_registry() if item.channel == spec.channel]
    result = SampleResult(spec, total_entries, generated_sumw, strategy=strategy)
    result.cutflow = OrderedDict(
        (step, CutStat()) for step in cutflow_steps(spec.channel, strategy)
    )
    result.histograms = {item.key: WeightedHistogram(item.edges.copy()) for item in relevant_specs}
    pull_specs = {
        item.key: item
        for item in relevant_specs
        if item.key in PULL_OBSERVABLE_KEYS
    }
    if set(pull_specs) != set(PULL_OBSERVABLE_KEYS):
        raise RuntimeError(f"Incomplete pull-observable registry for {spec.channel}")
    result.pull_observable_moments = {
        key: PullObservableMoments(pull_specs[key].edges.copy())
        for key in PULL_OBSERVABLE_KEYS
    }
    folded = result.pull_observable_moments["folded_pull_angle"]
    result.pull_bin_sumw = folded.bin_sumw
    result.pull_event_second_sumw = folded.event_second_sumw
    result.pull_mc_second_sumw2 = folded.mc_second_sumw2
    return result


def load_completed_run(
    run_dir: Path,
    luminosity_override: Optional[Sequence[float]] = None,
) -> Tuple[AnalysisConfig, List[SampleResult], Dict[str, Any]]:
    """Reconstruct cut-based and XGBoost results from an immutable run."""
    run_dir = run_dir.resolve()
    metadata_path = run_dir / "run.json"
    summary_path = run_dir / "summaries" / "analysis.json"
    histogram_path = run_dir / "summaries" / "histograms.npz"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise ValueError(f"Source run is not complete: {run_dir}")
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    configuration = metadata["configuration"]
    config = analysis_config_from_payload(configuration, luminosity_override)
    samples = config.samples
    specs_by_name = {sample.name: sample for sample in samples}
    results: List[SampleResult] = []
    with np.load(histogram_path, allow_pickle=False) as arrays:
        available = set(arrays.files)
        for source in summary_payload["samples"]:
            sample_name = str(source["sample"]["name"])
            if sample_name not in specs_by_name:
                raise ValueError(f"Source summary references unknown sample {sample_name}")
            spec = specs_by_name[sample_name]
            strategy = str(source.get("strategy", "cutbased"))
            result = initialize_result(
                spec,
                int(source["total_entries"]),
                float(source["generated_sumw"]),
                strategy=strategy,
            )
            result.processed_entries = int(source["processed_entries"])
            result.application_scope = str(source.get("application_scope", "all_events"))
            result.processed_sumw = float(source["processed_sumw"])
            result.invalid_events = int(source.get("invalid_events", 0))
            result.files = list(source.get("files", []))
            result.cutflow = OrderedDict(
                (step, CutStat(**source["cutflow"][step]))
                for step in cutflow_steps(spec.channel, strategy)
            )
            missing = []
            for item in plot_registry():
                if item.channel != spec.channel or item.key == "folded_pull_angle":
                    continue
                prefix = f"{strategy}__{spec.name}__{item.key}"
                legacy_prefix = f"{spec.name}__{item.key}"
                keys = tuple(f"{prefix}__{suffix}" for suffix in ("edges", "sumw", "sumw2"))
                legacy_keys = tuple(
                    f"{legacy_prefix}__{suffix}" for suffix in ("edges", "sumw", "sumw2")
                )
                if not all(key in available for key in keys) and strategy == "cutbased":
                    keys = legacy_keys
                if not all(key in available for key in keys):
                    missing.append(item.key)
                    continue
                histogram = WeightedHistogram(np.asarray(arrays[keys[0]], dtype=np.float64).copy())
                histogram.sumw = np.asarray(arrays[keys[1]], dtype=np.float64).copy()
                histogram.sumw2 = np.asarray(arrays[keys[2]], dtype=np.float64).copy()
                result.histograms[item.key] = histogram
            if missing:
                raise ValueError(f"Source histogram archive for {spec.name} is missing {missing}")
            folded_prefix = f"{strategy}__{spec.name}__folded_pull_angle"
            folded_keys = tuple(
                f"{folded_prefix}__{suffix}" for suffix in ("edges", "sumw", "sumw2")
            )
            legacy_folded_prefix = f"{spec.name}__folded_pull_angle"
            legacy_folded_keys = tuple(
                f"{legacy_folded_prefix}__{suffix}" for suffix in ("edges", "sumw", "sumw2")
            )
            if not all(key in available for key in folded_keys) and strategy == "cutbased":
                folded_keys = legacy_folded_keys
            if all(key in available for key in folded_keys):
                folded = WeightedHistogram(np.asarray(arrays[folded_keys[0]], dtype=np.float64).copy())
                folded.sumw = np.asarray(arrays[folded_keys[1]], dtype=np.float64).copy()
                folded.sumw2 = np.asarray(arrays[folded_keys[2]], dtype=np.float64).copy()
            else:
                folded = fold_symmetric_histogram(result.histograms["signed_pull_angle"])
            result.histograms["folded_pull_angle"] = folded
            pull = source["pull"]
            result.pull_total_sumw = float(pull["sumw"])
            f_beam = pull.get("f_beam")
            result.pull_beam_sumw = 0.0 if f_beam is None else float(f_beam) * result.pull_total_sumw
            result.pull_left_sumw = float(pull["left_sumw"])
            result.pull_right_sumw = float(pull["right_sumw"])
            result.zero_pull_jets = int(pull.get("zero_magnitude_jets", 0))
            result.zero_pull_sumw = float(pull.get("zero_magnitude_sumw", 0.0))
            result_prefix = f"{strategy}__{spec.name}"
            legacy_moment_keys = {
                "bin": f"{result_prefix}__pull_bin_sumw",
                "event": f"{result_prefix}__pull_event_second_sumw",
                "mc": f"{result_prefix}__pull_mc_second_sumw2",
            }
            exact_observables = True
            loaded_moments: Dict[str, PullObservableMoments] = {}
            for observable in PULL_OBSERVABLE_KEYS:
                prefix = f"{result_prefix}__pull_moment__{observable}"
                moment_keys = {
                    "edges": f"{prefix}__edges",
                    "bin": f"{prefix}__bin_sumw",
                    "event": f"{prefix}__event_second_sumw",
                    "mc": f"{prefix}__mc_second_sumw2",
                }
                histogram = result.histograms[observable]
                moments = PullObservableMoments(histogram.edges.copy())
                if all(key in available for key in moment_keys.values()):
                    stored_edges = np.asarray(arrays[moment_keys["edges"]], dtype=np.float64)
                    if not np.array_equal(stored_edges, histogram.edges):
                        raise ValueError(
                            f"Stored pull-moment edges differ for {spec.name} {observable}"
                        )
                    moments.bin_sumw = np.asarray(
                        arrays[moment_keys["bin"]], dtype=np.float64
                    ).copy()
                    moments.event_second_sumw = np.asarray(
                        arrays[moment_keys["event"]], dtype=np.float64
                    ).copy()
                    moments.mc_second_sumw2 = np.asarray(
                        arrays[moment_keys["mc"]], dtype=np.float64
                    ).copy()
                else:
                    exact_observables = False
                    moments.bin_sumw = histogram.sumw.copy()
                    total = float(np.sum(histogram.sumw, dtype=np.float64))
                    fractions = (
                        histogram.sumw / total
                        if total
                        else np.zeros(len(histogram.sumw), dtype=np.float64)
                    )
                    moments.event_second_sumw = total * (
                        0.5 * np.diag(fractions) + 0.5 * np.outer(fractions, fractions)
                    )
                    moments.mc_second_sumw2 = np.diag(histogram.sumw2)
                loaded_moments[observable] = moments

            if all(key in available for key in legacy_moment_keys.values()):
                folded_moments = loaded_moments["folded_pull_angle"]
                folded_moments.bin_sumw = np.asarray(
                    arrays[legacy_moment_keys["bin"]], dtype=np.float64
                ).copy()
                folded_moments.event_second_sumw = np.asarray(
                    arrays[legacy_moment_keys["event"]], dtype=np.float64
                ).copy()
                folded_moments.mc_second_sumw2 = np.asarray(
                    arrays[legacy_moment_keys["mc"]], dtype=np.float64
                ).copy()
            result.pull_observable_moments = loaded_moments
            folded_moments = loaded_moments["folded_pull_angle"]
            result.pull_bin_sumw = folded_moments.bin_sumw
            result.pull_event_second_sumw = folded_moments.event_second_sumw
            result.pull_mc_second_sumw2 = folded_moments.mc_second_sumw2
            result.pull_moment_model = (
                str(pull.get("moment_model", PULL_MOMENT_MODEL))
                if exact_observables
                else "legacy_folded_only_or_independent_jet_reconstruction"
            )
            score_pull_prefix = f"{result_prefix}__score_pull"
            score_pull_keys = {
                "score_edges": f"{score_pull_prefix}__score_edges",
                "pull_edges": f"{score_pull_prefix}__pull_edges",
                "bin": f"{score_pull_prefix}__bin_sumw",
                "event": f"{score_pull_prefix}__event_second_sumw",
                "mc": f"{score_pull_prefix}__mc_second_sumw2",
                "count": f"{score_pull_prefix}__event_count",
            }
            if all(key in available for key in score_pull_keys.values()):
                score_pull = ScorePullMoments(
                    np.asarray(arrays[score_pull_keys["score_edges"]], dtype=np.float64),
                    np.asarray(arrays[score_pull_keys["pull_edges"]], dtype=np.float64),
                )
                expected_shape = score_pull.bin_sumw.shape
                flat_bins = expected_shape[0] * expected_shape[1]
                stored_bin = np.asarray(arrays[score_pull_keys["bin"]], dtype=np.float64)
                stored_event = np.asarray(arrays[score_pull_keys["event"]], dtype=np.float64)
                stored_mc = np.asarray(arrays[score_pull_keys["mc"]], dtype=np.float64)
                if (
                    stored_bin.shape != expected_shape
                    or stored_event.shape != (flat_bins, flat_bins)
                    or stored_mc.shape != (flat_bins, flat_bins)
                ):
                    raise ValueError(
                        f"Stored score-pull moment dimensions are invalid for {spec.name}"
                    )
                score_pull.bin_sumw = stored_bin.copy()
                score_pull.event_second_sumw = stored_event.copy()
                score_pull.mc_second_sumw2 = stored_mc.copy()
                score_pull.event_count = int(arrays[score_pull_keys["count"]])
                result.score_pull_moments = score_pull
                result.score_pull_moment_model = SCORE_PULL_MOMENT_MODEL
            results.append(result)
    return config, results, metadata


def fill_cut_steps(result: SampleResult, steps: Iterable[str], weight: float) -> None:
    for step in steps:
        result.cutflow[step].fill(weight)


def common_observable_values(
    channel: str,
    particles: EventParticles,
    candidate: BosonCandidate,
    selected_jets: Sequence[Any],
    vbf: VBFDecision,
) -> Dict[str, float]:
    jet1, jet2 = selected_jets[:2]
    values = {
        "leading_object_pt": float(particles.pt[candidate.leading_index]),
        "subleading_object_pt": float(particles.pt[candidate.subleading_index]),
        "leading_object_eta": float(particles.eta[candidate.leading_index]),
        "subleading_object_eta": float(particles.eta[candidate.subleading_index]),
        "boson_mass": candidate.p4.mass,
        "boson_pt": candidate.p4.pt,
        "boson_y": candidate.p4.rapidity,
        "n_jets": float(len(selected_jets)),
        "leading_jet_pt": pseudojet_pt(jet1),
        "subleading_jet_pt": pseudojet_pt(jet2),
        "leading_jet_y": _pseudojet_value(jet1, "rapidity"),
        "subleading_jet_y": _pseudojet_value(jet2, "rapidity"),
        "mjj": vbf.mjj,
        "abs_delta_yjj": vbf.abs_delta_y,
        "zstar": vbf.zstar,
    }
    if channel == "higgs":
        values["leading_photon_isolation"] = float(candidate.leading_isolation or 0.0)
        values["subleading_photon_isolation"] = float(candidate.subleading_isolation or 0.0)
    return values


def common_observable_keys(channel: str) -> Tuple[str, ...]:
    return tuple(
        item.key
        for item in plot_registry()
        if item.channel == channel and item.key not in {
            "pull_t_beam",
            "pull_t_phi",
            "pull_magnitude",
            "signed_pull_angle",
            "folded_pull_angle",
        }
    )


def fill_common_histograms_from_values(
    result: SampleResult,
    values: Mapping[str, float],
    weight: float,
) -> None:
    for key, value in values.items():
        result.histograms[key].fill(value, weight)


def fill_common_histograms(
    result: SampleResult,
    particles: EventParticles,
    candidate: BosonCandidate,
    selected_jets: Sequence[Any],
    vbf: VBFDecision,
    weight: float,
) -> Dict[str, float]:
    values = common_observable_values(
        result.spec.channel, particles, candidate, selected_jets, vbf
    )
    fill_common_histograms_from_values(result, values, weight)
    return values


def fill_pull_histograms(result: SampleResult, pulls: Sequence[PullVector], event_weight: float) -> None:
    if len(pulls) != 2:
        raise ValueError("Exactly two tagging-jet pulls are required per selected event")
    observable_values = {
        "pull_t_beam": [pull.t_beam for pull in pulls],
        "pull_t_phi": [pull.t_phi for pull in pulls],
        "pull_magnitude": [pull.magnitude for pull in pulls],
        "signed_pull_angle": [pull.signed_angle for pull in pulls],
        "folded_pull_angle": [
            fold_signed_pull_angle(pull.signed_angle) for pull in pulls
        ],
    }
    for key, values in observable_values.items():
        result.pull_observable_moments[key].fill(values, event_weight)
    for pull in pulls:
        entry_weight = 0.5 * float(event_weight)
        result.histograms["pull_t_beam"].fill(pull.t_beam, entry_weight)
        result.histograms["pull_t_phi"].fill(pull.t_phi, entry_weight)
        result.histograms["pull_magnitude"].fill(pull.magnitude, entry_weight)
        result.histograms["signed_pull_angle"].fill(pull.signed_angle, entry_weight)
        result.histograms["folded_pull_angle"].fill(fold_signed_pull_angle(pull.signed_angle), entry_weight)
        result.pull_total_sumw += entry_weight
        if abs(pull.signed_angle) < 0.5 * math.pi:
            result.pull_beam_sumw += entry_weight
        if pull.signed_angle < 0.0:
            result.pull_left_sumw += entry_weight
        elif pull.signed_angle > 0.0:
            result.pull_right_sumw += entry_weight
        if pull.zero_magnitude:
            result.zero_pull_jets += 1
            result.zero_pull_sumw += entry_weight


def load_runtime() -> Tuple[Any, Any]:
    try:
        import ROOT  # type: ignore
        import fastjet  # type: ignore
    except ImportError as error:
        raise RuntimeError("HwSim analysis requires PyROOT and the FastJet Python bindings") from error
    ROOT.gROOT.SetBatch(True)
    return ROOT, fastjet


def inspect_sample(ROOT: Any, config: AnalysisConfig, spec: SampleSpec) -> Tuple[int, float, List[Dict[str, Any]]]:
    total_entries = 0
    total_sumw = 0.0
    files: List[Dict[str, Any]] = []
    for filename in spec.files:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"ROOT input does not exist: {filename}")
        root_file = ROOT.TFile.Open(str(path), "READ")
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Unable to open ROOT input: {filename}")
        tree = root_file.Get(config.tree_name)
        if tree is None:
            root_file.Close()
            raise RuntimeError(f"Tree {config.tree_name!r} not found in {filename}")
        for branch in ("numparticles", "objects", "evweight"):
            if tree.GetBranch(branch) is None:
                root_file.Close()
                raise RuntimeError(f"Branch {branch!r} not found in {filename}")
        entries = int(tree.GetEntries())
        sumw = float(ROOT.RDataFrame(tree).Sum("evweight").GetValue())
        total_entries += entries
        total_sumw += sumw
        files.append(
            {
                "path": str(path.resolve()),
                "entries": entries,
                "sumw": sumw,
                "size_bytes": path.stat().st_size,
                "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            }
        )
        root_file.Close()
    if total_sumw == 0.0 or not math.isfinite(total_sumw):
        raise RuntimeError(f"Sample {spec.name} has an invalid generated sum of weights: {total_sumw}")
    return total_entries, total_sumw, files


def build_sample_shards(
    file_metadata: Sequence[Mapping[str, Any]],
    max_events: Optional[int],
    requested_shards: int,
) -> Tuple[SampleShard, ...]:
    """Partition the processed prefix into deterministic contiguous event ranges."""
    if requested_shards <= 0:
        raise ValueError("requested_shards must be positive")
    file_entries = [int(item["entries"]) for item in file_metadata]
    if any(entries < 0 for entries in file_entries):
        raise ValueError("ROOT file entry counts cannot be negative")
    total_entries = sum(file_entries)
    processed_entries = (
        total_entries if max_events is None else min(total_entries, int(max_events))
    )
    if processed_entries <= 0:
        raise ValueError("Cannot shard a sample with no events to process")
    shard_count = min(int(requested_shards), processed_entries)
    offsets = np.cumsum(np.asarray([0] + file_entries, dtype=np.int64))
    shards: List[SampleShard] = []
    for index in range(shard_count):
        global_start = processed_entries * index // shard_count
        global_stop = processed_entries * (index + 1) // shard_count
        ranges: List[EventRange] = []
        for file_index, entries in enumerate(file_entries):
            file_global_start = int(offsets[file_index])
            file_global_stop = file_global_start + entries
            overlap_start = max(global_start, file_global_start)
            overlap_stop = min(global_stop, file_global_stop)
            if overlap_start < overlap_stop:
                ranges.append(
                    EventRange(
                        source_file_index=file_index,
                        start=overlap_start - file_global_start,
                        stop=overlap_stop - file_global_start,
                    )
                )
        shard = SampleShard(
            index=index,
            shard_count=shard_count,
            global_start=global_start,
            global_stop=global_stop,
            ranges=tuple(ranges),
        )
        if sum(event_range.count for event_range in shard.ranges) != shard.event_count:
            raise RuntimeError(f"Shard {index} event-range coverage does not close")
        shards.append(shard)
    if sum(shard.event_count for shard in shards) != processed_entries:
        raise RuntimeError("Sample shard event counts do not cover the processed prefix")
    for left, right in zip(shards[:-1], shards[1:]):
        if left.global_stop != right.global_start:
            raise RuntimeError("Sample shard boundaries contain a gap or overlap")
    return tuple(shards)


def merge_sample_shard_results(
    spec: SampleSpec,
    total_entries: int,
    generated_sumw: float,
    file_metadata: Sequence[Mapping[str, Any]],
    shard_results: Sequence[Tuple[SampleShard, SampleResult]],
) -> SampleResult:
    """Merge independently processed ranges while retaining event-level moments."""
    if not shard_results:
        raise ValueError(f"No shard results were supplied for {spec.name}")
    ordered = sorted(shard_results, key=lambda item: item[0].index)
    shards = [item[0] for item in ordered]
    results = [item[1] for item in ordered]
    if [shard.index for shard in shards] != list(range(len(shards))):
        raise ValueError(f"Shard indices for {spec.name} are incomplete or duplicated")
    expected_shard_count = shards[0].shard_count
    if expected_shard_count != len(shards) or any(
        shard.shard_count != expected_shard_count for shard in shards
    ):
        raise ValueError(f"Shard-count metadata differs for {spec.name}")
    for left, right in zip(shards[:-1], shards[1:]):
        if left.global_stop != right.global_start:
            raise ValueError(f"Shard ranges for {spec.name} contain a gap or overlap")
    if any(result.spec != spec or result.strategy != "cutbased" for result in results):
        raise ValueError(f"Shard result identity differs for {spec.name}")
    if any(
        result.total_entries != int(total_entries)
        or not math.isclose(
            result.generated_sumw,
            float(generated_sumw),
            rel_tol=1.0e-13,
            abs_tol=1.0e-13,
        )
        for result in results
    ):
        raise ValueError(f"Shard normalization metadata differs for {spec.name}")

    merged = initialize_result(spec, int(total_entries), float(generated_sumw))
    merged.files = [dict(item) for item in file_metadata]
    scopes = {result.application_scope for result in results}
    if len(scopes) != 1:
        raise ValueError(f"Shard application scopes differ for {spec.name}")
    merged.application_scope = scopes.pop()
    merged.processed_entries = sum(result.processed_entries for result in results)
    merged.processed_sumw = math.fsum(result.processed_sumw for result in results)
    merged.invalid_events = sum(result.invalid_events for result in results)

    for step in merged.cutflow:
        merged.cutflow[step].raw_count = sum(
            result.cutflow[step].raw_count for result in results
        )
        merged.cutflow[step].sumw = math.fsum(
            result.cutflow[step].sumw for result in results
        )
        merged.cutflow[step].sumw2 = math.fsum(
            result.cutflow[step].sumw2 for result in results
        )
    for key, histogram in merged.histograms.items():
        for result in results:
            source = result.histograms[key]
            if not np.array_equal(source.edges, histogram.edges):
                raise ValueError(f"Shard histogram edges differ for {spec.name} {key}")
            histogram.sumw += source.sumw
            histogram.sumw2 += source.sumw2
            histogram.entries += source.entries

    merged.pull_total_sumw = math.fsum(result.pull_total_sumw for result in results)
    merged.pull_beam_sumw = math.fsum(result.pull_beam_sumw for result in results)
    merged.pull_left_sumw = math.fsum(result.pull_left_sumw for result in results)
    merged.pull_right_sumw = math.fsum(result.pull_right_sumw for result in results)
    merged.zero_pull_jets = sum(result.zero_pull_jets for result in results)
    merged.zero_pull_sumw = math.fsum(result.zero_pull_sumw for result in results)
    moment_models = {result.pull_moment_model for result in results}
    if len(moment_models) != 1:
        raise ValueError(f"Shard pull-moment models differ for {spec.name}")
    merged.pull_moment_model = moment_models.pop()
    for key, moments in merged.pull_observable_moments.items():
        for result in results:
            source = result.pull_observable_moments[key]
            if not np.array_equal(source.edges, moments.edges):
                raise ValueError(f"Shard pull-moment edges differ for {spec.name} {key}")
            moments.bin_sumw += source.bin_sumw
            moments.event_second_sumw += source.event_second_sumw
            moments.mc_second_sumw2 += source.mc_second_sumw2
    folded = merged.pull_observable_moments["folded_pull_angle"]
    merged.pull_bin_sumw = folded.bin_sumw
    merged.pull_event_second_sumw = folded.event_second_sumw
    merged.pull_mc_second_sumw2 = folded.mc_second_sumw2

    common_presence = [result.common_events is not None for result in results]
    if any(common_presence) and not all(common_presence):
        raise ValueError(f"Only some shards retained XGBoost events for {spec.name}")
    if all(common_presence):
        tables = [result.common_events for result in results]
        observable_keys = tables[0].observable_keys
        if any(table.observable_keys != observable_keys for table in tables):
            raise ValueError(f"Shard common-event feature order differs for {spec.name}")
        merged.common_events = CommonEventTable(
            observable_keys=observable_keys,
            weights=np.concatenate([table.weights for table in tables]),
            observables=np.concatenate([table.observables for table in tables], axis=0),
            pulls=np.concatenate([table.pulls for table in tables], axis=0),
            source_file_indices=np.concatenate(
                [table.source_file_indices for table in tables]
            ),
            source_entries=np.concatenate([table.source_entries for table in tables]),
        )
        common = merged.common_events
        if len(common) > 1:
            files = common.source_file_indices.astype(np.int64, copy=False)
            entries = common.source_entries
            non_increasing = (files[1:] < files[:-1]) | (
                (files[1:] == files[:-1]) & (entries[1:] <= entries[:-1])
            )
            if np.any(non_increasing):
                raise RuntimeError(
                    f"Merged common-event identities are not ordered for {spec.name}"
                )
    return merged


def particles_from_root(objects: Any, numparticles: int) -> EventParticles:
    array = np.asarray(objects)
    if array.ndim == 1:
        if array.size % 8 != 0:
            raise ValueError(f"Unexpected flattened objects size: {array.size}")
        array = array.reshape((8, array.size // 8))
    if array.ndim != 2 or array.shape[0] < 5 or numparticles > array.shape[1]:
        raise ValueError(f"Unexpected objects shape {array.shape} for {numparticles} particles")
    return EventParticles(
        array[0, :numparticles],
        array[1, :numparticles],
        array[2, :numparticles],
        array[3, :numparticles],
        array[4, :numparticles].astype(np.int64, copy=False),
    )


def cluster_selected_jets(
    particles: EventParticles,
    excluded_indices: Iterable[int],
    fastjet: Any,
    jet_definition: Any,
) -> Tuple[Any, List[Any]]:
    excluded = set(int(value) for value in excluded_indices)
    inputs: List[Any] = []
    for index in range(len(particles)):
        if index in excluded or abs(int(particles.pid[index])) in NEUTRINO_IDS:
            continue
        values = (
            float(particles.px[index]),
            float(particles.py[index]),
            float(particles.pz[index]),
            float(particles.energy[index]),
        )
        if not all(math.isfinite(value) for value in values) or values[3] <= 0.0:
            continue
        pseudojet = fastjet.PseudoJet(values[0], values[1], values[2], values[3])
        pseudojet.set_user_index(index)
        inputs.append(pseudojet)
    cluster = fastjet.ClusterSequence(inputs, jet_definition)
    jets = fastjet.sorted_by_pt(cluster.inclusive_jets(CUTS["jet_pt_min_gev"]))
    selected = [
        jet for jet in jets if abs(_pseudojet_value(jet, "rapidity")) < CUTS["jet_abs_y_max"]
    ]
    return cluster, selected


def analyze_sample_shard(
    ROOT: Any,
    fastjet: Any,
    config: AnalysisConfig,
    spec: SampleSpec,
    total_entries: int,
    generated_sumw: float,
    file_metadata: Sequence[Mapping[str, Any]],
    shard: SampleShard,
    collect_common_events: bool = False,
) -> SampleResult:
    result = initialize_result(spec, total_entries, generated_sumw)
    result.files = [dict(item) for item in file_metadata]
    jet_definition = fastjet.JetDefinition(fastjet.antikt_algorithm, CUTS["jet_radius"])
    common_buffer = (
        CommonEventBuffer(common_observable_keys(spec.channel))
        if collect_common_events
        else None
    )
    processed_sumw_correction = 0.0
    started = time.monotonic()
    shard_label = f"{spec.name} shard {shard.index + 1}/{shard.shard_count}"
    logging.info(
        "%s: entries [%d, %d), %d events; full sample sumw %.12g",
        shard_label,
        shard.global_start,
        shard.global_stop,
        shard.event_count,
        generated_sumw,
    )
    attempted = 0
    report_every = max(1000, shard.event_count // 20) if shard.event_count else 1

    for event_range in shard.ranges:
        source_file_index = event_range.source_file_index
        filename = spec.files[source_file_index]
        root_file = ROOT.TFile.Open(filename, "READ")
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Unable to open ROOT input: {filename}")
        try:
            tree = root_file.Get(config.tree_name)
            if tree is None:
                raise RuntimeError(f"Tree {config.tree_name!r} not found in {filename}")
            entries = int(tree.GetEntries())
            if event_range.start < 0 or event_range.stop > entries:
                raise ValueError(
                    f"{shard_label} range [{event_range.start}, {event_range.stop}) "
                    f"lies outside {filename} with {entries} entries"
                )
            tree.SetBranchStatus("*", 0)
            for branch in ("numparticles", "objects", "evweight"):
                tree.SetBranchStatus(branch, 1)
            for entry in range(event_range.start, event_range.stop):
                tree.GetEntry(entry)
                attempted += 1
                event_weight = float(tree.evweight)
                if not math.isfinite(event_weight):
                    result.invalid_events += 1
                    continue
                try:
                    particles = particles_from_root(tree.objects, int(tree.numparticles))
                    decision = (
                        select_higgs_candidate(particles)
                        if spec.channel == "higgs"
                        else select_z_candidate(particles)
                    )
                    result.processed_entries += 1
                    result.processed_sumw, processed_sumw_correction = compensated_add(
                        result.processed_sumw,
                        processed_sumw_correction,
                        event_weight,
                    )
                    result.cutflow["all_events"].fill(event_weight)
                    fill_cut_steps(result, decision.passed_steps, event_weight)
                    if decision.candidate is None:
                        continue
                    candidate = decision.candidate
                    cluster, jets = cluster_selected_jets(
                        particles,
                        (candidate.leading_index, candidate.subleading_index),
                        fastjet,
                        jet_definition,
                    )
                    if len(jets) < 2:
                        continue
                    result.cutflow["at_least_two_jets"].fill(event_weight)
                    jet1, jet2 = jets[0], jets[1]
                    jet1_y = _pseudojet_value(jet1, "rapidity")
                    jet2_y = _pseudojet_value(jet2, "rapidity")
                    if jet1_y * jet2_y >= 0.0:
                        continue
                    result.cutflow["opposite_hemispheres"].fill(event_weight)
                    vbf = evaluate_vbf_selection(
                        pseudojet_p4(jet1), pseudojet_p4(jet2), candidate.p4
                    )
                    observable_values = fill_common_histograms(
                        result, particles, candidate, jets, vbf, event_weight
                    )
                    pulls: Optional[Tuple[PullVector, PullVector]] = None
                    if common_buffer is not None:
                        pulls = (
                            calculate_pull_vector(jet1),
                            calculate_pull_vector(jet2),
                        )
                        common_buffer.append(
                            event_weight,
                            observable_values,
                            pulls,
                            source_file_index,
                            entry,
                        )
                    fill_cut_steps(result, vbf.passed_steps, event_weight)
                    if len(vbf.passed_steps) != 3:
                        continue
                    if pulls is None:
                        pulls = (
                            calculate_pull_vector(jet1),
                            calculate_pull_vector(jet2),
                        )
                    fill_pull_histograms(result, pulls, event_weight)
                    del cluster
                except (ArithmeticError, ValueError, RuntimeError) as error:
                    result.invalid_events += 1
                    logging.debug(
                        "%s file %d entry %d rejected as invalid: %s",
                        shard_label,
                        source_file_index,
                        entry,
                        error,
                    )
                finally:
                    if attempted and attempted % report_every == 0:
                        logging.info(
                            "%s: processed %d/%d assigned entries",
                            shard_label,
                            attempted,
                            shard.event_count,
                        )
        finally:
            root_file.Close()
    if common_buffer is not None:
        result.common_events = common_buffer.finalize()
        logging.info(
            "%s: retained %d common-selected events for XGBoost",
            shard_label,
            len(result.common_events),
        )
    logging.info(
        "%s complete: %d processed in %.1f s; final sumw %.12g",
        shard_label,
        result.processed_entries,
        time.monotonic() - started,
        result.cutflow[cutflow_steps(spec.channel, result.strategy)[-1]].sumw,
    )
    return result


def analyze_sample(
    ROOT: Any,
    fastjet: Any,
    config: AnalysisConfig,
    spec: SampleSpec,
    max_events: Optional[int],
    collect_common_events: bool = False,
) -> SampleResult:
    """Backward-compatible unsharded sample analysis."""
    total_entries, generated_sumw, file_metadata = inspect_sample(ROOT, config, spec)
    shard = build_sample_shards(file_metadata, max_events, 1)[0]
    return analyze_sample_shard(
        ROOT,
        fastjet,
        config,
        spec,
        total_entries,
        generated_sumw,
        file_metadata,
        shard,
        collect_common_events,
    )


def analyze_sample_worker(
    config: AnalysisConfig,
    spec: SampleSpec,
    max_events: Optional[int],
    log_level: str,
    collect_common_events: bool = False,
) -> SampleResult:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    ROOT, fastjet = load_runtime()
    return analyze_sample(
        ROOT,
        fastjet,
        config,
        spec,
        max_events,
        collect_common_events=collect_common_events,
    )


def inspect_sample_worker(
    config: AnalysisConfig,
    spec: SampleSpec,
    log_level: str,
) -> Tuple[int, float, List[Dict[str, Any]]]:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    ROOT, _ = load_runtime()
    return inspect_sample(ROOT, config, spec)


def analyze_sample_shard_worker(
    config: AnalysisConfig,
    spec: SampleSpec,
    total_entries: int,
    generated_sumw: float,
    file_metadata: Sequence[Mapping[str, Any]],
    shard: SampleShard,
    log_level: str,
    collect_common_events: bool = False,
) -> SampleResult:
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s %(processName)s %(levelname)s %(message)s",
    )
    ROOT, fastjet = load_runtime()
    return analyze_sample_shard(
        ROOT,
        fastjet,
        config,
        spec,
        total_entries,
        generated_sumw,
        file_metadata,
        shard,
        collect_common_events,
    )


def process_root_samples(
    config: AnalysisConfig,
    max_events: Optional[int],
    event_shards_per_sample: int,
    requested_workers: int,
    log_level: str,
    collect_common_events: bool,
) -> Tuple[List[SampleResult], Dict[str, Any], int]:
    """Inspect once, process event ranges concurrently, then merge by event order."""
    if event_shards_per_sample <= 0 or requested_workers <= 0:
        raise ValueError("Event shards and workers must be positive")
    context = multiprocessing.get_context("spawn")
    inspection_by_name: Dict[
        str, Tuple[int, float, List[Dict[str, Any]]]
    ] = {}
    inspection_workers = min(requested_workers, len(config.samples))
    if inspection_workers == 1:
        ROOT, _ = load_runtime()
        for spec in config.samples:
            inspection_by_name[spec.name] = inspect_sample(ROOT, config, spec)
    else:
        logging.info(
            "Inspecting %d samples with %d workers before event sharding",
            len(config.samples),
            inspection_workers,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=inspection_workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(inspect_sample_worker, config, spec, log_level): spec.name
                for spec in config.samples
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                inspection_by_name[name] = future.result()
                logging.info("Inspected sample %s", name)

    plans: Dict[str, Tuple[SampleShard, ...]] = {}
    sample_plan_payload: Dict[str, Any] = {}
    jobs: List[
        Tuple[SampleSpec, int, float, List[Dict[str, Any]], SampleShard]
    ] = []
    for sample_order, spec in enumerate(config.samples):
        total_entries, generated_sumw, file_metadata = inspection_by_name[spec.name]
        shards = build_sample_shards(
            file_metadata,
            max_events,
            event_shards_per_sample,
        )
        plans[spec.name] = shards
        sample_plan_payload[spec.name] = {
            "sample_order": sample_order,
            "total_entries": total_entries,
            "processed_prefix_entries": sum(shard.event_count for shard in shards),
            "requested_shards": event_shards_per_sample,
            "actual_shards": len(shards),
            "shards": [
                {
                    "index": shard.index,
                    "global_start": shard.global_start,
                    "global_stop": shard.global_stop,
                    "event_count": shard.event_count,
                    "file_ranges": [asdict(event_range) for event_range in shard.ranges],
                }
                for shard in shards
            ],
        }
        jobs.extend(
            (spec, total_entries, generated_sumw, file_metadata, shard)
            for shard in shards
        )
    sample_order_by_name = {
        sample.name: index for index, sample in enumerate(config.samples)
    }
    jobs.sort(
        key=lambda item: (
            -item[4].event_count,
            sample_order_by_name[item[0].name],
            item[4].index,
        )
    )
    worker_count = min(requested_workers, len(jobs))
    logging.info(
        "Processing %d event shards from %d samples with %d workers",
        len(jobs),
        len(config.samples),
        worker_count,
    )

    merged_by_name: Dict[str, SampleResult] = {}
    buckets: Dict[str, List[Optional[SampleResult]]] = {
        spec.name: [None] * len(plans[spec.name]) for spec in config.samples
    }

    def accept_result(spec: SampleSpec, shard: SampleShard, result: SampleResult) -> None:
        bucket = buckets[spec.name]
        if bucket[shard.index] is not None:
            raise RuntimeError(f"Duplicate completed shard {spec.name} {shard.index}")
        bucket[shard.index] = result
        logging.info(
            "Collected %s shard %d/%d",
            spec.name,
            shard.index + 1,
            shard.shard_count,
        )
        if all(item is not None for item in bucket):
            total_entries, generated_sumw, file_metadata = inspection_by_name[spec.name]
            pairs = [
                (plans[spec.name][index], item)
                for index, item in enumerate(bucket)
                if item is not None
            ]
            merged_by_name[spec.name] = merge_sample_shard_results(
                spec,
                total_entries,
                generated_sumw,
                file_metadata,
                pairs,
            )
            del buckets[spec.name]
            logging.info(
                "Merged %d ordered event shards for %s",
                len(pairs),
                spec.name,
            )

    if worker_count == 1:
        ROOT, fastjet = load_runtime()
        for spec, total_entries, generated_sumw, file_metadata, shard in jobs:
            result = analyze_sample_shard(
                ROOT,
                fastjet,
                config,
                spec,
                total_entries,
                generated_sumw,
                file_metadata,
                shard,
                collect_common_events,
            )
            accept_result(spec, shard, result)
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=context,
        ) as executor:
            future_jobs = {
                executor.submit(
                    analyze_sample_shard_worker,
                    config,
                    spec,
                    total_entries,
                    generated_sumw,
                    file_metadata,
                    shard,
                    log_level,
                    collect_common_events,
                ): (spec, shard)
                for spec, total_entries, generated_sumw, file_metadata, shard in jobs
            }
            for future in concurrent.futures.as_completed(future_jobs):
                spec, shard = future_jobs.pop(future)
                accept_result(spec, shard, future.result())

    if buckets or set(merged_by_name) != {spec.name for spec in config.samples}:
        raise RuntimeError("Not all event shards were merged into sample results")
    plan_payload = {
        "requested_shards_per_sample": event_shards_per_sample,
        "inspection_workers": inspection_workers,
        "analysis_workers": worker_count,
        "total_shards": len(jobs),
        "samples": sample_plan_payload,
    }
    return [merged_by_name[spec.name] for spec in config.samples], plan_payload, worker_count


def _table_pull_vectors(table: CommonEventTable, row: int) -> Tuple[PullVector, PullVector]:
    vectors: List[PullVector] = []
    for values in table.pulls[int(row)]:
        vectors.append(
            PullVector(
                t_y=0.0,
                t_phi=float(values[1]),
                t_beam=float(values[0]),
                magnitude=float(values[2]),
                signed_angle=float(values[3]),
                zero_magnitude=bool(values[4]),
            )
        )
    return vectors[0], vectors[1]


def _copy_common_cutflow(source: SampleResult, destination: SampleResult) -> None:
    for step in cutflow_steps(source.spec.channel, "cutbased")[:-3]:
        statistic = source.cutflow[step]
        destination.cutflow[step] = CutStat(
            raw_count=statistic.raw_count,
            sumw=statistic.sumw,
            sumw2=statistic.sumw2,
        )


def _split_dataset(
    channel_results: Sequence[SampleResult],
    splits: Mapping[str, xgbtools.SplitIndices],
    split_name: str,
    inverse_probability_correction: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[SampleResult, np.ndarray, float]]]:
    feature_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    selections: List[Tuple[SampleResult, np.ndarray, float]] = []
    for result in channel_results:
        table = result.common_events
        if table is None:
            raise RuntimeError(f"Sample {result.spec.name} has no retained common-event table")
        indices = splits[result.spec.name].as_dict()[split_name]
        if len(indices) == 0:
            raise RuntimeError(f"Sample {result.spec.name} has an empty {split_name} split")
        correction = (
            xgbtools.inverse_split_probability(len(table), len(indices))
            if inverse_probability_correction
            else 1.0
        )
        scale_pb = result.spec.cross_section_pb / result.generated_sumw
        physical_weights = scale_pb * correction * table.weights[indices]
        if np.any(physical_weights <= 0.0) or not np.all(np.isfinite(physical_weights)):
            raise RuntimeError(
                f"Sample {result.spec.name} has non-positive or non-finite XGBoost weights"
            )
        feature_parts.append(table.feature_matrix()[indices])
        label = 1 if result.spec.role == "signal" else 0
        label_parts.append(np.full(len(indices), label, dtype=np.int8))
        weight_parts.append(physical_weights)
        selections.append((result, indices, correction))
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(label_parts),
        np.concatenate(weight_parts),
        selections,
    )


def _all_event_dataset(
    channel_results: Sequence[SampleResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[SampleResult, np.ndarray, float]]]:
    feature_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    weight_parts: List[np.ndarray] = []
    selections: List[Tuple[SampleResult, np.ndarray, float]] = []
    for result in channel_results:
        table = result.common_events
        if table is None or len(table) == 0:
            raise RuntimeError(f"Sample {result.spec.name} has no common-selected XGBoost events")
        indices = np.arange(len(table), dtype=np.int64)
        scale_pb = result.spec.cross_section_pb / result.generated_sumw
        weights = scale_pb * table.weights
        if np.any(weights <= 0.0) or not np.all(np.isfinite(weights)):
            raise RuntimeError(
                f"Sample {result.spec.name} has non-positive or non-finite XGBoost weights"
            )
        feature_parts.append(table.feature_matrix())
        label_parts.append(
            np.full(len(table), 1 if result.spec.role == "signal" else 0, dtype=np.int8)
        )
        weight_parts.append(weights)
        selections.append((result, indices, 1.0))
    return (
        np.concatenate(feature_parts, axis=0),
        np.concatenate(label_parts),
        np.concatenate(weight_parts),
        selections,
    )


def _fill_xgboost_sample_result(
    cut_result: SampleResult,
    indices: np.ndarray,
    scores: np.ndarray,
    threshold: Any,
    correction: float,
) -> SampleResult:
    table = cut_result.common_events
    if table is None or len(indices) != len(scores):
        raise RuntimeError("Invalid XGBoost sample application payload")
    threshold_values = np.asarray(threshold, dtype=np.float64)
    if threshold_values.ndim == 0:
        threshold_values = np.full(len(scores), float(threshold_values), dtype=np.float64)
    if threshold_values.shape != scores.shape or not np.all(np.isfinite(threshold_values)):
        raise RuntimeError("Invalid XGBoost threshold payload")
    result = initialize_result(
        cut_result.spec,
        cut_result.total_entries,
        cut_result.generated_sumw,
        strategy="xgboost",
    )
    result.processed_entries = cut_result.processed_entries
    result.processed_sumw = cut_result.processed_sumw
    result.invalid_events = cut_result.invalid_events
    result.files = list(cut_result.files)
    _copy_common_cutflow(cut_result, result)
    application = result.cutflow["xgboost_application_sample"]
    selected = result.cutflow["xgboost_score"]
    for local_position, source_row in enumerate(indices):
        corrected_weight = correction * float(table.weights[int(source_row)])
        application.fill(corrected_weight)
        if float(scores[local_position]) < float(threshold_values[local_position]):
            continue
        selected.fill(corrected_weight)
        values = {
            key: float(value)
            for key, value in zip(
                table.observable_keys,
                table.observables[int(source_row)],
            )
        }
        fill_common_histograms_from_values(result, values, corrected_weight)
        fill_pull_histograms(
            result,
            _table_pull_vectors(table, int(source_row)),
            corrected_weight,
        )
    return result


def _importance_by_feature(classifier: Any) -> Dict[str, float]:
    raw = classifier.get_booster().get_score(importance_type="gain")
    importance: Dict[str, float] = {}
    for index, name in enumerate(xgbtools.FEATURE_NAMES):
        importance[name] = float(raw.get(f"f{index}", raw.get(name, 0.0)))
    return importance


def _mean_feature_importance(classifiers: Sequence[Any]) -> Dict[str, float]:
    if not classifiers:
        return {name: 0.0 for name in xgbtools.FEATURE_NAMES}
    per_model = [_importance_by_feature(classifier) for classifier in classifiers]
    return {
        name: float(np.mean([values[name] for values in per_model], dtype=np.float64))
        for name in xgbtools.FEATURE_NAMES
    }


def _weighted_confusion_from_pass_mask(
    labels: np.ndarray,
    passed: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    classes = np.asarray(labels, dtype=np.int8)
    selections = np.asarray(passed, dtype=bool)
    physical_weights = np.asarray(weights, dtype=np.float64)
    if classes.shape != selections.shape or classes.shape != physical_weights.shape:
        raise ValueError("Confusion inputs must have identical one-dimensional shapes")
    matrix = np.zeros((2, 2), dtype=np.float64)
    for truth in (0, 1):
        for prediction in (0, 1):
            mask = (classes == truth) & (selections == bool(prediction))
            matrix[truth, prediction] = np.sum(physical_weights[mask], dtype=np.float64)
    return matrix


def _crossfit_split_metadata(
    channel_results: Sequence[SampleResult],
    crossfits: Mapping[str, Sequence[xgbtools.CrossFitSplit]],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for result in channel_results:
        table = result.common_events
        if table is None:
            raise RuntimeError(f"Sample {result.spec.name} has no retained common-event table")
        population = len(table)
        sample_folds = crossfits[result.spec.name]
        test_coverage = np.concatenate([split.test for split in sample_folds])
        validation_coverage = np.concatenate([split.validation for split in sample_folds])
        train_coverage = np.concatenate([split.train for split in sample_folds])
        if not np.array_equal(np.sort(test_coverage), np.arange(population)):
            raise RuntimeError(f"Sample {result.spec.name} cross-fit test coverage is not exhaustive")
        if not np.array_equal(np.sort(validation_coverage), np.arange(population)):
            raise RuntimeError(f"Sample {result.spec.name} cross-fit validation coverage is not exhaustive")
        train_counts = np.bincount(train_coverage, minlength=population)
        if not np.all(train_counts == xgbtools.CROSS_FIT_FOLDS - 2):
            raise RuntimeError(f"Sample {result.spec.name} cross-fit training multiplicity is invalid")
        payload[result.spec.name] = {
            "population": population,
            "test_coverage_count": len(test_coverage),
            "validation_coverage_count": len(validation_coverage),
            "training_multiplicity": xgbtools.CROSS_FIT_FOLDS - 2,
            "folds": [
                {
                    "fold": split.fold,
                    "validation_fold": split.validation_fold,
                    "train_count": len(split.train),
                    "validation_count": len(split.validation),
                    "test_count": len(split.test),
                    "train_fraction": len(split.train) / population,
                    "validation_fraction": len(split.validation) / population,
                    "test_fraction": len(split.test) / population,
                }
                for split in sample_folds
            ],
        }
    return payload


def build_xgboost_results(
    cut_results: Sequence[SampleResult],
    run_dir: Path,
    model_run: Optional[Path] = None,
) -> Tuple[List[SampleResult], Dict[str, Any], Dict[str, Any]]:
    """Train/load channel models and build post-score sample results.

    Nominal training uses rotating five-fold cross-fitting.  In each pipeline,
    three folds train the classifier, a fourth independently fixes its score
    threshold, and the fifth supplies physics entries.  The five test folds
    cover every common-selected event exactly once with no inverse-probability
    correction.  Frozen applications route each independent event through one
    of the same five saved model/threshold pipelines.
    """
    xgboost_results: List[SampleResult] = []
    metadata: Dict[str, Any] = {
        "feature_names": list(xgbtools.FEATURE_NAMES),
        "model_parameters": dict(xgbtools.MODEL_PARAMETERS),
        "runtime_versions": xgbtools.runtime_versions(),
        "training_seed": xgbtools.CROSS_FIT_SEED,
        "cross_fitting": {
            "folds": xgbtools.CROSS_FIT_FOLDS,
            "seed": xgbtools.CROSS_FIT_SEED,
            "scheme": "rotating_nested_60_20_20",
            "validation_rotation": "(test_fold + 1) modulo folds",
            "nominal_event_usage": {
                "physics_test": 1,
                "validation": 1,
                "training": xgbtools.CROSS_FIT_FOLDS - 2,
            },
        },
        "channels": {},
        "source_model_run": str(model_run.resolve()) if model_run is not None else None,
    }
    diagnostics: Dict[str, Any] = {}
    source_metadata: Optional[Mapping[str, Any]] = None
    if model_run is not None:
        source_summary_path = model_run.resolve() / "summaries" / "xgboost.json"
        if not source_summary_path.is_file():
            raise FileNotFoundError(f"Frozen XGBoost summary does not exist: {source_summary_path}")
        source_metadata = json.loads(source_summary_path.read_text(encoding="utf-8"))
        if tuple(source_metadata.get("feature_names", ())) != xgbtools.FEATURE_NAMES:
            raise ValueError("Frozen model feature order does not match this analysis")

    for channel in ("higgs", "z"):
        channel_results = [result for result in cut_results if result.spec.channel == channel]
        roles = {result.spec.role for result in channel_results}
        if roles != {"signal", "background"}:
            raise ValueError(f"Channel {channel} requires both signal and background roles")
        channel_event_count = sum(
            len(result.common_events or ()) for result in channel_results
        )
        logging.info(
            "Starting XGBoost %s channel with %d common-selected events across %d samples (%s)",
            channel,
            channel_event_count,
            len(channel_results),
            "nominal five-fold training" if source_metadata is None else "frozen-model application",
        )
        channel_source = source_metadata["channels"][channel] if source_metadata is not None else None
        source_models = list(channel_source.get("models", ())) if channel_source is not None else []
        if source_models:
            source_crossfit = source_metadata.get("cross_fitting", {})
            if (
                int(source_crossfit.get("folds", -1)) != xgbtools.CROSS_FIT_FOLDS
                or int(source_crossfit.get("seed", -1)) != xgbtools.CROSS_FIT_SEED
            ):
                raise ValueError(
                    "Frozen XGBoost ensemble fold count or assignment seed does not match this analysis"
                )
        use_crossfit = source_metadata is None or bool(source_models)
        crossfits: Dict[str, Tuple[xgbtools.CrossFitSplit, ...]] = {}
        split_metadata: Dict[str, Any] = {}
        if use_crossfit:
            crossfits = {
                result.spec.name: xgbtools.deterministic_crossfit_splits(
                    len(result.common_events or ()),
                    result.spec.name,
                    folds=xgbtools.CROSS_FIT_FOLDS,
                    seed=xgbtools.CROSS_FIT_SEED,
                )
                for result in channel_results
            }
            split_metadata = _crossfit_split_metadata(channel_results, crossfits)

        sample_scores = {
            result.spec.name: np.full(len(result.common_events or ()), np.nan, dtype=np.float64)
            for result in channel_results
        }
        sample_thresholds = {
            result.spec.name: np.full(len(result.common_events or ()), np.nan, dtype=np.float64)
            for result in channel_results
        }
        sample_fold_ids = {
            result.spec.name: np.full(len(result.common_events or ()), -1, dtype=np.int8)
            for result in channel_results
        }
        assigned = {
            result.spec.name: np.zeros(len(result.common_events or ()), dtype=bool)
            for result in channel_results
        }
        classifiers: List[Any] = []
        model_records: List[Dict[str, Any]] = []
        fold_optima: List[xgbtools.ThresholdResult] = []
        training_balances: List[Dict[str, float]] = []
        train_diagnostic_parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        validation_diagnostic_parts: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        def assign_scores(
            fold: int,
            scores: np.ndarray,
            selections: Sequence[Tuple[SampleResult, np.ndarray, float]],
            threshold: float,
        ) -> None:
            offset = 0
            for cut_result, indices, correction in selections:
                if correction != 1.0:
                    raise RuntimeError("Cross-fit physics entries must not use inverse weights")
                count = len(indices)
                values = scores[offset : offset + count]
                offset += count
                name = cut_result.spec.name
                if np.any(assigned[name][indices]):
                    raise RuntimeError(f"Sample {name} received duplicate cross-fit physics scores")
                sample_scores[name][indices] = values
                sample_thresholds[name][indices] = float(threshold)
                sample_fold_ids[name][indices] = int(fold)
                assigned[name][indices] = True
            if offset != len(scores):
                raise RuntimeError("XGBoost per-sample score slicing did not close")

        if source_metadata is None:
            for fold in range(xgbtools.CROSS_FIT_FOLDS):
                fold_splits = {
                    result.spec.name: crossfits[result.spec.name][fold].as_split_indices()
                    for result in channel_results
                }
                train_x, train_y, train_physical_w, _ = _split_dataset(
                    channel_results, fold_splits, "train"
                )
                validation_x, validation_y, validation_w, _ = _split_dataset(
                    channel_results, fold_splits, "validation"
                )
                _, validation_physical_y, validation_physical_w, _ = _split_dataset(
                    channel_results,
                    fold_splits,
                    "validation",
                    inverse_probability_correction=False,
                )
                test_x, test_y, test_w, test_selections = _split_dataset(
                    channel_results,
                    fold_splits,
                    "test",
                    inverse_probability_correction=False,
                )
                balanced_weights, balance = xgbtools.balanced_training_weights(
                    train_physical_w, train_y
                )
                logging.info(
                    "Training XGBoost %s fold %d/%d (train=%d, validation=%d, application=%d)",
                    channel,
                    fold + 1,
                    xgbtools.CROSS_FIT_FOLDS,
                    len(train_x),
                    len(validation_x),
                    len(test_x),
                )
                classifier = xgbtools.train_classifier(train_x, train_y, balanced_weights)
                train_scores = xgbtools.signal_scores(classifier, train_x)
                validation_scores = xgbtools.signal_scores(classifier, validation_x)
                test_scores = xgbtools.signal_scores(classifier, test_x)
                optimum = xgbtools.optimize_significance_threshold(
                    validation_scores, validation_y, validation_w
                )
                assign_scores(fold, test_scores, test_selections, optimum.threshold)

                model_relative = (
                    Path("models") / "xgboost" / channel / f"fold-{fold + 1}.json"
                )
                model_hash = xgbtools.save_classifier(classifier, run_dir / model_relative)
                reloaded = xgbtools.load_classifier(run_dir / model_relative)
                reload_probe = test_x[: min(len(test_x), 10000)]
                reload_identical = np.array_equal(
                    xgbtools.signal_scores(classifier, reload_probe),
                    xgbtools.signal_scores(reloaded, reload_probe),
                )
                if not reload_identical:
                    raise RuntimeError(
                        f"Reloaded {channel} fold {fold + 1} XGBoost model changed its predictions"
                    )
                importance = _importance_by_feature(classifier)
                classifiers.append(classifier)
                fold_optima.append(optimum)
                training_balances.append(balance)
                model_records.append(
                    {
                        "fold": fold,
                        "validation_fold": (fold + 1) % xgbtools.CROSS_FIT_FOLDS,
                        "model_path": model_relative.as_posix(),
                        "model_sha256": model_hash,
                        "reload_predictions_identical": reload_identical,
                        "score_threshold": optimum.threshold,
                        "training_balance": balance,
                        "validation_optimum": {
                            "significance": optimum.significance,
                            "signal_cross_section_pb": optimum.signal_weight,
                            "background_cross_section_pb": optimum.background_weight,
                            "selected_count": optimum.selected_count,
                        },
                        "feature_importance_gain": importance,
                    }
                )
                train_diagnostic_parts.append((train_scores, train_y, train_physical_w))
                if not np.array_equal(validation_y, validation_physical_y):
                    raise RuntimeError("Corrected and physical validation labels differ")
                validation_diagnostic_parts.append(
                    (validation_scores, validation_y, validation_physical_w)
                )
                logging.info(
                    "Completed XGBoost %s fold %d/%d (score cut=%.8g, validation S/sqrt(S+B)=%.8g)",
                    channel,
                    fold + 1,
                    xgbtools.CROSS_FIT_FOLDS,
                    optimum.threshold,
                    optimum.significance,
                )
            application_scope = "five_fold_out_of_fold_all_events"
            validation_signal = float(
                np.mean([optimum.signal_weight for optimum in fold_optima], dtype=np.float64)
            )
            validation_background = float(
                np.mean([optimum.background_weight for optimum in fold_optima], dtype=np.float64)
            )
            validation_significance = (
                validation_signal / math.sqrt(validation_signal + validation_background)
                if validation_signal + validation_background > 0.0
                else 0.0
            )
            validation_summary = {
                "aggregation": "mean cross sections across fold-specific validation optima",
                "significance": validation_significance,
                "signal_cross_section_pb": validation_signal,
                "background_cross_section_pb": validation_background,
                "mean_selected_count": float(
                    np.mean([optimum.selected_count for optimum in fold_optima], dtype=np.float64)
                ),
            }
        elif source_models:
            by_fold = {int(values["fold"]): values for values in source_models}
            expected_folds = set(range(xgbtools.CROSS_FIT_FOLDS))
            if set(by_fold) != expected_folds:
                raise ValueError(
                    f"Frozen {channel} model ensemble must contain folds {sorted(expected_folds)}"
                )
            for fold in range(xgbtools.CROSS_FIT_FOLDS):
                source_record = by_fold[fold]
                logging.info(
                    "Applying frozen XGBoost %s fold %d/%d",
                    channel,
                    fold + 1,
                    xgbtools.CROSS_FIT_FOLDS,
                )
                classifier = xgbtools.load_classifier(
                    model_run.resolve() / str(source_record["model_path"])
                )
                threshold = float(source_record["score_threshold"])
                fold_splits = {
                    result.spec.name: crossfits[result.spec.name][fold].as_split_indices()
                    for result in channel_results
                }
                test_x, test_y, test_w, test_selections = _split_dataset(
                    channel_results,
                    fold_splits,
                    "test",
                    inverse_probability_correction=False,
                )
                test_scores = xgbtools.signal_scores(classifier, test_x)
                assign_scores(fold, test_scores, test_selections, threshold)
                model_relative = (
                    Path("models") / "xgboost" / channel / f"fold-{fold + 1}.json"
                )
                model_hash = xgbtools.save_classifier(classifier, run_dir / model_relative)
                reloaded = xgbtools.load_classifier(run_dir / model_relative)
                reload_probe = test_x[: min(len(test_x), 10000)]
                reload_identical = np.array_equal(
                    xgbtools.signal_scores(classifier, reload_probe),
                    xgbtools.signal_scores(reloaded, reload_probe),
                )
                if not reload_identical:
                    raise RuntimeError(
                        f"Reloaded frozen {channel} fold {fold + 1} model changed its predictions"
                    )
                classifiers.append(classifier)
                model_records.append(
                    {
                        **dict(source_record),
                        "model_path": model_relative.as_posix(),
                        "model_sha256": model_hash,
                        "reload_predictions_identical": reload_identical,
                    }
                )
                logging.info(
                    "Completed frozen XGBoost %s fold %d/%d (application=%d, score cut=%.8g)",
                    channel,
                    fold + 1,
                    xgbtools.CROSS_FIT_FOLDS,
                    len(test_x),
                    threshold,
                )
            training_balances = [
                dict(values.get("training_balance", {})) for values in model_records
            ]
            validation_summary = dict(channel_source["validation_optimum"])
            application_scope = "five_fold_routed_independent_events"
        else:
            # Backward-compatible application of a pre-cross-fit nominal run.
            classifier = xgbtools.load_classifier(
                model_run.resolve() / str(channel_source["model_path"])
            )
            threshold = float(channel_source["score_threshold"])
            all_x, all_y, all_w, all_selections = _all_event_dataset(channel_results)
            all_scores = xgbtools.signal_scores(classifier, all_x)
            offset = 0
            for cut_result, indices, correction in all_selections:
                count = len(indices)
                name = cut_result.spec.name
                sample_scores[name][indices] = all_scores[offset : offset + count]
                sample_thresholds[name][indices] = threshold
                sample_fold_ids[name][indices] = 0
                assigned[name][indices] = True
                offset += count
            if offset != len(all_scores):
                raise RuntimeError("Legacy XGBoost application score slicing did not close")
            model_relative = Path("models") / "xgboost" / f"{channel}.json"
            model_hash = xgbtools.save_classifier(classifier, run_dir / model_relative)
            reloaded = xgbtools.load_classifier(run_dir / model_relative)
            reload_probe = all_x[: min(len(all_x), 10000)]
            reload_identical = np.array_equal(
                xgbtools.signal_scores(classifier, reload_probe),
                xgbtools.signal_scores(reloaded, reload_probe),
            )
            if not reload_identical:
                raise RuntimeError(f"Reloaded legacy {channel} model changed its predictions")
            classifiers = [classifier]
            model_records = []
            validation_summary = dict(channel_source["validation_optimum"])
            training_balances = [dict(channel_source.get("training_balance", {}))]
            application_scope = "legacy_single_model_independent_events"

        for result in channel_results:
            name = result.spec.name
            if not np.all(assigned[name]):
                raise RuntimeError(f"Sample {name} did not receive exactly one physics score per event")
            if not np.all(np.isfinite(sample_scores[name])) or not np.all(
                np.isfinite(sample_thresholds[name])
            ):
                raise RuntimeError(f"Sample {name} received invalid physics scores or thresholds")

        edge_score_parts: List[np.ndarray] = []
        edge_weight_parts: List[np.ndarray] = []
        for cut_result in channel_results:
            table = cut_result.common_events
            if table is None:
                raise RuntimeError(f"Sample {cut_result.spec.name} lost its common-event table")
            edge_score_parts.append(sample_scores[cut_result.spec.name])
            edge_weight_parts.append(
                (cut_result.spec.cross_section_pb / cut_result.generated_sumw) * table.weights
            )
        if source_metadata is None:
            score_pull_edges, score_edge_scheme = weighted_score_quantile_edges(
                np.concatenate(edge_score_parts),
                np.concatenate(edge_weight_parts),
            )
            score_edge_source = "this nominal out-of-fold prediction"
        else:
            source_diagnostic = channel_source.get("score_pull_diagnostic")
            if not source_diagnostic:
                raise ValueError(
                    f"Frozen nominal XGBoost metadata for {channel} lacks score-pull "
                    "quantile edges; rerun the nominal analysis with analysis version 2.3 or later"
                )
            score_pull_edges = np.asarray(
                source_diagnostic.get("score_edges", ()), dtype=np.float64
            )
            if (
                score_pull_edges.shape != (SCORE_PULL_BIN_COUNT + 1,)
                or not np.all(np.diff(score_pull_edges) > 0.0)
            ):
                raise ValueError(f"Frozen {channel} score-pull edges are invalid")
            score_edge_scheme = str(source_diagnostic.get("edge_scheme", "frozen_nominal"))
            score_edge_source = str(model_run.resolve())

        score_parts: List[np.ndarray] = []
        threshold_parts: List[np.ndarray] = []
        label_parts: List[np.ndarray] = []
        weight_parts: List[np.ndarray] = []
        sample_application: Dict[str, Dict[str, Any]] = {}
        for cut_result in channel_results:
            table = cut_result.common_events
            if table is None:
                raise RuntimeError(f"Sample {cut_result.spec.name} lost its common-event table")
            name = cut_result.spec.name
            indices = np.arange(len(table), dtype=np.int64)
            scores = sample_scores[name]
            thresholds = sample_thresholds[name]
            xgb_result = _fill_xgboost_sample_result(
                cut_result, indices, scores, thresholds, correction=1.0
            )
            xgb_result.application_scope = application_scope
            score_pull_moments = ScorePullMoments(score_pull_edges.copy())
            signed_angle_index = PULL_VALUE_NAMES.index("signed_angle")
            score_pull_moments.fill_batch(
                scores,
                table.pulls[:, :, signed_angle_index],
                table.weights,
            )
            xgb_result.score_pull_moments = score_pull_moments
            xgb_result.score_pull_moment_model = SCORE_PULL_MOMENT_MODEL
            xgboost_results.append(xgb_result)
            passed = scores >= thresholds
            sample_application[name] = {
                "input_count": len(indices),
                "selected_count": xgb_result.cutflow["xgboost_score"].raw_count,
                "inverse_probability": 1.0,
                "selected_sumw": xgb_result.cutflow["xgboost_score"].sumw,
                "fold_input_counts": {
                    str(fold): int(np.sum(sample_fold_ids[name] == fold))
                    for fold in sorted(set(int(value) for value in sample_fold_ids[name]))
                },
                "fold_selected_counts": {
                    str(fold): int(np.sum(passed & (sample_fold_ids[name] == fold)))
                    for fold in sorted(set(int(value) for value in sample_fold_ids[name]))
                },
            }
            scale_pb = cut_result.spec.cross_section_pb / cut_result.generated_sumw
            score_parts.append(scores)
            threshold_parts.append(thresholds)
            label_parts.append(
                np.full(len(table), 1 if cut_result.spec.role == "signal" else 0, dtype=np.int8)
            )
            weight_parts.append(scale_pb * table.weights)

        physics_scores = np.concatenate(score_parts)
        physics_thresholds = np.concatenate(threshold_parts)
        physics_y = np.concatenate(label_parts)
        physics_w = np.concatenate(weight_parts)
        physics_passed = physics_scores >= physics_thresholds
        fpr, tpr, _, auc = xgbtools.weighted_roc_curve(
            physics_y, physics_scores, physics_w
        )
        confusion = _weighted_confusion_from_pass_mask(
            physics_y, physics_passed, physics_w
        )
        signal_total = float(np.sum(physics_w[physics_y == 1], dtype=np.float64))
        background_total = float(np.sum(physics_w[physics_y == 0], dtype=np.float64))
        signal_selected = float(confusion[1, 1])
        background_selected = float(confusion[0, 1])
        physics_significance = signal_selected / math.sqrt(signal_selected + background_selected) \
            if signal_selected + background_selected > 0.0 else 0.0
        performance = {
            "scope": application_scope,
            "weighted_auc": auc,
            "signal_efficiency": signal_selected / signal_total if signal_total else None,
            "background_efficiency": background_selected / background_total if background_total else None,
            "signal_cross_section_pb": signal_selected,
            "background_cross_section_pb": background_selected,
            "significance_per_sqrt_fb": physics_significance * math.sqrt(1000.0),
            "weighted_confusion_pb": confusion.tolist(),
        }
        threshold_list = [float(values["score_threshold"]) for values in model_records]
        feature_importance = _mean_feature_importance(classifiers)
        channel_metadata = {
            "application_scope": application_scope,
            "crossfit_split": split_metadata,
            "validation_optimum": validation_summary,
            "out_of_fold" if source_metadata is None else "application": performance,
            "feature_importance_gain": feature_importance,
            "samples": sample_application,
            "score_pull_diagnostic": {
                "moment_model": SCORE_PULL_MOMENT_MODEL,
                "score_bins": SCORE_PULL_BIN_COUNT,
                "score_edges": score_pull_edges.tolist(),
                "pull_bins": PULL_BIN_COUNT,
                "pull_edges": PULL_BIN_EDGES.tolist(),
                "edge_scheme": score_edge_scheme,
                "edge_source": score_edge_source,
                "selection": "common selection through opposite hemispheres; no score cut",
                "entry_model": "two half-weight tagging-jet entries per event",
            },
        }
        if model_records:
            channel_metadata.update(
                {
                    "models": model_records,
                    "model_count": len(model_records),
                    "score_thresholds": threshold_list,
                    "score_threshold_summary": {
                        "minimum": min(threshold_list),
                        "mean": float(np.mean(threshold_list, dtype=np.float64)),
                        "maximum": max(threshold_list),
                    },
                    "reload_predictions_identical": all(
                        bool(values["reload_predictions_identical"])
                        for values in model_records
                    ),
                    "training_balance_by_fold": training_balances,
                }
            )
        else:
            channel_metadata.update(
                {
                    "model_path": model_relative.as_posix(),
                    "model_sha256": model_hash,
                    "score_threshold": threshold,
                    "reload_predictions_identical": reload_identical,
                    "training_balance": training_balances[0],
                }
            )
        metadata["channels"][channel] = channel_metadata
        if source_metadata is None:
            diagnostic_splits = {
                "train": {
                    "scores": np.concatenate([values[0] for values in train_diagnostic_parts]),
                    "labels": np.concatenate([values[1] for values in train_diagnostic_parts]),
                    "weights": np.concatenate([values[2] for values in train_diagnostic_parts]),
                },
                "validation": {
                    "scores": np.concatenate([values[0] for values in validation_diagnostic_parts]),
                    "labels": np.concatenate([values[1] for values in validation_diagnostic_parts]),
                    "weights": np.concatenate([values[2] for values in validation_diagnostic_parts]),
                },
                "out-of-fold": {
                    "scores": physics_scores,
                    "labels": physics_y,
                    "weights": physics_w,
                },
            }
        else:
            diagnostic_splits = {
                "application": {
                    "scores": physics_scores,
                    "labels": physics_y,
                    "weights": physics_w,
                }
            }
        diagnostics[channel] = {
            "thresholds": threshold_list if threshold_list else [float(threshold)],
            "roc_fpr": fpr,
            "roc_tpr": tpr,
            "auc": auc,
            "roc_label": "Five-fold out-of-fold" if source_metadata is None else "Independent application",
            "feature_importance": feature_importance,
            "splits": diagnostic_splits,
        }
        logging.info(
            "Completed XGBoost %s channel (AUC=%.8g, selected signal=%.8g pb, selected background=%.8g pb)",
            channel,
            auc,
            signal_selected,
            background_selected,
        )
    return xgboost_results, metadata, diagnostics


def _format_luminosity(value: float) -> str:
    return f"{value:g}fb"


def result_summary(result: SampleResult, luminosities: Sequence[float]) -> Dict[str, Any]:
    last_step = cutflow_steps(result.spec.channel, result.strategy)[-1]
    selected_sumw = result.cutflow[last_step].sumw
    signed_hist_integral = result.histograms["signed_pull_angle"].integral
    folded_hist_integral = result.histograms["folded_pull_angle"].integral
    pull_match_tolerance = max(1.0e-10, 1.0e-10 * abs(selected_sumw))
    yields = {}
    for luminosity in luminosities:
        factor = normalization_factor(luminosity, result.spec.cross_section_pb, result.generated_sumw)
        yields[str(luminosity)] = {
            "inclusive_expected": 1000.0 * luminosity * result.spec.cross_section_pb,
            "processed_expected": factor * result.processed_sumw,
            "selected_expected": factor * selected_sumw,
        }
    pull_total = result.pull_total_sumw
    scale_pb = result.spec.cross_section_pb / result.generated_sumw
    differential = differential_pull_statistics(
        scale_pb * result.pull_bin_sumw,
        scale_pb * result.pull_event_second_sumw,
        scale_pb * scale_pb * result.pull_mc_second_sumw2,
        luminosities,
    )
    score_pull_summary = None
    if result.score_pull_moments is not None:
        score_pull_summary = {
            "moment_model": result.score_pull_moment_model,
            "event_count": result.score_pull_moments.event_count,
            "score_edges": result.score_pull_moments.score_edges.tolist(),
            "pull_edges": result.score_pull_moments.pull_edges.tolist(),
            "sumw": float(np.sum(result.score_pull_moments.bin_sumw, dtype=np.float64)),
            "selection": "common selection through opposite hemispheres",
        }
    return {
        "strategy": result.strategy,
        "application_scope": result.application_scope,
        "sample": asdict(result.spec),
        "total_entries": result.total_entries,
        "generated_sumw": result.generated_sumw,
        "processed_entries": result.processed_entries,
        "processed_sumw": result.processed_sumw,
        "processed_fraction": result.processed_entries / result.total_entries if result.total_entries else 0.0,
        "invalid_events": result.invalid_events,
        "files": result.files,
        "cutflow": {step: asdict(stat) for step, stat in result.cutflow.items()},
        "pull": {
            "sumw": pull_total,
            "f_beam": differential["f_beam"],
            "left_right_asymmetry": (result.pull_right_sumw - result.pull_left_sumw) / pull_total if pull_total else None,
            "left_sumw": result.pull_left_sumw,
            "right_sumw": result.pull_right_sumw,
            "zero_magnitude_jets": result.zero_pull_jets,
            "zero_magnitude_sumw": result.zero_pull_sumw,
            "signed_angle_histogram_integral": signed_hist_integral,
            "folded_angle_histogram_integral": folded_hist_integral,
            "selected_cutflow_sumw": selected_sumw,
            "integral_matches_selected": abs(signed_hist_integral - selected_sumw) <= pull_match_tolerance,
            "folded_integral_matches_selected": abs(folded_hist_integral - selected_sumw) <= pull_match_tolerance,
            "differential": differential,
            "moment_model": result.pull_moment_model,
        },
        "score_pull_diagnostic": score_pull_summary,
        "yields": yields,
    }


def channel_pull_summary(
    channel: str,
    results: Sequence[SampleResult],
    luminosities: Sequence[float],
    strategy: str = "cutbased",
) -> Dict[str, Any]:
    channel_results = [
        result
        for result in results
        if result.spec.channel == channel and result.strategy == strategy
    ]
    total_pb = 0.0
    left_pb = 0.0
    right_pb = 0.0
    selected_pb = 0.0
    bin_cross_sections_pb = np.zeros(PULL_BIN_COUNT, dtype=np.float64)
    event_second_pb = np.zeros((PULL_BIN_COUNT, PULL_BIN_COUNT), dtype=np.float64)
    mc_second_pb2 = np.zeros((PULL_BIN_COUNT, PULL_BIN_COUNT), dtype=np.float64)
    for result in channel_results:
        scale_pb = result.spec.cross_section_pb / result.generated_sumw
        total_pb += scale_pb * result.pull_total_sumw
        left_pb += scale_pb * result.pull_left_sumw
        right_pb += scale_pb * result.pull_right_sumw
        selected_pb += scale_pb * result.cutflow[
            cutflow_steps(channel, strategy)[-1]
        ].sumw
        bin_cross_sections_pb += scale_pb * result.pull_bin_sumw
        event_second_pb += scale_pb * result.pull_event_second_sumw
        mc_second_pb2 += scale_pb * scale_pb * result.pull_mc_second_sumw2
    differential = differential_pull_statistics(
        bin_cross_sections_pb,
        event_second_pb,
        mc_second_pb2,
        luminosities,
    )
    f_beam = differential["f_beam"]
    expected_yields = {
        str(luminosity): 1000.0 * luminosity * selected_pb for luminosity in luminosities
    }
    return {
        "channel": channel,
        "strategy": strategy,
        "selected_cross_section_pb": selected_pb,
        "pull_histogram_cross_section_pb": total_pb,
        "integral_matches_selected": math.isclose(total_pb, selected_pb, rel_tol=1.0e-10, abs_tol=1.0e-12),
        "f_beam": f_beam,
        "left_right_asymmetry": (right_pb - left_pb) / total_pb if total_pb else None,
        "expected_selected_yields": expected_yields,
        "f_beam_statistical_error": differential["f_beam_statistical_error"],
        "f_beam_mc_statistical_error": differential["f_beam_mc_statistical_error"],
        "statistical_error_model": "event_level_two_tagging_jet_covariance",
        "mc_error_model": "event_level_weighted_outer_products",
        "pull_entries_per_selected_event": 2.0,
        "differential": differential,
    }


def total_pull_observable_statistics(
    channel: str,
    strategy: str,
    observable: str,
    results: Sequence[SampleResult],
    luminosity_fb: float,
    reference_cross_sections_pb: Mapping[str, float],
) -> Dict[str, np.ndarray]:
    """Combine process moments using one reference cross-section convention."""
    if observable not in PULL_OBSERVABLE_KEYS:
        raise ValueError(f"Unsupported pull observable: {observable}")
    selected = [
        result
        for result in results
        if result.spec.channel == channel and result.strategy == strategy
    ]
    if not selected:
        raise ValueError(f"No {channel} {strategy} results are available")
    edges = selected[0].pull_observable_moments[observable].edges
    bin_cross_sections_pb = np.zeros(len(edges) - 1, dtype=np.float64)
    event_second_pb = np.zeros((len(edges) - 1, len(edges) - 1), dtype=np.float64)
    mc_second_pb2 = np.zeros_like(event_second_pb)
    selected_cross_section_pb = 0.0
    for result in selected:
        if result.pull_moment_model != PULL_MOMENT_MODEL:
            raise ValueError(
                f"Run lacks exact all-observable event moments for "
                f"{result.spec.name} {strategy}: {result.pull_moment_model}"
            )
        moments = result.pull_observable_moments[observable]
        if not np.array_equal(moments.edges, edges):
            raise ValueError(f"Inconsistent {observable} binning in {channel} {strategy}")
        if result.spec.name not in reference_cross_sections_pb:
            raise ValueError(f"Reference cross section is missing for {result.spec.name}")
        reference_cross_section = float(reference_cross_sections_pb[result.spec.name])
        scale_pb = reference_cross_section / result.generated_sumw
        bin_cross_sections_pb += scale_pb * moments.bin_sumw
        event_second_pb += scale_pb * moments.event_second_sumw
        mc_second_pb2 += scale_pb * scale_pb * moments.mc_second_sumw2
        final_step = cutflow_steps(channel, strategy)[-1]
        selected_cross_section_pb += scale_pb * result.cutflow[final_step].sumw
    if not np.all(np.isfinite(bin_cross_sections_pb)):
        raise ValueError(f"Non-finite total {observable} prediction")
    if not math.isclose(
        float(np.sum(bin_cross_sections_pb, dtype=np.float64)),
        selected_cross_section_pb,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            f"{channel} {strategy} {observable} integral does not match selected yield"
        )
    luminosity_scale = 1000.0 * float(luminosity_fb)
    return {
        "edges": edges.copy(),
        "bin_cross_sections_pb": bin_cross_sections_pb,
        "event_second_pb": event_second_pb,
        "mc_second_pb2": mc_second_pb2,
        "yield": luminosity_scale * bin_cross_sections_pb,
        "data_covariance": luminosity_scale * event_second_pb,
        "mc_covariance": luminosity_scale * luminosity_scale * mc_second_pb2,
        "selected_cross_section_pb": np.asarray(selected_cross_section_pb),
    }


def total_score_pull_statistics(
    channel: str,
    results: Sequence[SampleResult],
    luminosity_fb: float,
    reference_cross_sections_pb: Mapping[str, float],
) -> Dict[str, np.ndarray]:
    """Combine the common-selection score-pull moments over all processes."""
    selected = [
        result
        for result in results
        if result.spec.channel == channel and result.strategy == "xgboost"
    ]
    if not selected:
        raise ValueError(f"No {channel} XGBoost results are available")
    first = selected[0].score_pull_moments
    if first is None:
        raise ValueError(f"{channel} run lacks joint score-pull moments")
    score_edges = first.score_edges
    pull_edges = first.pull_edges
    shape = first.bin_sumw.shape
    flat_bins = shape[0] * shape[1]
    bin_cross_sections_pb = np.zeros(shape, dtype=np.float64)
    event_second_pb = np.zeros((flat_bins, flat_bins), dtype=np.float64)
    mc_second_pb2 = np.zeros_like(event_second_pb)
    common_cross_section_pb = 0.0
    for result in selected:
        moments = result.score_pull_moments
        if moments is None or result.score_pull_moment_model != SCORE_PULL_MOMENT_MODEL:
            raise ValueError(
                f"Run lacks exact joint score-pull moments for {result.spec.name}"
            )
        if not (
            np.array_equal(moments.score_edges, score_edges)
            and np.array_equal(moments.pull_edges, pull_edges)
        ):
            raise ValueError(f"Inconsistent score-pull binning in {channel}")
        if result.spec.name not in reference_cross_sections_pb:
            raise ValueError(f"Reference cross section is missing for {result.spec.name}")
        scale_pb = float(reference_cross_sections_pb[result.spec.name]) / result.generated_sumw
        bin_cross_sections_pb += scale_pb * moments.bin_sumw
        event_second_pb += scale_pb * moments.event_second_sumw
        mc_second_pb2 += scale_pb * scale_pb * moments.mc_second_sumw2
        common_cross_section_pb += scale_pb * result.cutflow[
            "xgboost_application_sample"
        ].sumw
    if not math.isclose(
        float(np.sum(bin_cross_sections_pb, dtype=np.float64)),
        common_cross_section_pb,
        rel_tol=1.0e-10,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(f"{channel} score-pull integral does not match common selection")
    luminosity_scale = 1000.0 * float(luminosity_fb)
    return {
        "score_edges": score_edges.copy(),
        "pull_edges": pull_edges.copy(),
        "bin_cross_sections_pb": bin_cross_sections_pb,
        "event_second_pb": event_second_pb,
        "mc_second_pb2": mc_second_pb2,
        "yield": luminosity_scale * bin_cross_sections_pb,
        "data_covariance": luminosity_scale * event_second_pb,
        "mc_covariance": luminosity_scale * luminosity_scale * mc_second_pb2,
        "common_cross_section_pb": np.asarray(common_cross_section_pb),
    }


def score_category_transformation(
    score_bins: int,
    pull_bins: int,
    score_ranges: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Map flattened score×pull bins into conditional pull categories."""
    transform = np.zeros(
        (len(score_ranges) * pull_bins, score_bins * pull_bins), dtype=np.float64
    )
    for category, (start, stop) in enumerate(score_ranges):
        if not (0 <= start < stop <= score_bins):
            raise ValueError(f"Invalid score-category range {(start, stop)}")
        for score_bin in range(start, stop):
            for pull_bin in range(pull_bins):
                transform[category * pull_bins + pull_bin, score_bin * pull_bins + pull_bin] = 1.0
    return transform


def conditional_score_pull_statistics(
    statistics: Mapping[str, np.ndarray],
    score_ranges: Sequence[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Normalize the six-bin pull distribution independently in each category."""
    score_edges = np.asarray(statistics["score_edges"], dtype=np.float64)
    pull_edges = np.asarray(statistics["pull_edges"], dtype=np.float64)
    bin_yields = np.asarray(statistics["yield"], dtype=np.float64)
    score_bins, pull_bins = bin_yields.shape
    if pull_bins != PULL_BIN_COUNT or not np.array_equal(pull_edges, PULL_BIN_EDGES):
        raise ValueError("Conditional score-pull statistics require the six folded-angle bins")
    transform = score_category_transformation(score_bins, pull_bins, score_ranges)
    flat_yields = bin_yields.reshape(-1)
    category_yields = transform @ flat_yields
    data_unnormalized = (
        transform @ np.asarray(statistics["data_covariance"], dtype=np.float64) @ transform.T
    )
    mc_unnormalized = (
        transform @ np.asarray(statistics["mc_covariance"], dtype=np.float64) @ transform.T
    )
    fractions = np.zeros_like(category_yields)
    jacobian = np.zeros((len(category_yields), len(category_yields)), dtype=np.float64)
    totals = np.zeros(len(score_ranges), dtype=np.float64)
    for category in range(len(score_ranges)):
        block = slice(category * pull_bins, (category + 1) * pull_bins)
        values = category_yields[block]
        total = float(np.sum(values, dtype=np.float64))
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Every score category must have a positive finite prediction")
        totals[category] = total
        fractions[block] = values / total
        jacobian[block, block] = (
            np.eye(pull_bins, dtype=np.float64) - fractions[block, None]
        ) / total
    data_covariance = jacobian @ data_unnormalized @ jacobian.T
    mc_covariance = jacobian @ mc_unnormalized @ jacobian.T
    return {
        "score_edges": score_edges,
        "pull_edges": pull_edges,
        "score_ranges": np.asarray(score_ranges, dtype=np.int64),
        "category_yields": totals,
        "R": fractions,
        "data_covariance": 0.5 * (data_covariance + data_covariance.T),
        "mc_covariance": 0.5 * (mc_covariance + mc_covariance.T),
    }


def contiguous_score_partitions(
    score_bins: int,
    categories: int,
) -> Tuple[Tuple[Tuple[int, int], ...], ...]:
    if categories < 1 or categories > score_bins:
        raise ValueError("Score-category count is outside the available score bins")
    partitions: List[Tuple[Tuple[int, int], ...]] = []
    for cuts in itertools.combinations(range(1, score_bins), categories - 1):
        boundaries = (0,) + tuple(cuts) + (score_bins,)
        partitions.append(
            tuple((boundaries[index], boundaries[index + 1]) for index in range(categories))
        )
    return tuple(partitions)


def score_partition_comparison(
    reference: Mapping[str, np.ndarray],
    variation: Mapping[str, np.ndarray],
    score_ranges: Sequence[Tuple[int, int]],
) -> Dict[str, Any]:
    """Evaluate conditional pull-shape separation for one fixed partition."""
    reference_conditional = conditional_score_pull_statistics(reference, score_ranges)
    variation_conditional = conditional_score_pull_statistics(variation, score_ranges)
    delta = variation_conditional["R"] - reference_conditional["R"]
    independent_mc = (
        reference_conditional["mc_covariance"]
        + variation_conditional["mc_covariance"]
    )
    nominal_data = reference_conditional["data_covariance"]
    variation_data = variation_conditional["data_covariance"]
    delta_fbeam = []
    for category in range(len(score_ranges)):
        block = slice(category * PULL_BIN_COUNT, (category + 1) * PULL_BIN_COUNT)
        delta_fbeam.append(float(np.sum(delta[block][: PULL_BIN_COUNT // 2])))
    return {
        "score_ranges": [list(values) for values in score_ranges],
        "reference_category_yields": reference_conditional["category_yields"].tolist(),
        "variation_category_yields": variation_conditional["category_yields"].tolist(),
        "delta_f_beam_by_category": delta_fbeam,
        "nominal_truth": {
            "data_stat_only": mahalanobis_distance(delta, nominal_data),
            "data_plus_mc_stat": mahalanobis_distance(delta, nominal_data + independent_mc),
        },
        "variation_truth": {
            "data_stat_only": mahalanobis_distance(delta, variation_data),
            "data_plus_mc_stat": mahalanobis_distance(
                delta, variation_data + independent_mc
            ),
        },
    }


def propagated_independent_ratio_errors(
    numerator: np.ndarray,
    numerator_covariance: np.ndarray,
    reference: np.ndarray,
    reference_covariance: np.ndarray,
    *,
    include_reference: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return central ratios and delta-method errors, masking zero reference bins."""
    numerator = np.asarray(numerator, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    numerator_covariance = np.asarray(numerator_covariance, dtype=np.float64)
    reference_covariance = np.asarray(reference_covariance, dtype=np.float64)
    if numerator.shape != reference.shape:
        raise ValueError("Ratio numerator and reference shapes differ")
    ratios = np.full(numerator.shape, np.nan, dtype=np.float64)
    errors = np.full(numerator.shape, np.nan, dtype=np.float64)
    mask = reference != 0.0
    ratios[mask] = numerator[mask] / reference[mask]
    variance = np.zeros_like(numerator)
    variance[mask] = np.diag(numerator_covariance)[mask] / np.square(reference[mask])
    if include_reference:
        variance[mask] += (
            np.square(numerator[mask])
            * np.diag(reference_covariance)[mask]
            / np.power(reference[mask], 4)
        )
    errors[mask] = np.sqrt(np.maximum(variance[mask], 0.0))
    return ratios, errors


def chi_square_log_survival(d_squared: float, degrees_of_freedom: int) -> float:
    """Return log P(chi2_k >= D2) without requiring SciPy.

    The regularized upper incomplete gamma function is evaluated with a series
    for its complement at small arguments and a continued fraction otherwise.
    Returning the logarithm keeps extremely small expected p-values usable.
    """
    value = float(d_squared)
    dof = int(degrees_of_freedom)
    if dof < 1:
        raise ValueError("Chi-square degrees of freedom must be positive")
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("Chi-square statistic must be finite and non-negative")
    if value == 0.0:
        return 0.0
    shape = 0.5 * dof
    argument = 0.5 * value
    epsilon = 3.0e-14
    max_iterations = 10000

    if argument < shape + 1.0:
        current_shape = shape
        term = 1.0 / shape
        total = term
        for _ in range(max_iterations):
            current_shape += 1.0
            term *= argument / current_shape
            total += term
            if abs(term) <= abs(total) * epsilon:
                break
        else:
            raise RuntimeError("Chi-square lower-gamma series did not converge")
        log_lower = (
            -argument
            + shape * math.log(argument)
            - math.lgamma(shape)
            + math.log(total)
        )
        log_lower = min(log_lower, 0.0)
        if log_lower < math.log(0.5):
            log_upper = math.log1p(-math.exp(log_lower))
        else:
            upper = -math.expm1(log_lower)
            if upper <= 0.0:
                return -math.inf
            log_upper = math.log(upper)
        return min(log_upper, 0.0)

    tiny = np.finfo(np.float64).tiny / epsilon
    denominator = argument + 1.0 - shape
    if abs(denominator) < tiny:
        denominator = tiny
    reciprocal = 1.0 / denominator
    continued = 1.0 / tiny
    fraction = reciprocal
    for iteration in range(1, max_iterations + 1):
        coefficient = -float(iteration) * (float(iteration) - shape)
        denominator += 2.0
        reciprocal = coefficient * reciprocal + denominator
        if abs(reciprocal) < tiny:
            reciprocal = tiny
        continued = denominator + coefficient / continued
        if abs(continued) < tiny:
            continued = tiny
        reciprocal = 1.0 / reciprocal
        update = reciprocal * continued
        fraction *= update
        if abs(update - 1.0) <= epsilon:
            break
    else:
        raise RuntimeError("Chi-square upper-gamma continued fraction did not converge")
    if fraction <= 0.0 or not math.isfinite(fraction):
        raise RuntimeError("Chi-square upper-gamma continued fraction is invalid")
    return min(
        -argument
        + shape * math.log(argument)
        - math.lgamma(shape)
        + math.log(fraction),
        0.0,
    )


def chi_square_survival(d_squared: float, degrees_of_freedom: int) -> float:
    """Return the upper-tail chi-square probability, allowing underflow to zero."""
    log_probability = chi_square_log_survival(d_squared, degrees_of_freedom)
    if log_probability < math.log(np.finfo(np.float64).tiny):
        return 0.0
    return math.exp(log_probability)


def mahalanobis_distance(
    difference: np.ndarray,
    covariance: np.ndarray,
    rcond: float = COMPARISON_PINV_RCOND,
) -> Dict[str, Any]:
    """Evaluate D^2 in the supported covariance subspace."""
    delta = np.asarray(difference, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (len(delta), len(delta)):
        raise ValueError("Mahalanobis covariance shape differs from the vector")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    maximum = max(float(np.max(np.abs(eigenvalues))), 0.0)
    tolerance = max(float(rcond) * maximum, np.finfo(np.float64).eps)
    supported = eigenvalues > tolerance
    rank = int(np.count_nonzero(supported))
    if rank:
        projection = eigenvectors[:, supported].T @ delta
        d_squared = float(np.sum(np.square(projection) / eigenvalues[supported]))
    else:
        d_squared = 0.0
    d_squared = max(d_squared, 0.0)
    log_probability = (
        chi_square_log_survival(d_squared, rank) if rank else 0.0
    )
    return {
        "D2": d_squared,
        "mahalanobis_separation": math.sqrt(d_squared),
        "covariance_rank": rank,
        "pseudoinverse_tolerance": tolerance,
        "pseudoinverse_rcond": float(rcond),
        "p_value": (
            math.exp(log_probability)
            if log_probability >= math.log(np.finfo(np.float64).tiny)
            else 0.0
        ),
        "log10_p_value": log_probability / math.log(10.0),
    }


def observable_hypothesis_tests(
    reference: Mapping[str, np.ndarray],
    variation: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """Compare two binned predictions with rate+shape and shape-only tests."""
    reference_yield = np.asarray(reference["yield"], dtype=np.float64)
    variation_yield = np.asarray(variation["yield"], dtype=np.float64)
    if reference_yield.shape != variation_yield.shape:
        raise ValueError("Observable hypothesis-test yields have different shapes")
    reference_data = np.asarray(reference["data_covariance"], dtype=np.float64)
    variation_data = np.asarray(variation["data_covariance"], dtype=np.float64)
    reference_mc = np.asarray(reference["mc_covariance"], dtype=np.float64)
    variation_mc = np.asarray(variation["mc_covariance"], dtype=np.float64)

    def covariance_tests(
        difference: np.ndarray,
        nominal_data: np.ndarray,
        varied_data: np.ndarray,
        independent_mc: np.ndarray,
    ) -> Dict[str, Any]:
        return {
            "mc_stat_only": mahalanobis_distance(difference, independent_mc),
            "nominal_truth": {
                "data_stat_only": mahalanobis_distance(difference, nominal_data),
                "data_plus_mc_stat": mahalanobis_distance(
                    difference, nominal_data + independent_mc
                ),
            },
            "variation_truth": {
                "data_stat_only": mahalanobis_distance(difference, varied_data),
                "data_plus_mc_stat": mahalanobis_distance(
                    difference, varied_data + independent_mc
                ),
            },
        }

    yield_difference = variation_yield - reference_yield
    rate_and_shape = covariance_tests(
        yield_difference,
        reference_data,
        variation_data,
        reference_mc + variation_mc,
    )
    reference_fraction, reference_shape_data = normalized_binned_prediction(
        reference_yield, reference_data
    )
    variation_fraction, variation_shape_data = normalized_binned_prediction(
        variation_yield, variation_data
    )
    _, reference_shape_mc = normalized_binned_prediction(
        reference_yield, reference_mc
    )
    _, variation_shape_mc = normalized_binned_prediction(
        variation_yield, variation_mc
    )
    shape_only = covariance_tests(
        variation_fraction - reference_fraction,
        reference_shape_data,
        variation_shape_data,
        reference_shape_mc + variation_shape_mc,
    )
    return {
        "rate_and_shape": rate_and_shape,
        "shape_only": shape_only,
    }


def _xgboost_model_signature(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    signature: Dict[str, Any] = {"feature_names": list(metadata.get("feature_names", ())), "channels": {}}
    for channel, values in metadata.get("channels", {}).items():
        if values.get("models"):
            models = sorted(values["models"], key=lambda item: int(item["fold"]))
            channel_signature = [
                {
                    "fold": int(item["fold"]),
                    "model_sha256": str(item["model_sha256"]),
                    "score_threshold": float(item["score_threshold"]),
                }
                for item in models
            ]
        else:
            channel_signature = [
                {
                    "fold": 0,
                    "model_sha256": str(values["model_sha256"]),
                    "score_threshold": float(values["score_threshold"]),
                }
            ]
        signature["channels"][str(channel)] = channel_signature
    return signature


def _scenario_from_run(
    run_dir: Path,
    config: AnalysisConfig,
    metadata: Mapping[str, Any],
    index: int,
) -> ScenarioSpec:
    if config.scenario is not None:
        scenario = config.scenario
    else:
        fallback = sanitize_run_name(str(metadata.get("run_name") or metadata["run_id"]))
        scenario = ScenarioSpec(
            identifier=fallback or f"scenario-{index + 1}",
            label=str(metadata.get("run_name") or metadata["run_id"]),
        )
    color = scenario.color or COMPARISON_COLORS[index % len(COMPARISON_COLORS)]
    return ScenarioSpec(scenario.identifier, scenario.label, color, dict(scenario.parameters))


def resolve_comparison_source(
    value: Path,
    output_root: Path,
    luminosities: Optional[Sequence[float]],
    index: int,
) -> ComparisonSource:
    candidate = value.expanduser()
    if not candidate.exists():
        candidate = output_root.expanduser().resolve() / "runs" / str(value)
    run_dir = candidate.resolve()
    expected_parent = (output_root.expanduser().resolve() / "runs").resolve()
    if run_dir.parent != expected_parent:
        raise ValueError(
            f"Comparison source {run_dir} is not inside {expected_parent}; "
            "portable source links require one output root"
        )
    config, results, metadata = load_completed_run(run_dir, luminosities)
    xgb_path = run_dir / "summaries" / "xgboost.json"
    xgb_metadata = (
        json.loads(xgb_path.read_text(encoding="utf-8")) if xgb_path.is_file() else None
    )
    return ComparisonSource(
        run_dir=run_dir,
        config=config,
        results=tuple(results),
        metadata=metadata,
        scenario=_scenario_from_run(run_dir, config, metadata, index),
        xgboost_metadata=xgb_metadata,
    )


def validate_comparison_sources(
    sources: Sequence[ComparisonSource],
    analyses: Sequence[str],
) -> Dict[str, float]:
    if len(sources) < 2:
        raise ValueError("At least two completed runs are required for comparison")
    if len({source.run_dir for source in sources}) != len(sources):
        raise ValueError("Comparison source runs must be distinct")
    if len({source.scenario.identifier for source in sources}) != len(sources):
        raise ValueError("Comparison scenario ids must be unique")
    reference = sources[0]
    reference_samples = {
        sample.name: (sample.channel, sample.role) for sample in reference.config.samples
    }
    reference_cross_sections = {
        sample.name: float(sample.cross_section_pb) for sample in reference.config.samples
    }
    reference_cuts = reference.metadata.get("configuration", {}).get("cuts", {})
    reference_tree = reference.config.tree_name
    reference_partial = bool(reference.metadata.get("partial"))
    reference_max_events = reference.metadata.get("max_events_per_sample")
    reference_version = reference.metadata.get("analysis_version")
    reference_strategies = {result.strategy for result in reference.results}
    reference_xgb_signature = (
        _xgboost_model_signature(reference.xgboost_metadata)
        if "xgboost" in analyses and reference.xgboost_metadata is not None
        else None
    )
    for source in sources:
        available = {result.strategy for result in source.results}
        if available != reference_strategies:
            raise ValueError(f"Analysis strategies differ in source run {source.run_dir}")
        missing = set(analyses) - available
        if missing:
            raise ValueError(f"Source run {source.run_dir} is missing analyses {sorted(missing)}")
        identities = {
            sample.name: (sample.channel, sample.role) for sample in source.config.samples
        }
        if identities != reference_samples:
            raise ValueError(f"Process identities differ in source run {source.run_dir}")
        if source.config.tree_name != reference_tree:
            raise ValueError(f"ROOT tree setting differs in source run {source.run_dir}")
        if source.metadata.get("analysis_version") != reference_version:
            raise ValueError(f"Analysis version differs in source run {source.run_dir}")
        if source.metadata.get("configuration", {}).get("cuts", {}) != reference_cuts:
            raise ValueError(f"Analysis cuts differ in source run {source.run_dir}")
        if bool(source.metadata.get("partial")) != reference_partial or source.metadata.get(
            "max_events_per_sample"
        ) != reference_max_events:
            raise ValueError("Full and partial runs, or unequal event limits, cannot be compared")
        for result in source.results:
            if result.strategy not in analyses:
                continue
            if result.pull_moment_model != PULL_MOMENT_MODEL:
                raise ValueError(
                    f"Comparison source {source.run_dir.name} lacks required event-level "
                    f"moments for {result.spec.name} {result.strategy}"
                )
            for observable in PULL_OBSERVABLE_KEYS:
                histogram = result.histograms[observable]
                moments = result.pull_observable_moments[observable]
                if not np.array_equal(histogram.edges, moments.edges):
                    raise ValueError(
                        f"Histogram/moment binning mismatch for {observable} in {source.run_dir}"
                    )
                reference_result = next(
                    item
                    for item in reference.results
                    if item.spec.name == result.spec.name and item.strategy == result.strategy
                )
                if not np.array_equal(
                    moments.edges,
                    reference_result.pull_observable_moments[observable].edges,
                ):
                    raise ValueError(
                        f"Pull-observable binning differs for {observable} in {source.run_dir}"
                    )
            reference_result = next(
                item
                for item in reference.results
                if item.spec.name == result.spec.name and item.strategy == result.strategy
            )
            if set(result.histograms) != set(reference_result.histograms):
                raise ValueError(f"Histogram identities differ in source run {source.run_dir}")
            for key, histogram in result.histograms.items():
                if not np.array_equal(histogram.edges, reference_result.histograms[key].edges):
                    raise ValueError(
                        f"Histogram binning differs for {key} in source run {source.run_dir}"
                    )
        if "xgboost" in analyses:
            if source.xgboost_metadata is None or reference_xgb_signature is None:
                raise ValueError("Every XGBoost comparison source must contain model metadata")
            if _xgboost_model_signature(source.xgboost_metadata) != reference_xgb_signature:
                raise ValueError(
                    f"Frozen XGBoost model hashes or thresholds differ in {source.run_dir}"
                )
            for channel in ("higgs", "z"):
                reference_joint_edges = None
                for result in source.results:
                    if result.strategy != "xgboost" or result.spec.channel != channel:
                        continue
                    if (
                        result.score_pull_moments is None
                        or result.score_pull_moment_model != SCORE_PULL_MOMENT_MODEL
                    ):
                        raise ValueError(
                            f"Comparison source {source.run_dir.name} lacks joint score-pull "
                            f"moments for {result.spec.name}; rerun it with analysis version "
                            "2.3 or later"
                        )
                    edges = result.score_pull_moments.score_edges
                    if reference_joint_edges is None:
                        reference_joint_edges = edges
                    elif not np.array_equal(edges, reference_joint_edges):
                        raise ValueError(
                            f"Score-pull edges differ between {channel} processes in "
                            f"{source.run_dir}"
                        )
                    nominal_result = next(
                        item
                        for item in reference.results
                        if item.strategy == "xgboost"
                        and item.spec.channel == channel
                        and item.spec.name == result.spec.name
                    )
                    if (
                        nominal_result.score_pull_moments is None
                        or not np.array_equal(
                            edges, nominal_result.score_pull_moments.score_edges
                        )
                    ):
                        raise ValueError(
                            f"Frozen nominal score-quantile edges differ in {source.run_dir}"
                        )
    return reference_cross_sections


def build_comparison_statistics(
    sources: Sequence[ComparisonSource],
    analyses: Sequence[str],
    luminosities: Sequence[float],
    reference_cross_sections_pb: Mapping[str, float],
) -> Tuple[Dict[str, Any], Dict[Tuple[str, str, float, str, str], Dict[str, np.ndarray]]]:
    """Build serializable comparison summaries plus arrays used by plots/artifacts."""
    source_rows = [
        {
            "run_id": source.metadata["run_id"],
            "run_name": source.metadata.get("run_name"),
            "scenario": asdict(source.scenario),
            "partial": bool(source.metadata.get("partial")),
            "analysis_version": source.metadata.get("analysis_version"),
            "configuration_hash": source.metadata.get("configuration_hash"),
        }
        for source in sources
    ]
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "reference_run_id": sources[0].metadata["run_id"],
        "reference_scenario_id": sources[0].scenario.identifier,
        "source_runs": source_rows,
        "normalization": {
            "cross_section_source": "first comparison run",
            "reference_cross_sections_pb": dict(reference_cross_sections_pb),
            "formula": "1000 * luminosity_fb * sigma_reference_pb / generated_sumw_scenario",
        },
        "independence_assumption": "process samples and scenario samples are statistically independent",
        "observable_hypothesis_tests": {
            "statistic": "Mahalanobis D2 in the covariance-supported bin subspace",
            "p_value": "upper-tail chi-square probability with covariance rank degrees of freedom",
            "primary_ranking": "shape_only / nominal_truth / data_plus_mc_stat",
            "rate_and_shape": "absolute expected-yield histogram, including normalization differences",
            "shape_only": "each scenario histogram normalized independently to unit area",
            "scope": "local expected (Asimov) p-values; no cross-observable look-elsewhere correction",
        },
        "xgboost_model_signature": (
            _xgboost_model_signature(sources[0].xgboost_metadata)
            if "xgboost" in analyses and sources[0].xgboost_metadata is not None
            else None
        ),
        "analyses": {},
    }
    numerical: Dict[
        Tuple[str, str, float, str, str], Dict[str, np.ndarray]
    ] = {}
    ranking_candidates: Dict[str, List[Dict[str, Any]]] = {
        "shape_only": [],
        "rate_and_shape": [],
    }
    selector = np.zeros(PULL_BIN_COUNT, dtype=np.float64)
    selector[: PULL_BIN_COUNT // 2] = 1.0
    for strategy in analyses:
        strategy_payload: Dict[str, Any] = {}
        for channel in ("higgs", "z"):
            reference_id = sources[0].scenario.identifier
            channel_payload: Dict[str, Any] = {
                "scenarios": {},
                "differences_from_reference": {},
                "observables": {},
            }
            differentials: Dict[str, Dict[str, Any]] = {}
            for source in sources:
                folded = total_pull_observable_statistics(
                    channel,
                    strategy,
                    "folded_pull_angle",
                    source.results,
                    luminosities[0],
                    reference_cross_sections_pb,
                )
                differential = differential_pull_statistics(
                    folded["bin_cross_sections_pb"],
                    folded["event_second_pb"],
                    folded["mc_second_pb2"],
                    luminosities,
                )
                if differential["R"] is None:
                    raise RuntimeError(
                        f"Empty folded pull prediction for {source.scenario.identifier} "
                        f"{channel} {strategy}"
                    )
                differentials[source.scenario.identifier] = differential
                channel_payload["scenarios"][source.scenario.identifier] = {
                    "label": source.scenario.label,
                    "color": source.scenario.color,
                    "selected_cross_section_pb": float(folded["selected_cross_section_pb"]),
                    "R": differential["R"],
                    "f_beam": differential["f_beam"],
                    "mc_statistical_covariance": differential["mc_statistical_covariance"],
                    "f_beam_mc_statistical_error": differential[
                        "f_beam_mc_statistical_error"
                    ],
                    "expected_statistical_covariance": differential[
                        "expected_statistical_covariance"
                    ],
                    "f_beam_statistical_error": differential[
                        "f_beam_statistical_error"
                    ],
                }

            for observable in PULL_OBSERVABLE_KEYS:
                observable_payload: Dict[str, Any] = {
                    "bin_edges": None,
                    "luminosities": {},
                    "comparisons_to_reference": {},
                }
                for luminosity in luminosities:
                    luminosity_payload: Dict[str, Any] = {}
                    luminosity_statistics: Dict[str, Dict[str, np.ndarray]] = {}
                    for source in sources:
                        statistics = total_pull_observable_statistics(
                            channel,
                            strategy,
                            observable,
                            source.results,
                            luminosity,
                            reference_cross_sections_pb,
                        )
                        key = (
                            strategy,
                            channel,
                            float(luminosity),
                            observable,
                            source.scenario.identifier,
                        )
                        numerical[key] = statistics
                        luminosity_statistics[source.scenario.identifier] = statistics
                        if observable_payload["bin_edges"] is None:
                            observable_payload["bin_edges"] = statistics["edges"].tolist()
                        luminosity_payload[source.scenario.identifier] = {
                            "total_yield": statistics["yield"].tolist(),
                            "data_statistical_covariance": statistics[
                                "data_covariance"
                            ].tolist(),
                            "mc_statistical_covariance": statistics["mc_covariance"].tolist(),
                        }
                    observable_payload["luminosities"][str(float(luminosity))] = (
                        luminosity_payload
                    )
                    reference_statistics = luminosity_statistics[reference_id]
                    for source in sources[1:]:
                        scenario_id = source.scenario.identifier
                        tests = observable_hypothesis_tests(
                            reference_statistics,
                            luminosity_statistics[scenario_id],
                        )
                        comparison_values = observable_payload[
                            "comparisons_to_reference"
                        ].setdefault(
                            scenario_id,
                            {"label": source.scenario.label, "luminosities": {}},
                        )
                        comparison_values["luminosities"][str(float(luminosity))] = tests
                        for test_scope in ("shape_only", "rate_and_shape"):
                            distance = tests[test_scope]["nominal_truth"][
                                "data_plus_mc_stat"
                            ]
                            ranking_candidates[test_scope].append(
                                {
                                    "strategy": strategy,
                                    "channel": channel,
                                    "observable": observable,
                                    "scenario": scenario_id,
                                    "scenario_label": source.scenario.label,
                                    "luminosity_fb": float(luminosity),
                                    "D2": distance["D2"],
                                    "covariance_rank": distance["covariance_rank"],
                                    "p_value": distance["p_value"],
                                    "log10_p_value": distance["log10_p_value"],
                                }
                            )
                channel_payload["observables"][observable] = observable_payload

            reference_differential = differentials[reference_id]
            reference_r = np.asarray(reference_differential["R"], dtype=np.float64)
            reference_mc_covariance = np.asarray(
                reference_differential["mc_statistical_covariance"], dtype=np.float64
            )
            for source in sources[1:]:
                scenario_id = source.scenario.identifier
                differential = differentials[scenario_id]
                scenario_r = np.asarray(differential["R"], dtype=np.float64)
                delta_r = scenario_r - reference_r
                delta_fbeam = float(selector @ delta_r)
                scenario_mc_covariance = np.asarray(
                    differential["mc_statistical_covariance"], dtype=np.float64
                )
                independent_mc_covariance = (
                    scenario_mc_covariance + reference_mc_covariance
                )
                luminosity_summaries: Dict[str, Any] = {}
                for luminosity in luminosities:
                    lumi_key = str(float(luminosity))
                    reference_data_covariance = np.asarray(
                        reference_differential["expected_statistical_covariance"][lumi_key],
                        dtype=np.float64,
                    )
                    scenario_data_covariance = np.asarray(
                        differential["expected_statistical_covariance"][lumi_key],
                        dtype=np.float64,
                    )
                    truth_summaries = {}
                    for truth_name, data_covariance in (
                        ("nominal_truth", reference_data_covariance),
                        ("variation_truth", scenario_data_covariance),
                    ):
                        data_fbeam_variance = max(
                            float(selector @ data_covariance @ selector), 0.0
                        )
                        combined_fbeam_variance = max(
                            float(
                                selector
                                @ (data_covariance + independent_mc_covariance)
                                @ selector
                            ),
                            0.0,
                        )
                        truth_summaries[truth_name] = {
                            "directional_f_beam_significance_data_stat_only": (
                                delta_fbeam / math.sqrt(data_fbeam_variance)
                                if data_fbeam_variance > 0.0
                                else None
                            ),
                            "directional_f_beam_significance_data_plus_mc_stat": (
                                delta_fbeam / math.sqrt(combined_fbeam_variance)
                                if combined_fbeam_variance > 0.0
                                else None
                            ),
                            "six_bin_data_stat_only": mahalanobis_distance(
                                delta_r, data_covariance
                            ),
                            "six_bin_data_plus_mc_stat": mahalanobis_distance(
                                delta_r, data_covariance + independent_mc_covariance
                            ),
                        }
                    luminosity_summaries[lumi_key] = truth_summaries
                channel_payload["differences_from_reference"][scenario_id] = {
                    "label": source.scenario.label,
                    "delta_R": delta_r.tolist(),
                    "delta_f_beam": delta_fbeam,
                    "independent_mc_difference_covariance": independent_mc_covariance.tolist(),
                    "luminosities": luminosity_summaries,
                }
            strategy_payload[channel] = channel_payload
        payload["analyses"][strategy] = strategy_payload
    rankings: Dict[str, Any] = {}
    for test_scope, candidates in ranking_candidates.items():
        if not candidates:
            continue
        best = min(candidates, key=lambda item: float(item["log10_p_value"]))
        trial_count = len(candidates)
        bonferroni_log10 = min(
            0.0,
            float(best["log10_p_value"]) + math.log10(float(trial_count)),
        )
        bonferroni_probability = (
            10.0 ** bonferroni_log10
            if bonferroni_log10 >= math.log10(np.finfo(np.float64).tiny)
            else 0.0
        )
        rankings[test_scope] = {
            "test_count": trial_count,
            "most_significant_local_test": dict(best),
            "bonferroni_upper_bound_for_selected_minimum_p": bonferroni_probability,
            "log10_bonferroni_upper_bound": bonferroni_log10,
            "correlation_note": (
                "The tests share events and are correlated; the Bonferroni value is a "
                "conservative bound, not an exact global p-value."
            ),
        }
    payload["observable_hypothesis_test_rankings"] = rankings
    return payload, numerical


def build_score_pull_diagnostic(
    sources: Sequence[ComparisonSource],
    luminosities: Sequence[float],
    reference_cross_sections_pb: Mapping[str, float],
) -> Tuple[
    Dict[str, Any],
    Dict[Tuple[str, float, str], Dict[str, np.ndarray]],
]:
    """Build the exploratory score-quantile × folded-pull comparison."""
    reference_id = sources[0].scenario.identifier
    variation_ids = [source.scenario.identifier for source in sources[1:]]
    numerical: Dict[Tuple[str, float, str], Dict[str, np.ndarray]] = {}
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "moment_model": SCORE_PULL_MOMENT_MODEL,
        "selection": "common selection through opposite hemispheres; no VBF or score cut",
        "score_binning": "ten total-prediction weighted quantiles frozen by the nominal run",
        "pull_binning": "six equal absolute signed-pull-angle bins on [0, pi]",
        "category_objective": (
            "conditional pull-shape Mahalanobis D2; each score category is normalized "
            "independently, with nominal-truth projected-data covariance plus independent "
            "nominal/variation MC covariance"
        ),
        "multi_variation_objective": "maximize the minimum D2 across all supplied variations",
        "interpretation": (
            "exploratory design diagnostic; boundaries are fixed across scenarios but must be "
            "confirmed with statistically independent simulations before a significance claim"
        ),
        "reference_scenario_id": reference_id,
        "variation_scenario_ids": variation_ids,
        "channels": {},
    }
    singleton_ranges = tuple((index, index + 1) for index in range(SCORE_PULL_BIN_COUNT))
    for channel in ("higgs", "z"):
        channel_payload: Dict[str, Any] = {"luminosities": {}}
        for luminosity in luminosities:
            for source in sources:
                numerical[(channel, float(luminosity), source.scenario.identifier)] = (
                    total_score_pull_statistics(
                        channel,
                        source.results,
                        luminosity,
                        reference_cross_sections_pb,
                    )
                )
            reference = numerical[(channel, float(luminosity), reference_id)]
            score_edges = np.asarray(reference["score_edges"], dtype=np.float64)
            for scenario_id in variation_ids:
                if not np.array_equal(
                    numerical[(channel, float(luminosity), scenario_id)]["score_edges"],
                    score_edges,
                ):
                    raise ValueError(
                        f"Frozen score edges differ for {channel} scenario {scenario_id}"
                    )
            reference_singletons = conditional_score_pull_statistics(
                reference, singleton_ranges
            )
            variations_payload: Dict[str, Any] = {}
            for source in sources[1:]:
                scenario_id = source.scenario.identifier
                variation = numerical[(channel, float(luminosity), scenario_id)]
                variation_singletons = conditional_score_pull_statistics(
                    variation, singleton_ranges
                )
                delta = variation_singletons["R"] - reference_singletons["R"]
                independent_mc = (
                    reference_singletons["mc_covariance"]
                    + variation_singletons["mc_covariance"]
                )
                quantiles = []
                selector = np.concatenate(
                    (np.ones(PULL_BIN_COUNT // 2), np.zeros(PULL_BIN_COUNT // 2))
                )
                for score_bin in range(SCORE_PULL_BIN_COUNT):
                    block = slice(
                        score_bin * PULL_BIN_COUNT,
                        (score_bin + 1) * PULL_BIN_COUNT,
                    )
                    reference_r = reference_singletons["R"][block]
                    variation_r = variation_singletons["R"][block]
                    delta_r = delta[block]
                    data_covariance = reference_singletons["data_covariance"][block, block]
                    mc_covariance = independent_mc[block, block]
                    delta_fbeam = float(selector @ delta_r)
                    quantiles.append(
                        {
                            "score_bin": score_bin + 1,
                            "score_low": float(score_edges[score_bin]),
                            "score_high": float(score_edges[score_bin + 1]),
                            "reference_yield": float(
                                reference_singletons["category_yields"][score_bin]
                            ),
                            "variation_yield": float(
                                variation_singletons["category_yields"][score_bin]
                            ),
                            "reference_R": reference_r.tolist(),
                            "variation_R": variation_r.tolist(),
                            "delta_R": delta_r.tolist(),
                            "reference_f_beam": float(selector @ reference_r),
                            "variation_f_beam": float(selector @ variation_r),
                            "delta_f_beam": delta_fbeam,
                            "delta_f_beam_data_plus_mc_error": math.sqrt(
                                max(
                                    float(
                                        selector
                                        @ (data_covariance + mc_covariance)
                                        @ selector
                                    ),
                                    0.0,
                                )
                            ),
                            "six_bin_data_stat_only": mahalanobis_distance(
                                delta_r, data_covariance
                            ),
                            "six_bin_data_plus_mc_stat": mahalanobis_distance(
                                delta_r, data_covariance + mc_covariance
                            ),
                        }
                    )
                variations_payload[scenario_id] = {
                    "label": source.scenario.label,
                    "quantiles": quantiles,
                }

            category_scans: Dict[str, Any] = {}
            for category_count in (2, 3):
                candidates = []
                for score_ranges in contiguous_score_partitions(
                    SCORE_PULL_BIN_COUNT, category_count
                ):
                    per_variation = {}
                    data_plus_mc_values = []
                    data_only_values = []
                    for source in sources[1:]:
                        scenario_id = source.scenario.identifier
                        comparison = score_partition_comparison(
                            reference,
                            numerical[(channel, float(luminosity), scenario_id)],
                            score_ranges,
                        )
                        data_plus_mc = comparison["nominal_truth"][
                            "data_plus_mc_stat"
                        ]["D2"]
                        data_only = comparison["nominal_truth"]["data_stat_only"]["D2"]
                        data_plus_mc_values.append(float(data_plus_mc))
                        data_only_values.append(float(data_only))
                        per_variation[scenario_id] = {
                            "D2_data_plus_mc_stat": float(data_plus_mc),
                            "D2_data_stat_only": float(data_only),
                            "delta_f_beam_by_category": comparison[
                                "delta_f_beam_by_category"
                            ],
                        }
                    candidates.append(
                        {
                            "score_ranges": [list(values) for values in score_ranges],
                            "boundary_scores": [
                                float(score_edges[stop]) for _, stop in score_ranges[:-1]
                            ],
                            "minimum_D2_data_plus_mc_stat": min(data_plus_mc_values),
                            "mean_D2_data_plus_mc_stat": float(
                                np.mean(data_plus_mc_values, dtype=np.float64)
                            ),
                            "minimum_D2_data_stat_only": min(data_only_values),
                            "per_variation": per_variation,
                        }
                    )
                candidates.sort(
                    key=lambda values: (
                        values["minimum_D2_data_plus_mc_stat"],
                        values["mean_D2_data_plus_mc_stat"],
                    ),
                    reverse=True,
                )
                category_scans[str(category_count)] = {
                    "recommended": candidates[0],
                    "candidates": candidates,
                }
            channel_payload["luminosities"][str(float(luminosity))] = {
                "score_edges": score_edges.tolist(),
                "variations": variations_payload,
                "category_scans": category_scans,
            }
        payload["channels"][channel] = channel_payload
    return payload, numerical


def write_comparison_artifacts(
    run_dir: Path,
    payload: Mapping[str, Any],
    numerical: Mapping[Tuple[str, str, float, str, str], Mapping[str, np.ndarray]],
) -> Tuple[Path, Path, Path]:
    json_path = run_dir / "summaries" / "comparison.json"
    csv_path = run_dir / "summaries" / "comparison.csv"
    npz_path = run_dir / "summaries" / "comparison.npz"
    write_json_exclusive(json_path, payload)
    fields = [
        "record_type",
        "strategy",
        "channel",
        "luminosity_fb",
        "observable",
        "scenario",
        "reference_scenario",
        "bin",
        "low_edge",
        "high_edge",
        "expected_yield",
        "data_statistical_error",
        "mc_statistical_error",
        "R_i",
        "f_beam",
        "delta_R_i",
        "delta_f_beam",
        "test_scope",
        "truth_hypothesis",
        "uncertainty_model",
        "D2",
        "covariance_rank",
        "p_value",
        "log10_p_value",
        "directional_fbeam_z_nominal_truth_data",
        "directional_fbeam_z_nominal_truth_data_plus_mc",
        "directional_fbeam_z_variation_truth_data",
        "directional_fbeam_z_variation_truth_data_plus_mc",
        "six_bin_D2_nominal_truth_data",
        "six_bin_D2_nominal_truth_data_plus_mc",
        "six_bin_covariance_rank_nominal_truth_data_plus_mc",
        "six_bin_p_value_nominal_truth_data",
        "six_bin_p_value_nominal_truth_data_plus_mc",
        "six_bin_log10_p_value_nominal_truth_data_plus_mc",
    ]
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    for (strategy, channel, luminosity, observable, scenario), statistics in numerical.items():
        prefix = (
            f"{strategy}__{channel}__{_format_luminosity(luminosity)}__"
            f"{observable}__{scenario}"
        )
        for name in (
            "edges",
            "bin_cross_sections_pb",
            "event_second_pb",
            "mc_second_pb2",
            "yield",
            "data_covariance",
            "mc_covariance",
        ):
            arrays[f"{prefix}__{name}"] = np.asarray(statistics[name], dtype=np.float64)
        yields = np.asarray(statistics["yield"], dtype=np.float64)
        data_errors = np.sqrt(
            np.maximum(np.diag(np.asarray(statistics["data_covariance"])), 0.0)
        )
        mc_errors = np.sqrt(
            np.maximum(np.diag(np.asarray(statistics["mc_covariance"])), 0.0)
        )
        edges = np.asarray(statistics["edges"], dtype=np.float64)
        for index, value in enumerate(yields):
            rows.append(
                {
                    "record_type": "observable_yield",
                    "strategy": strategy,
                    "channel": channel,
                    "luminosity_fb": luminosity,
                    "observable": observable,
                    "scenario": scenario,
                    "bin": index + 1,
                    "low_edge": edges[index],
                    "high_edge": edges[index + 1],
                    "expected_yield": value,
                    "data_statistical_error": data_errors[index],
                    "mc_statistical_error": mc_errors[index],
                }
            )
    reference_id = str(payload["reference_scenario_id"])
    for strategy, strategy_values in payload["analyses"].items():
        for channel, channel_values in strategy_values.items():
            for observable, observable_values in channel_values["observables"].items():
                for scenario, comparison_values in observable_values[
                    "comparisons_to_reference"
                ].items():
                    for lumi_key, tests in comparison_values["luminosities"].items():
                        luminosity = float(lumi_key)
                        lumi_tag = _format_luminosity(luminosity)
                        prefix = (
                            f"{strategy}__{channel}__{lumi_tag}__{observable}__"
                            f"{scenario}__minus__{reference_id}__hypothesis_test"
                        )
                        for test_scope in ("shape_only", "rate_and_shape"):
                            scope_values = tests[test_scope]
                            test_variants = (
                                (
                                    "independent_scenarios",
                                    "mc_stat_only",
                                    scope_values["mc_stat_only"],
                                ),
                                (
                                    "nominal_truth",
                                    "data_stat_only",
                                    scope_values["nominal_truth"]["data_stat_only"],
                                ),
                                (
                                    "nominal_truth",
                                    "data_plus_mc_stat",
                                    scope_values["nominal_truth"]["data_plus_mc_stat"],
                                ),
                                (
                                    "variation_truth",
                                    "data_stat_only",
                                    scope_values["variation_truth"]["data_stat_only"],
                                ),
                                (
                                    "variation_truth",
                                    "data_plus_mc_stat",
                                    scope_values["variation_truth"]["data_plus_mc_stat"],
                                ),
                            )
                            for truth_hypothesis, uncertainty_model, distance in test_variants:
                                distance_prefix = (
                                    f"{prefix}__{test_scope}__{truth_hypothesis}__"
                                    f"{uncertainty_model}"
                                )
                                arrays[f"{distance_prefix}__D2"] = np.asarray(
                                    distance["D2"], dtype=np.float64
                                )
                                arrays[f"{distance_prefix}__rank"] = np.asarray(
                                    distance["covariance_rank"], dtype=np.int64
                                )
                                arrays[f"{distance_prefix}__p_value"] = np.asarray(
                                    distance["p_value"], dtype=np.float64
                                )
                                arrays[f"{distance_prefix}__log10_p_value"] = np.asarray(
                                    distance["log10_p_value"], dtype=np.float64
                                )
                                rows.append(
                                    {
                                        "record_type": "hypothesis_test",
                                        "strategy": strategy,
                                        "channel": channel,
                                        "luminosity_fb": luminosity,
                                        "observable": observable,
                                        "scenario": scenario,
                                        "reference_scenario": reference_id,
                                        "test_scope": test_scope,
                                        "truth_hypothesis": truth_hypothesis,
                                        "uncertainty_model": uncertainty_model,
                                        "D2": distance["D2"],
                                        "covariance_rank": distance["covariance_rank"],
                                        "p_value": distance["p_value"],
                                        "log10_p_value": distance["log10_p_value"],
                                    }
                                )
            for scenario, scenario_values in channel_values["scenarios"].items():
                prefix = f"{strategy}__{channel}__{scenario}"
                arrays[f"{prefix}__R"] = np.asarray(scenario_values["R"], dtype=np.float64)
                arrays[f"{prefix}__f_beam"] = np.asarray(
                    scenario_values["f_beam"], dtype=np.float64
                )
                arrays[f"{prefix}__mc_R_covariance"] = np.asarray(
                    scenario_values["mc_statistical_covariance"], dtype=np.float64
                )
                arrays[f"{prefix}__f_beam_mc_error"] = np.asarray(
                    scenario_values["f_beam_mc_statistical_error"], dtype=np.float64
                )
                for lumi_key, covariance in scenario_values[
                    "expected_statistical_covariance"
                ].items():
                    luminosity = float(lumi_key)
                    arrays[
                        f"{prefix}__data_R_covariance__{_format_luminosity(luminosity)}"
                    ] = np.asarray(covariance, dtype=np.float64)
                    arrays[
                        f"{prefix}__f_beam_data_error__{_format_luminosity(luminosity)}"
                    ] = np.asarray(
                        scenario_values["f_beam_statistical_error"][lumi_key],
                        dtype=np.float64,
                    )
                    data_errors = np.sqrt(
                        np.maximum(np.diag(np.asarray(covariance, dtype=np.float64)), 0.0)
                    )
                    mc_errors = np.sqrt(
                        np.maximum(
                            np.diag(
                                np.asarray(
                                    scenario_values["mc_statistical_covariance"],
                                    dtype=np.float64,
                                )
                            ),
                            0.0,
                        )
                    )
                    for index, fraction in enumerate(scenario_values["R"]):
                        rows.append(
                            {
                                "record_type": "differential_fraction",
                                "strategy": strategy,
                                "channel": channel,
                                "luminosity_fb": luminosity,
                                "observable": "folded_pull_angle",
                                "scenario": scenario,
                                "reference_scenario": reference_id,
                                "bin": index + 1,
                                "low_edge": PULL_BIN_EDGES[index],
                                "high_edge": PULL_BIN_EDGES[index + 1],
                                "data_statistical_error": data_errors[index],
                                "mc_statistical_error": mc_errors[index],
                                "R_i": fraction,
                                "f_beam": scenario_values["f_beam"],
                            }
                        )
            for scenario, difference in channel_values[
                "differences_from_reference"
            ].items():
                prefix = f"{strategy}__{channel}__{scenario}__minus__{reference_id}"
                arrays[f"{prefix}__delta_R"] = np.asarray(
                    difference["delta_R"], dtype=np.float64
                )
                arrays[f"{prefix}__delta_f_beam"] = np.asarray(
                    difference["delta_f_beam"], dtype=np.float64
                )
                arrays[f"{prefix}__independent_mc_R_difference_covariance"] = np.asarray(
                    difference["independent_mc_difference_covariance"], dtype=np.float64
                )
                for lumi_key, truth_values in difference["luminosities"].items():
                    luminosity = float(lumi_key)
                    nominal = truth_values["nominal_truth"]
                    variation = truth_values["variation_truth"]
                    lumi_tag = _format_luminosity(luminosity)
                    for truth_name, truth_summary in (
                        ("nominal_truth", nominal),
                        ("variation_truth", variation),
                    ):
                        arrays[
                            f"{prefix}__{truth_name}__directional_f_beam_Z_data__{lumi_tag}"
                        ] = np.asarray(
                            truth_summary[
                                "directional_f_beam_significance_data_stat_only"
                            ],
                            dtype=np.float64,
                        )
                        arrays[
                            f"{prefix}__{truth_name}__directional_f_beam_Z_data_plus_mc__{lumi_tag}"
                        ] = np.asarray(
                            truth_summary[
                                "directional_f_beam_significance_data_plus_mc_stat"
                            ],
                            dtype=np.float64,
                        )
                        for covariance_name, distance in (
                            ("data", truth_summary["six_bin_data_stat_only"]),
                            (
                                "data_plus_mc",
                                truth_summary["six_bin_data_plus_mc_stat"],
                            ),
                        ):
                            distance_prefix = (
                                f"{prefix}__{truth_name}__six_bin_{covariance_name}__{lumi_tag}"
                            )
                            arrays[f"{distance_prefix}__D2"] = np.asarray(
                                distance["D2"], dtype=np.float64
                            )
                            arrays[f"{distance_prefix}__rank"] = np.asarray(
                                distance["covariance_rank"], dtype=np.int64
                            )
                            arrays[f"{distance_prefix}__pseudoinverse_tolerance"] = np.asarray(
                                distance["pseudoinverse_tolerance"], dtype=np.float64
                            )
                            arrays[f"{distance_prefix}__p_value"] = np.asarray(
                                distance["p_value"], dtype=np.float64
                            )
                            arrays[f"{distance_prefix}__log10_p_value"] = np.asarray(
                                distance["log10_p_value"], dtype=np.float64
                            )
                    for index, delta_fraction in enumerate(difference["delta_R"]):
                        rows.append(
                            {
                                "record_type": "difference_from_reference",
                                "strategy": strategy,
                                "channel": channel,
                                "luminosity_fb": luminosity,
                                "observable": "folded_pull_angle",
                                "scenario": scenario,
                                "reference_scenario": reference_id,
                                "bin": index + 1,
                                "low_edge": PULL_BIN_EDGES[index],
                                "high_edge": PULL_BIN_EDGES[index + 1],
                                "delta_R_i": delta_fraction,
                                "delta_f_beam": difference["delta_f_beam"],
                                "directional_fbeam_z_nominal_truth_data": nominal[
                                    "directional_f_beam_significance_data_stat_only"
                                ],
                                "directional_fbeam_z_nominal_truth_data_plus_mc": nominal[
                                    "directional_f_beam_significance_data_plus_mc_stat"
                                ],
                                "directional_fbeam_z_variation_truth_data": variation[
                                    "directional_f_beam_significance_data_stat_only"
                                ],
                                "directional_fbeam_z_variation_truth_data_plus_mc": variation[
                                    "directional_f_beam_significance_data_plus_mc_stat"
                                ],
                                "six_bin_D2_nominal_truth_data": nominal[
                                    "six_bin_data_stat_only"
                                ]["D2"],
                                "six_bin_D2_nominal_truth_data_plus_mc": nominal[
                                    "six_bin_data_plus_mc_stat"
                                ]["D2"],
                                "six_bin_covariance_rank_nominal_truth_data_plus_mc": nominal[
                                    "six_bin_data_plus_mc_stat"
                                ]["covariance_rank"],
                                "six_bin_p_value_nominal_truth_data": nominal[
                                    "six_bin_data_stat_only"
                                ]["p_value"],
                                "six_bin_p_value_nominal_truth_data_plus_mc": nominal[
                                    "six_bin_data_plus_mc_stat"
                                ]["p_value"],
                                "six_bin_log10_p_value_nominal_truth_data_plus_mc": nominal[
                                    "six_bin_data_plus_mc_stat"
                                ]["log10_p_value"],
                            }
                        )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return json_path, csv_path, npz_path


def write_score_pull_diagnostic_artifacts(
    run_dir: Path,
    payload: Mapping[str, Any],
    numerical: Mapping[Tuple[str, float, str], Mapping[str, np.ndarray]],
) -> Tuple[Path, Path, Path]:
    json_path = run_dir / "summaries" / "score_pull_diagnostic.json"
    csv_path = run_dir / "summaries" / "score_pull_diagnostic.csv"
    npz_path = run_dir / "summaries" / "score_pull_diagnostic.npz"
    write_json_exclusive(json_path, payload)
    arrays: Dict[str, np.ndarray] = {}
    for (channel, luminosity, scenario), statistics in numerical.items():
        prefix = f"{channel}__{_format_luminosity(luminosity)}__{scenario}"
        for name in (
            "score_edges",
            "pull_edges",
            "bin_cross_sections_pb",
            "event_second_pb",
            "mc_second_pb2",
            "yield",
            "data_covariance",
            "mc_covariance",
            "common_cross_section_pb",
        ):
            arrays[f"{prefix}__{name}"] = np.asarray(
                statistics[name], dtype=np.float64
            )
    rows: List[Dict[str, Any]] = []
    for channel, channel_values in payload["channels"].items():
        for luminosity_key, luminosity_values in channel_values["luminosities"].items():
            luminosity = float(luminosity_key)
            for scenario, scenario_values in luminosity_values["variations"].items():
                for quantile in scenario_values["quantiles"]:
                    for pull_bin, delta_r in enumerate(quantile["delta_R"], start=1):
                        rows.append(
                            {
                                "record_type": "score_quantile_pull_bin",
                                "channel": channel,
                                "luminosity_fb": luminosity,
                                "variation": scenario,
                                "category_count": 1,
                                "score_bin": quantile["score_bin"],
                                "score_low": quantile["score_low"],
                                "score_high": quantile["score_high"],
                                "pull_bin": pull_bin,
                                "delta_R_i": delta_r,
                                "delta_f_beam": quantile["delta_f_beam"],
                                "D2_data_plus_mc_stat": quantile[
                                    "six_bin_data_plus_mc_stat"
                                ]["D2"],
                            }
                        )
            for category_count, scan in luminosity_values["category_scans"].items():
                for rank, candidate in enumerate(scan["candidates"], start=1):
                    rows.append(
                        {
                            "record_type": "category_partition_candidate",
                            "channel": channel,
                            "luminosity_fb": luminosity,
                            "category_count": int(category_count),
                            "partition_rank": rank,
                            "score_ranges": json.dumps(candidate["score_ranges"]),
                            "boundary_scores": json.dumps(candidate["boundary_scores"]),
                            "D2_data_plus_mc_stat": candidate[
                                "minimum_D2_data_plus_mc_stat"
                            ],
                            "D2_data_stat_only": candidate[
                                "minimum_D2_data_stat_only"
                            ],
                        }
                    )
    fields = sorted({key for row in rows for key in row})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return json_path, csv_path, npz_path


def write_histogram_npz(run_dir: Path, results: Sequence[SampleResult]) -> Path:
    destination = run_dir / "summaries" / "histograms.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays: Dict[str, np.ndarray] = {}
    for result in results:
        result_prefix = f"{result.strategy}__{result.spec.name}"
        for key, histogram in result.histograms.items():
            prefix = f"{result_prefix}__{key}"
            arrays[f"{prefix}__edges"] = histogram.edges
            arrays[f"{prefix}__sumw"] = histogram.sumw
            arrays[f"{prefix}__sumw2"] = histogram.sumw2
        arrays[f"{result_prefix}__pull_bin_sumw"] = result.pull_bin_sumw
        arrays[f"{result_prefix}__pull_event_second_sumw"] = result.pull_event_second_sumw
        arrays[f"{result_prefix}__pull_mc_second_sumw2"] = result.pull_mc_second_sumw2
        for observable, moments in result.pull_observable_moments.items():
            if observable not in PULL_OBSERVABLE_KEYS:
                raise ValueError(f"Unsupported pull-observable moments: {observable}")
            prefix = f"{result_prefix}__pull_moment__{observable}"
            arrays[f"{prefix}__edges"] = moments.edges
            arrays[f"{prefix}__bin_sumw"] = moments.bin_sumw
            arrays[f"{prefix}__event_second_sumw"] = moments.event_second_sumw
            arrays[f"{prefix}__mc_second_sumw2"] = moments.mc_second_sumw2
        if result.score_pull_moments is not None:
            prefix = f"{result_prefix}__score_pull"
            moments = result.score_pull_moments
            arrays[f"{prefix}__score_edges"] = moments.score_edges
            arrays[f"{prefix}__pull_edges"] = moments.pull_edges
            arrays[f"{prefix}__bin_sumw"] = moments.bin_sumw
            arrays[f"{prefix}__event_second_sumw"] = moments.event_second_sumw
            arrays[f"{prefix}__mc_second_sumw2"] = moments.mc_second_sumw2
            arrays[f"{prefix}__event_count"] = np.asarray(
                moments.event_count, dtype=np.int64
            )
    with destination.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return destination


def write_cutflow_files(
    run_dir: Path,
    channel: str,
    results: Sequence[SampleResult],
    luminosities: Sequence[float],
    strategy: str = "cutbased",
) -> Tuple[Path, Path]:
    channel_results = [
        result
        for result in results
        if result.spec.channel == channel and result.strategy == strategy
    ]
    csv_path = run_dir / "cutflows" / strategy / f"{channel}.csv"
    markdown_path = run_dir / "cutflows" / strategy / f"{channel}.md"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["strategy", "channel", "cut", "cut_label", "sample", "role", "raw_count", "sumw", "sumw2", "efficiency", "fiducial_cross_section_pb"]
    fields.extend(f"yield_{_format_luminosity(value)}" for value in luminosities)
    rows: List[Dict[str, Any]] = []
    for step in cutflow_steps(channel, strategy):
        for result in channel_results:
            stat = result.cutflow[step]
            all_sumw = result.cutflow["all_events"].sumw
            row: Dict[str, Any] = {
                "strategy": strategy,
                "channel": channel,
                "cut": step,
                "cut_label": CUT_LABELS[step],
                "sample": result.spec.name,
                "role": result.spec.role,
                "raw_count": stat.raw_count,
                "sumw": stat.sumw,
                "sumw2": stat.sumw2,
                "efficiency": stat.sumw / all_sumw if all_sumw else math.nan,
                "fiducial_cross_section_pb": result.spec.cross_section_pb * stat.sumw / result.generated_sumw,
            }
            for luminosity in luminosities:
                row[f"yield_{_format_luminosity(luminosity)}"] = normalization_factor(
                    luminosity, result.spec.cross_section_pb, result.generated_sumw
                ) * stat.sumw
            rows.append(row)
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    header = "| " + " | ".join(fields) + " |"
    separator = "| " + " | ".join("---" for _ in fields) + " |"
    lines = [f"# {channel.capitalize()} {strategy} cutflow", "", header, separator]
    for row in rows:
        values = []
        for field_name in fields:
            value = row[field_name]
            if isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    write_text_exclusive(markdown_path, "\n".join(lines) + "\n")
    return csv_path, markdown_path


def _step_values(values: np.ndarray) -> np.ndarray:
    return np.r_[values, values[-1]] if len(values) else values


def _save_figure_exclusive(figure: Any, path: Path, **kwargs: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite plot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, **kwargs)


def _place_plot_legend(
    axis: Any,
    *,
    outside: bool = False,
    location: str = "best",
    columns: int = 1,
) -> Any:
    """Draw a high-contrast legend inside the data panel by default."""
    handles, labels = axis.get_legend_handles_labels()
    if not handles:
        return None
    common = {
        "handles": handles,
        "labels": labels,
        "ncol": max(1, int(columns)),
        "frameon": True,
        "framealpha": 1.0,
        "facecolor": "#f8fafc",
        "edgecolor": "#aeb8c6",
        "labelcolor": "#172033",
        "fancybox": True,
        "borderpad": 0.65,
        "handlelength": 2.2,
        "handletextpad": 0.65,
    }
    if outside:
        legend = axis.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            **common,
        )
    else:
        legend = axis.legend(loc=location, **common)
    legend.set_zorder(20)
    legend_handles = getattr(
        legend,
        "legend_handles",
        getattr(legend, "legendHandles", ()),
    )
    for handle in legend_handles:
        if hasattr(handle, "set_markeredgecolor"):
            handle.set_markeredgecolor("#172033")
        if hasattr(handle, "set_markeredgewidth"):
            handle.set_markeredgewidth(0.6)
        if hasattr(handle, "set_edgecolor"):
            handle.set_edgecolor("#172033")
        if hasattr(handle, "set_linewidth") and handle.__class__.__name__ != "Line2D":
            handle.set_linewidth(0.6)
    return legend


def generate_plots(
    run_dir: Path,
    results: Sequence[SampleResult],
    luminosities: Sequence[float],
    partial: bool,
) -> List[Dict[str, Any]]:
    matplotlib_config = Path(tempfile.gettempdir()) / "pullpheno-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "figure.figsize": (7.2, 5.2),
            "savefig.facecolor": "white",
        }
    )
    plot_records: List[Dict[str, Any]] = []
    planned_paths = set()
    registry = plot_registry()
    strategies = [name for name in ANALYSIS_STRATEGIES if any(r.strategy == name for r in results)]
    for strategy in strategies:
        for spec in registry:
            channel_results = sorted(
                (
                    result
                    for result in results
                    if result.spec.channel == spec.channel and result.strategy == strategy
                ),
                key=lambda result: result.spec.stack_order,
            )
            if not channel_results:
                continue
            stage = spec.stage if strategy == "cutbased" else "xgboost"
            display_title = spec.title + (" · XGBoost selection" if strategy == "xgboost" else "")
            for luminosity in luminosities:
                lumi_tag = _format_luminosity(luminosity)
                relative_base = (
                    Path("plots") / strategy / "yields" / lumi_tag / spec.channel / stage / spec.key
                )
                for extension in ("png", "pdf"):
                    candidate_path = str(relative_base.with_suffix(f".{extension}"))
                    if candidate_path in planned_paths:
                        raise RuntimeError(f"Duplicate planned plot path: {candidate_path}")
                    planned_paths.add(candidate_path)
                fig, ax = plt.subplots()
                edges = spec.edges
                widths = np.diff(edges)
                bottom = np.zeros(len(edges) - 1)
                variance = np.zeros_like(bottom)
                for result in channel_results:
                    histogram = result.histograms[spec.key]
                    factor = normalization_factor(
                        luminosity, result.spec.cross_section_pb, result.generated_sumw
                    )
                    values = factor * histogram.sumw
                    variance += factor * factor * histogram.sumw2
                    ax.bar(
                        edges[:-1], values, width=widths, align="edge", bottom=bottom,
                        color=result.spec.color, edgecolor="none", linewidth=0.0,
                        label=result.spec.label,
                    )
                    bottom += values
                uncertainty = np.sqrt(np.maximum(variance, 0.0))
                ax.fill_between(
                    edges,
                    _step_values(bottom - uncertainty),
                    _step_values(bottom + uncertainty),
                    step="post", color="#20242b", alpha=0.18, linewidth=0.0,
                    label="MC statistical uncertainty",
                )
                ax.set_xlim(edges[0], edges[-1])
                ax.set_xlabel(spec.xlabel)
                ax.set_ylabel(rf"Expected events / bin at {luminosity:g} fb$^{{-1}}$")
                ax.set_title(display_title, loc="left", pad=34, fontweight="semibold")
                subtitle = "Particle level · Herwig · 13.6 TeV"
                if strategy == "xgboost":
                    scopes = {item.application_scope for item in channel_results}
                    scope_labels = {
                        "held_out_test_only": "held-out test",
                        "five_fold_out_of_fold_all_events": "5-fold out-of-fold",
                        "five_fold_routed_independent_events": "frozen 5-fold ensemble",
                        "legacy_single_model_independent_events": "legacy frozen model",
                        "all_independent_events": "independent application",
                    }
                    subtitle += " · " + (
                        scope_labels.get(next(iter(scopes)), "independent application")
                        if len(scopes) == 1
                        else "mixed application scopes"
                    )
                if partial:
                    subtitle += " · PARTIAL RUN"
                ax.text(1.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                        color="#555b66", ha="right")
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-4, 4))
                formatter.set_useOffset(False)
                ax.yaxis.set_major_formatter(formatter)
                _place_plot_legend(ax)
                fig.tight_layout()
                png_path = run_dir / relative_base.with_suffix(".png")
                pdf_path = run_dir / relative_base.with_suffix(".pdf")
                _save_figure_exclusive(fig, png_path, dpi=170, bbox_inches="tight")
                _save_figure_exclusive(fig, pdf_path, bbox_inches="tight")
                plt.close(fig)
                plot_records.append({
                    "strategy": strategy, "observable": spec.key, "title": display_title,
                    "channel": spec.channel, "stage": stage, "kind": "yield",
                    "luminosity_fb": luminosity,
                    "png": png_path.relative_to(run_dir).as_posix(),
                    "pdf": pdf_path.relative_to(run_dir).as_posix(),
                })

            relative_base = Path("plots") / strategy / "shapes" / spec.channel / stage / spec.key
            for extension in ("png", "pdf"):
                candidate_path = str(relative_base.with_suffix(f".{extension}"))
                if candidate_path in planned_paths:
                    raise RuntimeError(f"Duplicate planned plot path: {candidate_path}")
                planned_paths.add(candidate_path)
            fig, ax = plt.subplots()
            for result in channel_results:
                histogram = result.histograms[spec.key]
                integral = histogram.integral
                if integral == 0.0:
                    continue
                values = histogram.sumw / integral
                errors = np.sqrt(np.maximum(histogram.sumw2, 0.0)) / abs(integral)
                centers = 0.5 * (histogram.edges[:-1] + histogram.edges[1:])
                ax.stairs(values, histogram.edges, color=result.spec.color, linewidth=1.8,
                           label=result.spec.label)
                ax.errorbar(centers, values, yerr=errors, fmt="none", color=result.spec.color,
                            linewidth=0.8, capsize=1.2)
            ax.set_xlim(spec.edges[0], spec.edges[-1])
            ax.set_xlabel(spec.xlabel)
            ax.set_ylabel("Fraction / bin")
            ax.set_title(display_title, loc="left", pad=34, fontweight="semibold")
            ax.text(1.0, 1.015, "Particle level · unit-area process comparison",
                    transform=ax.transAxes, fontsize=9, color="#555b66", ha="right")
            _place_plot_legend(ax)
            fig.tight_layout()
            png_path = run_dir / relative_base.with_suffix(".png")
            pdf_path = run_dir / relative_base.with_suffix(".pdf")
            _save_figure_exclusive(fig, png_path, dpi=170, bbox_inches="tight")
            _save_figure_exclusive(fig, pdf_path, bbox_inches="tight")
            plt.close(fig)
            plot_records.append({
                "strategy": strategy, "observable": spec.key, "title": display_title,
                "channel": spec.channel, "stage": stage, "kind": "shape",
                "luminosity_fb": None,
                "png": png_path.relative_to(run_dir).as_posix(),
                "pdf": pdf_path.relative_to(run_dir).as_posix(),
            })
    return plot_records


def generate_ri_plots(
    run_dir: Path,
    results: Sequence[SampleResult],
    luminosities: Sequence[float],
    partial: bool,
) -> List[Dict[str, Any]]:
    import matplotlib.pyplot as plt

    records: List[Dict[str, Any]] = []
    centers = 0.5 * (PULL_BIN_EDGES[:-1] + PULL_BIN_EDGES[1:])
    for strategy in ANALYSIS_STRATEGIES:
        if not any(result.strategy == strategy for result in results):
            continue
        for channel in ("higgs", "z"):
            channel_results = sorted(
                (
                    result for result in results
                    if result.strategy == strategy and result.spec.channel == channel
                ),
                key=lambda result: result.spec.stack_order,
            )
            if not channel_results:
                continue
            title_suffix = "cuts" if strategy == "cutbased" else "XGBoost"
            relative_base = (
                Path("plots") / strategy / "shapes" / channel / "ri" / "ri_processes"
            )
            fig, ax = plt.subplots()
            for result in channel_results:
                scale_pb = result.spec.cross_section_pb / result.generated_sumw
                statistics = differential_pull_statistics(
                    scale_pb * result.pull_bin_sumw,
                    scale_pb * result.pull_event_second_sumw,
                    scale_pb * scale_pb * result.pull_mc_second_sumw2,
                    luminosities,
                )
                if statistics["R"] is None:
                    continue
                fractions = np.asarray(statistics["R"])
                errors = np.asarray(statistics["mc_statistical_errors"])
                ax.errorbar(
                    centers, fractions, yerr=errors, marker="o", markersize=3.8,
                    linewidth=1.4, capsize=2.0, color=result.spec.color,
                    label=result.spec.label,
                )
            ax.axvline(0.5 * math.pi, color="#657086", linestyle="--", linewidth=1.0)
            ax.set_xlim(0.0, math.pi)
            ax.set_xlabel(r"$|\theta_s|$ [rad]")
            ax.set_ylabel(r"$R_i$")
            ax.set_title(f"Differential pull fractions · {title_suffix}", loc="left", pad=34,
                         fontweight="semibold")
            ax.text(1.0, 1.015, "Process comparison · MC errors", transform=ax.transAxes,
                    fontsize=9, color="#555b66", ha="right")
            _place_plot_legend(ax)
            fig.tight_layout()
            png_path = run_dir / relative_base.with_suffix(".png")
            pdf_path = run_dir / relative_base.with_suffix(".pdf")
            _save_figure_exclusive(fig, png_path, dpi=170, bbox_inches="tight")
            _save_figure_exclusive(fig, pdf_path, bbox_inches="tight")
            plt.close(fig)
            records.append({
                "strategy": strategy, "observable": "R_i", "title": f"Differential R_i · {title_suffix}",
                "channel": channel, "stage": strategy, "kind": "shape", "luminosity_fb": None,
                "png": png_path.relative_to(run_dir).as_posix(),
                "pdf": pdf_path.relative_to(run_dir).as_posix(),
            })

            total = channel_pull_summary(channel, results, luminosities, strategy)
            differential = total["differential"]
            if differential["R"] is None:
                continue
            fractions = np.asarray(differential["R"], dtype=np.float64)
            mc_errors = np.asarray(differential["mc_statistical_errors"], dtype=np.float64)
            for luminosity in luminosities:
                key = str(float(luminosity))
                statistical_errors = np.asarray(
                    differential["expected_statistical_errors"][key], dtype=np.float64
                )
                relative_base = (
                    Path("plots") / strategy / "yields" / _format_luminosity(luminosity)
                    / channel / "ri" / "ri_total"
                )
                fig, ax = plt.subplots()
                ax.fill_between(
                    PULL_BIN_EDGES,
                    _step_values(fractions - mc_errors),
                    _step_values(fractions + mc_errors),
                    step="post", color="#20242b", alpha=0.18, linewidth=0.0,
                    label="MC statistical uncertainty",
                )
                ax.errorbar(
                    centers, fractions, yerr=statistical_errors, fmt="o", color="#172033",
                    markersize=4.5, capsize=2.5, linewidth=1.1,
                    label=rf"Expected statistical uncertainty, {luminosity:g} fb$^{{-1}}$",
                )
                ax.stairs(fractions, PULL_BIN_EDGES, color="#275dad", linewidth=1.5,
                          label="Total prediction")
                ax.axvline(0.5 * math.pi, color="#657086", linestyle="--", linewidth=1.0)
                ax.set_xlim(0.0, math.pi)
                ax.set_xlabel(r"$|\theta_s|$ [rad]")
                ax.set_ylabel(r"$R_i$")
                ax.set_title(f"Total differential pull fractions · {title_suffix}", loc="left",
                             pad=34, fontweight="semibold")
                subtitle = "Event-level two-jet covariance"
                if partial:
                    subtitle += " · PARTIAL RUN"
                ax.text(1.0, 1.015, subtitle, transform=ax.transAxes, fontsize=9,
                        color="#555b66", ha="right")
                _place_plot_legend(ax)
                fig.tight_layout()
                png_path = run_dir / relative_base.with_suffix(".png")
                pdf_path = run_dir / relative_base.with_suffix(".pdf")
                _save_figure_exclusive(fig, png_path, dpi=170, bbox_inches="tight")
                _save_figure_exclusive(fig, pdf_path, bbox_inches="tight")
                plt.close(fig)
                records.append({
                    "strategy": strategy, "observable": "R_i", "title": f"Total differential R_i · {title_suffix}",
                    "channel": channel, "stage": strategy, "kind": "differential",
                    "luminosity_fb": luminosity,
                    "png": png_path.relative_to(run_dir).as_posix(),
                    "pdf": pdf_path.relative_to(run_dir).as_posix(),
                })
    return records


def generate_comparison_plots(
    run_dir: Path,
    sources: Sequence[ComparisonSource],
    analyses: Sequence[str],
    luminosities: Sequence[float],
    numerical: Mapping[Tuple[str, str, float, str, str], Mapping[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Draw total comparisons and nominal-stack overlays for pull predictions."""
    matplotlib_config = Path(tempfile.gettempdir()) / "pullpheno-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    records: List[Dict[str, Any]] = []
    registry = {(spec.channel, spec.key): spec for spec in plot_registry()}
    planned_paths = set()
    reference_id = sources[0].scenario.identifier
    for strategy in analyses:
        for channel in ("higgs", "z"):
            for luminosity in luminosities:
                folded_spec = registry[(channel, "folded_pull_angle")]
                reference_folded = numerical[
                    (
                        strategy,
                        channel,
                        float(luminosity),
                        "folded_pull_angle",
                        reference_id,
                    )
                ]
                folded_edges = np.asarray(reference_folded["edges"], dtype=np.float64)
                folded_centers = 0.5 * (folded_edges[:-1] + folded_edges[1:])
                folded_widths = np.diff(folded_edges)
                relative_base = (
                    Path("plots")
                    / "comparison"
                    / strategy
                    / "yields"
                    / _format_luminosity(luminosity)
                    / channel
                    / "reference-stack"
                    / "folded_pull_angle"
                )
                for extension in ("png", "pdf"):
                    candidate = str(relative_base.with_suffix(f".{extension}"))
                    if candidate in planned_paths:
                        raise RuntimeError(
                            f"Duplicate comparison plot path: {candidate}"
                        )
                    planned_paths.add(candidate)

                figure, axis = plt.subplots(figsize=(7.2, 5.2))
                reference_results = sorted(
                    (
                        result
                        for result in sources[0].results
                        if result.strategy == strategy
                        and result.spec.channel == channel
                    ),
                    key=lambda result: result.spec.stack_order,
                )
                if not reference_results:
                    raise ValueError(
                        f"Reference run has no {channel} {strategy} process results"
                    )
                stack_total = np.zeros(len(folded_edges) - 1, dtype=np.float64)
                for result in reference_results:
                    histogram = result.histograms["folded_pull_angle"]
                    if not np.array_equal(histogram.edges, folded_edges):
                        raise ValueError(
                            f"Reference folded-angle binning differs for {result.spec.name}"
                        )
                    factor = normalization_factor(
                        luminosity,
                        result.spec.cross_section_pb,
                        result.generated_sumw,
                    )
                    process_values = factor * histogram.sumw
                    axis.bar(
                        folded_edges[:-1],
                        process_values,
                        width=folded_widths,
                        align="edge",
                        bottom=stack_total,
                        color=result.spec.color,
                        edgecolor="none",
                        linewidth=0.0,
                        label=result.spec.label,
                        zorder=2,
                    )
                    stack_total += process_values

                reference_total = np.asarray(
                    reference_folded["yield"], dtype=np.float64
                )
                if not np.allclose(
                    stack_total,
                    reference_total,
                    rtol=1.0e-11,
                    atol=1.0e-8,
                ):
                    raise RuntimeError(
                        f"Reference stack does not close to the total {channel} "
                        f"{strategy} folded-angle prediction"
                    )
                reference_mc_covariance = np.asarray(
                    reference_folded["mc_covariance"], dtype=np.float64
                )
                reference_mc_error = np.sqrt(
                    np.maximum(np.diag(reference_mc_covariance), 0.0)
                )
                axis.fill_between(
                    folded_edges,
                    _step_values(reference_total - reference_mc_error),
                    _step_values(reference_total + reference_mc_error),
                    step="post",
                    color="#20242b",
                    alpha=0.18,
                    linewidth=0.0,
                    label=(
                        f"{sources[0].scenario.label} total MC statistical uncertainty"
                    ),
                    zorder=5,
                )

                variations = sources[1:]
                variation_folded_statistics = []
                for variation_index, source in enumerate(variations):
                    scenario_id = source.scenario.identifier
                    statistics = numerical[
                        (
                            strategy,
                            channel,
                            float(luminosity),
                            "folded_pull_angle",
                            scenario_id,
                        )
                    ]
                    values = np.asarray(statistics["yield"], dtype=np.float64)
                    covariance = np.asarray(
                        statistics["mc_covariance"], dtype=np.float64
                    )
                    errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
                    offset_fraction = (
                        variation_index - 0.5 * (len(variations) - 1)
                    ) * min(0.12, 0.55 / max(len(variations), 1))
                    shifted_centers = folded_centers + offset_fraction * folded_widths
                    color = str(source.scenario.color)
                    marker = COMPARISON_MARKERS[
                        (variation_index + 1) % len(COMPARISON_MARKERS)
                    ]
                    variation_folded_statistics.append(
                        (
                            source,
                            values,
                            covariance,
                            errors,
                            shifted_centers,
                            color,
                            marker,
                        )
                    )
                    axis.errorbar(
                        shifted_centers,
                        values,
                        yerr=errors,
                        fmt=marker,
                        linestyle="none",
                        markersize=5.2,
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=1.35,
                        capsize=2.6,
                        elinewidth=1.15,
                        color=color,
                        label=f"{source.scenario.label} total (MC stat.)",
                        zorder=12,
                    )

                display_title = folded_spec.title + (
                    " · XGBoost selection" if strategy == "xgboost" else ""
                )
                axis.set_xlim(folded_edges[0], folded_edges[-1])
                axis.set_ylim(bottom=0.0)
                axis.set_xlabel(folded_spec.xlabel)
                axis.set_ylabel(
                    rf"Expected events / bin at {luminosity:g} fb$^{{-1}}$"
                )
                axis.set_title(
                    display_title, loc="left", pad=34, fontweight="semibold"
                )
                subtitle = (
                    f"Particle level · {sources[0].scenario.label} processes stacked · "
                    "CR totals as points"
                )
                if bool(sources[0].metadata.get("partial")):
                    subtitle += " · PARTIAL RUN"
                axis.text(
                    1.0,
                    1.015,
                    subtitle,
                    transform=axis.transAxes,
                    fontsize=8.8,
                    color="#555b66",
                    ha="right",
                )
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-4, 4))
                formatter.set_useOffset(False)
                axis.yaxis.set_major_formatter(formatter)
                axis.yaxis.get_offset_text().set_x(-0.08)
                axis.yaxis.get_offset_text().set_y(1.01)
                axis.grid(False)
                _place_plot_legend(axis)
                figure.tight_layout()
                png_path = run_dir / relative_base.with_suffix(".png")
                pdf_path = run_dir / relative_base.with_suffix(".pdf")
                _save_figure_exclusive(
                    figure, png_path, dpi=170, bbox_inches="tight"
                )
                _save_figure_exclusive(figure, pdf_path, bbox_inches="tight")
                plt.close(figure)
                records.append(
                    {
                        "strategy": strategy,
                        "observable": "folded_pull_angle",
                        "title": (
                            f"{display_title} · reference process stack + CR totals"
                        ),
                        "channel": channel,
                        "stage": "comparison",
                        "kind": "reference-stack-plus-total",
                        "uncertainty": "mc-stat",
                        "luminosity_fb": luminosity,
                        "png": png_path.relative_to(run_dir).as_posix(),
                        "pdf": pdf_path.relative_to(run_dir).as_posix(),
                    }
                )

                ratio_relative_base = (
                    Path("plots")
                    / "comparison"
                    / strategy
                    / "yields"
                    / _format_luminosity(luminosity)
                    / channel
                    / "reference-stack-ratio"
                    / "folded_pull_angle"
                )
                for extension in ("png", "pdf"):
                    candidate = str(ratio_relative_base.with_suffix(f".{extension}"))
                    if candidate in planned_paths:
                        raise RuntimeError(
                            f"Duplicate comparison plot path: {candidate}"
                        )
                    planned_paths.add(candidate)

                ratio_figure, (ratio_stack_axis, ratio_axis) = plt.subplots(
                    2,
                    1,
                    figsize=(7.5, 6.5),
                    sharex=True,
                    constrained_layout=True,
                    gridspec_kw={"height_ratios": (3.1, 1.15), "hspace": 0.06},
                )
                ratio_stack_total = np.zeros(
                    len(folded_edges) - 1, dtype=np.float64
                )
                for result in reference_results:
                    histogram = result.histograms["folded_pull_angle"]
                    factor = normalization_factor(
                        luminosity,
                        result.spec.cross_section_pb,
                        result.generated_sumw,
                    )
                    process_values = factor * histogram.sumw
                    ratio_stack_axis.bar(
                        folded_edges[:-1],
                        process_values,
                        width=folded_widths,
                        align="edge",
                        bottom=ratio_stack_total,
                        color=result.spec.color,
                        edgecolor="none",
                        linewidth=0.0,
                        label=result.spec.label,
                        zorder=2,
                    )
                    ratio_stack_total += process_values
                if not np.allclose(
                    ratio_stack_total,
                    reference_total,
                    rtol=1.0e-11,
                    atol=1.0e-8,
                ):
                    raise RuntimeError(
                        f"Reference ratio-panel stack does not close to the total "
                        f"{channel} {strategy} folded-angle prediction"
                    )
                ratio_stack_axis.fill_between(
                    folded_edges,
                    _step_values(reference_total - reference_mc_error),
                    _step_values(reference_total + reference_mc_error),
                    step="post",
                    color="#20242b",
                    alpha=0.18,
                    linewidth=0.0,
                    label=(
                        f"{sources[0].scenario.label} total MC statistical uncertainty"
                    ),
                    zorder=5,
                )

                ratio_extent_values: List[float] = []
                for (
                    source,
                    values,
                    covariance,
                    errors,
                    shifted_centers,
                    color,
                    marker,
                ) in variation_folded_statistics:
                    ratio_stack_axis.errorbar(
                        shifted_centers,
                        values,
                        yerr=errors,
                        fmt=marker,
                        linestyle="none",
                        markersize=5.2,
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=1.35,
                        capsize=2.6,
                        elinewidth=1.15,
                        color=color,
                        label=f"{source.scenario.label} total (MC stat.)",
                        zorder=12,
                    )
                    ratios, ratio_errors = propagated_independent_ratio_errors(
                        values,
                        covariance,
                        reference_total,
                        reference_mc_covariance,
                        include_reference=True,
                    )
                    finite_ratio = np.isfinite(ratios) & np.isfinite(ratio_errors)
                    ratio_extent_values.extend(
                        (ratios[finite_ratio] - ratio_errors[finite_ratio]).tolist()
                    )
                    ratio_extent_values.extend(
                        (ratios[finite_ratio] + ratio_errors[finite_ratio]).tolist()
                    )
                    ratio_axis.errorbar(
                        shifted_centers,
                        ratios,
                        yerr=ratio_errors,
                        fmt=marker,
                        linestyle="none",
                        markersize=4.6,
                        markerfacecolor="white",
                        markeredgecolor=color,
                        markeredgewidth=1.25,
                        capsize=2.4,
                        elinewidth=1.05,
                        color=color,
                        zorder=12,
                    )

                ratio_stack_axis.set_xlim(folded_edges[0], folded_edges[-1])
                ratio_stack_axis.set_ylim(bottom=0.0)
                ratio_stack_axis.set_ylabel(
                    rf"Expected events / bin at {luminosity:g} fb$^{{-1}}$"
                )
                ratio_stack_axis.set_title(
                    display_title, loc="left", pad=34, fontweight="semibold"
                )
                ratio_subtitle = (
                    f"Particle level · {sources[0].scenario.label} processes stacked · "
                    "CR totals and ratio with MC uncertainty"
                )
                if bool(sources[0].metadata.get("partial")):
                    ratio_subtitle += " · PARTIAL RUN"
                ratio_stack_axis.text(
                    1.0,
                    1.015,
                    ratio_subtitle,
                    transform=ratio_stack_axis.transAxes,
                    fontsize=8.8,
                    color="#555b66",
                    ha="right",
                )
                ratio_formatter = ScalarFormatter(useMathText=True)
                ratio_formatter.set_scientific(True)
                ratio_formatter.set_powerlimits((-4, 4))
                ratio_formatter.set_useOffset(False)
                ratio_stack_axis.yaxis.set_major_formatter(ratio_formatter)
                ratio_stack_axis.yaxis.get_offset_text().set_x(-0.08)
                ratio_stack_axis.yaxis.get_offset_text().set_y(1.01)
                ratio_stack_axis.grid(False)
                _place_plot_legend(ratio_stack_axis)

                ratio_axis.axhline(1.0, color="#657086", linewidth=1.0, zorder=1)
                ratio_axis.set_ylabel("CR / default")
                ratio_axis.set_xlabel(folded_spec.xlabel)
                ratio_axis.set_xlim(folded_edges[0], folded_edges[-1])
                ratio_axis.grid(False)
                if ratio_extent_values:
                    deviation = max(
                        0.05,
                        max(abs(value - 1.0) for value in ratio_extent_values) * 1.12,
                    )
                    ratio_axis.set_ylim(1.0 - deviation, 1.0 + deviation)
                ratio_figure.align_ylabels((ratio_stack_axis, ratio_axis))
                ratio_png_path = run_dir / ratio_relative_base.with_suffix(".png")
                ratio_pdf_path = run_dir / ratio_relative_base.with_suffix(".pdf")
                _save_figure_exclusive(
                    ratio_figure,
                    ratio_png_path,
                    dpi=170,
                    bbox_inches="tight",
                )
                _save_figure_exclusive(
                    ratio_figure, ratio_pdf_path, bbox_inches="tight"
                )
                plt.close(ratio_figure)
                records.append(
                    {
                        "strategy": strategy,
                        "observable": "folded_pull_angle",
                        "title": (
                            f"{display_title} · reference process stack + CR totals "
                            "+ independent-MC ratio"
                        ),
                        "channel": channel,
                        "stage": "comparison",
                        "kind": "reference-stack-plus-total-ratio",
                        "uncertainty": "independent-mc-stat",
                        "luminosity_fb": luminosity,
                        "png": ratio_png_path.relative_to(run_dir).as_posix(),
                        "pdf": ratio_pdf_path.relative_to(run_dir).as_posix(),
                    }
                )

                for observable in PULL_OBSERVABLE_KEYS:
                    plot_spec = registry[(channel, observable)]
                    reference = numerical[
                        (strategy, channel, float(luminosity), observable, reference_id)
                    ]
                    edges = np.asarray(reference["edges"], dtype=np.float64)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    bin_widths = np.diff(edges)
                    for uncertainty_kind, covariance_key, include_reference in (
                        ("data-stat", "data_covariance", False),
                        ("mc-stat", "mc_covariance", True),
                    ):
                        relative_base = (
                            Path("plots")
                            / "comparison"
                            / strategy
                            / "yields"
                            / _format_luminosity(luminosity)
                            / channel
                            / uncertainty_kind
                            / observable
                        )
                        for extension in ("png", "pdf"):
                            candidate = str(relative_base.with_suffix(f".{extension}"))
                            if candidate in planned_paths:
                                raise RuntimeError(f"Duplicate comparison plot path: {candidate}")
                            planned_paths.add(candidate)
                        figure, (axis, ratio_axis) = plt.subplots(
                            2,
                            1,
                            figsize=(7.5, 6.5),
                            sharex=True,
                            constrained_layout=True,
                            gridspec_kw={"height_ratios": (3.1, 1.15), "hspace": 0.06},
                        )
                        reference_yield = np.asarray(reference["yield"], dtype=np.float64)
                        reference_covariance = np.asarray(
                            reference[covariance_key], dtype=np.float64
                        )
                        scenario_count = len(sources)
                        ratio_extent_values: List[float] = []
                        for source_index, source in enumerate(sources):
                            scenario_id = source.scenario.identifier
                            statistics = numerical[
                                (
                                    strategy,
                                    channel,
                                    float(luminosity),
                                    observable,
                                    scenario_id,
                                )
                            ]
                            values = np.asarray(statistics["yield"], dtype=np.float64)
                            covariance = np.asarray(
                                statistics[covariance_key], dtype=np.float64
                            )
                            errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
                            offset_fraction = (
                                source_index - 0.5 * (scenario_count - 1)
                            ) * min(0.12, 0.55 / max(scenario_count, 1))
                            shifted_centers = centers + offset_fraction * bin_widths
                            color = str(source.scenario.color)
                            marker = COMPARISON_MARKERS[
                                source_index % len(COMPARISON_MARKERS)
                            ]
                            axis.errorbar(
                                shifted_centers,
                                values,
                                yerr=errors,
                                fmt=marker,
                                linestyle="none",
                                markersize=4.5,
                                capsize=2.4,
                                elinewidth=1.05,
                                color=color,
                                label=source.scenario.label,
                            )
                            ratios, ratio_errors = propagated_independent_ratio_errors(
                                values,
                                covariance,
                                reference_yield,
                                reference_covariance,
                                include_reference=(
                                    include_reference and source_index != 0
                                ),
                            )
                            finite_ratio = np.isfinite(ratios) & np.isfinite(ratio_errors)
                            ratio_extent_values.extend(
                                (ratios[finite_ratio] - ratio_errors[finite_ratio]).tolist()
                            )
                            ratio_extent_values.extend(
                                (ratios[finite_ratio] + ratio_errors[finite_ratio]).tolist()
                            )
                            ratio_axis.errorbar(
                                shifted_centers,
                                ratios,
                                yerr=ratio_errors,
                                fmt=marker,
                                linestyle="none",
                                markersize=4.0,
                                capsize=2.2,
                                elinewidth=0.95,
                                color=color,
                            )
                        axis.set_ylabel(
                            rf"Expected events / bin at {luminosity:g} fb$^{{-1}}$"
                        )
                        axis.set_title(
                            f"{plot_spec.title} · total CR-scenario comparison",
                            loc="left",
                            pad=28,
                            fontweight="semibold",
                        )
                        subtitle = (
                            "Projected counting uncertainty"
                            if uncertainty_kind == "data-stat"
                            else "Finite-MC uncertainty · independent samples"
                        )
                        axis.text(
                            1.0,
                            1.015,
                            f"{strategy} · {channel} · {subtitle}",
                            transform=axis.transAxes,
                            fontsize=8.8,
                            color="#555b66",
                            ha="right",
                        )
                        formatter = ScalarFormatter(useMathText=True)
                        formatter.set_scientific(True)
                        formatter.set_powerlimits((-4, 4))
                        formatter.set_useOffset(False)
                        axis.yaxis.set_major_formatter(formatter)
                        axis.yaxis.get_offset_text().set_x(-0.08)
                        axis.yaxis.get_offset_text().set_y(1.01)
                        axis.grid(False)
                        ratio_axis.grid(False)
                        _place_plot_legend(axis)
                        ratio_axis.axhline(1.0, color="#657086", linewidth=1.0)
                        ratio_axis.set_ylabel("Scenario / ref.")
                        ratio_axis.set_xlabel(plot_spec.xlabel)
                        ratio_axis.set_xlim(edges[0], edges[-1])
                        if ratio_extent_values:
                            deviation = max(
                                0.05,
                                max(abs(value - 1.0) for value in ratio_extent_values) * 1.12,
                            )
                            ratio_axis.set_ylim(1.0 - deviation, 1.0 + deviation)
                        figure.align_ylabels((axis, ratio_axis))
                        png_path = run_dir / relative_base.with_suffix(".png")
                        pdf_path = run_dir / relative_base.with_suffix(".pdf")
                        _save_figure_exclusive(
                            figure, png_path, dpi=170, bbox_inches="tight"
                        )
                        _save_figure_exclusive(figure, pdf_path, bbox_inches="tight")
                        plt.close(figure)
                        records.append(
                            {
                                "strategy": strategy,
                                "observable": observable,
                                "title": (
                                    f"{plot_spec.title} · total scenarios · "
                                    f"{uncertainty_kind}"
                                ),
                                "channel": channel,
                                "stage": "comparison",
                                "kind": uncertainty_kind,
                                "luminosity_fb": luminosity,
                                "png": png_path.relative_to(run_dir).as_posix(),
                                "pdf": pdf_path.relative_to(run_dir).as_posix(),
                            }
                        )
    return records


def generate_score_pull_diagnostic_plots(
    run_dir: Path,
    sources: Sequence[ComparisonSource],
    luminosities: Sequence[float],
    payload: Mapping[str, Any],
    numerical: Mapping[Tuple[str, float, str], Mapping[str, np.ndarray]],
) -> List[Dict[str, Any]]:
    """Plot the exploratory score-quantile dependence of the absolute pull shape."""
    matplotlib_config = Path(tempfile.gettempdir()) / "pullpheno-matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    records: List[Dict[str, Any]] = []
    reference_id = sources[0].scenario.identifier
    source_by_id = {source.scenario.identifier: source for source in sources}
    singleton_ranges = tuple((index, index + 1) for index in range(SCORE_PULL_BIN_COUNT))
    for channel in ("higgs", "z"):
        for luminosity in luminosities:
            luminosity_key = str(float(luminosity))
            luminosity_payload = payload["channels"][channel]["luminosities"][
                luminosity_key
            ]
            reference = numerical[(channel, float(luminosity), reference_id)]
            reference_conditional = conditional_score_pull_statistics(
                reference, singleton_ranges
            )
            reference_r = reference_conditional["R"].reshape(
                SCORE_PULL_BIN_COUNT, PULL_BIN_COUNT
            )
            base = (
                Path("plots")
                / "comparison"
                / "xgboost"
                / "score-pull"
                / _format_luminosity(luminosity)
                / channel
            )
            for scenario_id, variation_payload in luminosity_payload[
                "variations"
            ].items():
                source = source_by_id[scenario_id]
                variation = numerical[(channel, float(luminosity), scenario_id)]
                variation_conditional = conditional_score_pull_statistics(
                    variation, singleton_ranges
                )
                variation_r = variation_conditional["R"].reshape(
                    SCORE_PULL_BIN_COUNT, PULL_BIN_COUNT
                )
                delta = variation_r - reference_r
                difference_covariance = (
                    reference_conditional["data_covariance"]
                    + reference_conditional["mc_covariance"]
                    + variation_conditional["mc_covariance"]
                )
                errors = np.sqrt(
                    np.maximum(np.diag(difference_covariance), 0.0)
                ).reshape(SCORE_PULL_BIN_COUNT, PULL_BIN_COUNT)
                standardized = np.divide(
                    delta,
                    errors,
                    out=np.full_like(delta, np.nan),
                    where=errors > 0.0,
                )
                delta_limit = max(float(np.max(np.abs(delta))), 1.0e-6)
                standardized_finite = np.abs(standardized[np.isfinite(standardized)])
                standardized_limit = max(
                    float(np.max(standardized_finite)) if standardized_finite.size else 0.0,
                    0.1,
                )
                figure, axes = plt.subplots(
                    1, 3, figsize=(13.2, 4.9), constrained_layout=True
                )
                images = [
                    axes[0].imshow(
                        reference_r,
                        origin="lower",
                        aspect="auto",
                        cmap="Blues",
                        vmin=0.0,
                    ),
                    axes[1].imshow(
                        delta,
                        origin="lower",
                        aspect="auto",
                        cmap="RdBu_r",
                        norm=TwoSlopeNorm(vcenter=0.0, vmin=-delta_limit, vmax=delta_limit),
                    ),
                    axes[2].imshow(
                        standardized,
                        origin="lower",
                        aspect="auto",
                        cmap="RdBu_r",
                        norm=TwoSlopeNorm(
                            vcenter=0.0,
                            vmin=-standardized_limit,
                            vmax=standardized_limit,
                        ),
                    ),
                ]
                titles = (
                    "Reference conditional $R_i$",
                    r"Variation $-$ reference: $\Delta R_i$",
                    r"$\Delta R_i / \sigma_{\mathrm{data+MC}}$",
                )
                colorbar_labels = (r"$R_i$", r"$\Delta R_i$", "signed displacement")
                for axis, image, title, colorbar_label in zip(
                    axes, images, titles, colorbar_labels
                ):
                    axis.set_title(title, fontsize=10.2)
                    axis.set_xticks(range(PULL_BIN_COUNT), [f"$R_{index}$" for index in range(1, 7)])
                    axis.set_yticks(
                        range(SCORE_PULL_BIN_COUNT),
                        [f"Q{index}" for index in range(1, SCORE_PULL_BIN_COUNT + 1)],
                    )
                    axis.set_xlabel(r"$|\theta_s|$ bin")
                    axis.grid(False)
                    figure.colorbar(image, ax=axis, shrink=0.82, label=colorbar_label)
                axes[0].set_ylabel("Frozen nominal score quantile (low → high)")
                figure.suptitle(
                    f"{channel.capitalize()} score × pull map · {source.scenario.label}\n"
                    f"common selection · {luminosity:g} fb$^{{-1}}$ · exploratory",
                    fontweight="semibold",
                )
                scenario_slug = sanitize_run_name(scenario_id) or "variation"
                map_base = base / f"score_pull_map__{scenario_slug}"
                map_png = run_dir / map_base.with_suffix(".png")
                map_pdf = run_dir / map_base.with_suffix(".pdf")
                _save_figure_exclusive(figure, map_png, dpi=170, bbox_inches="tight")
                _save_figure_exclusive(figure, map_pdf, bbox_inches="tight")
                plt.close(figure)
                records.append(
                    {
                        "strategy": "xgboost",
                        "observable": "score_pull_map",
                        "title": f"Score × pull map · {source.scenario.label}",
                        "channel": channel,
                        "stage": "diagnostic",
                        "kind": "score-pull-map",
                        "luminosity_fb": luminosity,
                        "png": map_png.relative_to(run_dir).as_posix(),
                        "pdf": map_pdf.relative_to(run_dir).as_posix(),
                    }
                )

                quantiles = variation_payload["quantiles"]
                centers = np.arange(1, SCORE_PULL_BIN_COUNT + 1, dtype=np.float64)
                delta_fbeam = np.asarray(
                    [values["delta_f_beam"] for values in quantiles], dtype=np.float64
                )
                fbeam_errors = np.asarray(
                    [
                        values["delta_f_beam_data_plus_mc_error"]
                        for values in quantiles
                    ],
                    dtype=np.float64,
                )
                figure, axis = plt.subplots(figsize=(7.4, 4.8), constrained_layout=True)
                axis.errorbar(
                    centers,
                    delta_fbeam,
                    yerr=fbeam_errors,
                    fmt="o",
                    color=str(source.scenario.color),
                    capsize=2.5,
                    markersize=4.5,
                    label=source.scenario.label,
                )
                axis.axhline(0.0, color="#657086", linewidth=1.0)
                axis.set_xticks(centers)
                axis.set_xlabel("Frozen nominal XGBoost-score quantile (low → high)")
                axis.set_ylabel(r"$\Delta f_{\mathrm{beam}}$ in score quantile")
                axis.set_title(
                    f"{channel.capitalize()} conditional beam fraction by score",
                    loc="left",
                    fontweight="semibold",
                )
                axis.text(
                    1.0,
                    1.015,
                    f"common selection · {luminosity:g} fb$^{{-1}}$ · data+MC stat.",
                    transform=axis.transAxes,
                    fontsize=8.8,
                    color="#555b66",
                    ha="right",
                )
                axis.grid(False)
                _place_plot_legend(axis)
                fbeam_base = base / f"delta_fbeam_by_score__{scenario_slug}"
                fbeam_png = run_dir / fbeam_base.with_suffix(".png")
                fbeam_pdf = run_dir / fbeam_base.with_suffix(".pdf")
                _save_figure_exclusive(figure, fbeam_png, dpi=170, bbox_inches="tight")
                _save_figure_exclusive(figure, fbeam_pdf, bbox_inches="tight")
                plt.close(figure)
                records.append(
                    {
                        "strategy": "xgboost",
                        "observable": "delta_fbeam_by_score",
                        "title": f"Conditional Δfbeam by score · {source.scenario.label}",
                        "channel": channel,
                        "stage": "diagnostic",
                        "kind": "score-pull-fbeam",
                        "luminosity_fb": luminosity,
                        "png": fbeam_png.relative_to(run_dir).as_posix(),
                        "pdf": fbeam_pdf.relative_to(run_dir).as_posix(),
                    }
                )

            two_scan = luminosity_payload["category_scans"]["2"]
            three_scan = luminosity_payload["category_scans"]["3"]
            two_candidates = sorted(
                two_scan["candidates"], key=lambda values: values["score_ranges"][0][1]
            )
            two_boundaries = np.asarray(
                [values["score_ranges"][0][1] for values in two_candidates], dtype=np.int64
            )
            two_d2 = np.asarray(
                [values["minimum_D2_data_plus_mc_stat"] for values in two_candidates],
                dtype=np.float64,
            )
            three_grid = np.full(
                (SCORE_PULL_BIN_COUNT - 1, SCORE_PULL_BIN_COUNT - 1),
                np.nan,
                dtype=np.float64,
            )
            for candidate in three_scan["candidates"]:
                first_cut = candidate["score_ranges"][0][1]
                second_cut = candidate["score_ranges"][1][1]
                three_grid[first_cut - 1, second_cut - 1] = candidate[
                    "minimum_D2_data_plus_mc_stat"
                ]
            figure, (left, right) = plt.subplots(
                1, 2, figsize=(11.2, 4.6), constrained_layout=True
            )
            left.plot(two_boundaries, two_d2, marker="o", color="#275DAD")
            best_two_cut = two_scan["recommended"]["score_ranges"][0][1]
            left.axvline(best_two_cut, color="#D05A47", linestyle="--", linewidth=1.2)
            left.set_xlabel("Boundary after nominal score quantile")
            left.set_ylabel(r"worst-case conditional-shape $D^2$")
            left.set_title("Two common score categories")
            left.grid(False)
            image = right.imshow(
                three_grid,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                interpolation="none",
            )
            best_three = three_scan["recommended"]["score_ranges"]
            right.plot(best_three[1][1] - 1, best_three[0][1] - 1, "x", color="white", ms=8, mew=2)
            right.set_xticks(range(SCORE_PULL_BIN_COUNT - 1), range(1, SCORE_PULL_BIN_COUNT))
            right.set_yticks(range(SCORE_PULL_BIN_COUNT - 1), range(1, SCORE_PULL_BIN_COUNT))
            right.set_xlabel("Second boundary after quantile")
            right.set_ylabel("First boundary after quantile")
            right.set_title("Three common score categories")
            right.grid(False)
            figure.colorbar(image, ax=right, label=r"worst-case $D^2$")
            figure.suptitle(
                f"{channel.capitalize()} fixed score-category scan · {luminosity:g} fb$^{{-1}}$\n"
                "conditional pull shapes · nominal truth · data+independent MC stat.",
                fontweight="semibold",
            )
            scan_base = base / "score_category_scan"
            scan_png = run_dir / scan_base.with_suffix(".png")
            scan_pdf = run_dir / scan_base.with_suffix(".pdf")
            _save_figure_exclusive(figure, scan_png, dpi=170, bbox_inches="tight")
            _save_figure_exclusive(figure, scan_pdf, bbox_inches="tight")
            plt.close(figure)
            records.append(
                {
                    "strategy": "xgboost",
                    "observable": "score_category_scan",
                    "title": "Common two/three score-category scan",
                    "channel": channel,
                    "stage": "diagnostic",
                    "kind": "score-category-scan",
                    "luminosity_fb": luminosity,
                    "png": scan_png.relative_to(run_dir).as_posix(),
                    "pdf": scan_pdf.relative_to(run_dir).as_posix(),
                }
            )
    return records


def write_ri_artifacts(
    run_dir: Path,
    results: Sequence[SampleResult],
    luminosities: Sequence[float],
) -> Tuple[Path, Path]:
    csv_path = run_dir / "summaries" / "ri.csv"
    npz_path = run_dir / "summaries" / "ri_covariances.npz"
    rows: List[Dict[str, Any]] = []
    arrays: Dict[str, np.ndarray] = {}
    for strategy in ANALYSIS_STRATEGIES:
        strategy_results = [result for result in results if result.strategy == strategy]
        if not strategy_results:
            continue
        for channel in ("higgs", "z"):
            scopes: List[Tuple[str, str, Dict[str, Any]]] = []
            total = channel_pull_summary(channel, results, luminosities, strategy)["differential"]
            scopes.append(("total", "total", total))
            for result in strategy_results:
                if result.spec.channel != channel:
                    continue
                scale_pb = result.spec.cross_section_pb / result.generated_sumw
                scopes.append(("process", result.spec.name, differential_pull_statistics(
                    scale_pb * result.pull_bin_sumw,
                    scale_pb * result.pull_event_second_sumw,
                    scale_pb * scale_pb * result.pull_mc_second_sumw2,
                    luminosities,
                )))
            for scope, name, statistics in scopes:
                if statistics["R"] is None:
                    continue
                prefix = f"{strategy}__{channel}__{scope}__{name}"
                arrays[f"{prefix}__mc_covariance"] = np.asarray(
                    statistics["mc_statistical_covariance"], dtype=np.float64
                )
                for luminosity in luminosities:
                    arrays[f"{prefix}__expected_covariance__{_format_luminosity(luminosity)}"] = np.asarray(
                        statistics["expected_statistical_covariance"][str(float(luminosity))],
                        dtype=np.float64,
                    )
                for index, fraction in enumerate(statistics["R"]):
                    row: Dict[str, Any] = {
                        "strategy": strategy, "channel": channel, "scope": scope,
                        "sample": name, "bin": index + 1,
                        "low_edge": statistics["bin_edges"][index],
                        "high_edge": statistics["bin_edges"][index + 1],
                        "R_i": fraction,
                        "mc_statistical_error": statistics["mc_statistical_errors"][index],
                        "f_beam": statistics["f_beam"],
                    }
                    for luminosity in luminosities:
                        row[f"statistical_error_{_format_luminosity(luminosity)}"] = (
                            statistics["expected_statistical_errors"][str(float(luminosity))][index]
                        )
                    rows.append(row)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No differential R_i rows were produced")
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return csv_path, npz_path


def generate_xgboost_diagnostic_plots(
    run_dir: Path,
    diagnostics: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    import matplotlib.pyplot as plt

    records: List[Dict[str, Any]] = []
    for channel, payload in diagnostics.items():
        base = Path("plots") / "xgboost" / "diagnostics" / channel

        fig, ax = plt.subplots()
        ax.plot(
            payload["roc_fpr"],
            payload["roc_tpr"],
            color="#275dad",
            linewidth=2.0,
            label=f"{payload.get('roc_label', 'Physics application')} AUC = {payload['auc']:.4f}",
        )
        ax.plot((0.0, 1.0), (0.0, 1.0), color="#8a93a3", linestyle="--", linewidth=1.0)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Background efficiency")
        ax.set_ylabel("Signal efficiency")
        ax.set_title(f"{channel.capitalize()} XGBoost ROC", loc="left", fontweight="semibold")
        _place_plot_legend(ax, outside=False, location="lower right")
        fig.tight_layout()
        roc_png = run_dir / base / "roc.png"
        roc_pdf = run_dir / base / "roc.pdf"
        _save_figure_exclusive(fig, roc_png, dpi=170, bbox_inches="tight")
        _save_figure_exclusive(fig, roc_pdf, bbox_inches="tight")
        plt.close(fig)
        records.append({
            "strategy": "xgboost", "observable": "xgboost_roc",
            "title": f"{channel.capitalize()} XGBoost ROC", "channel": channel,
            "stage": "diagnostic", "kind": "diagnostic", "luminosity_fb": None,
            "png": roc_png.relative_to(run_dir).as_posix(),
            "pdf": roc_pdf.relative_to(run_dir).as_posix(),
        })

        split_payloads = [
            (name, values) for name, values in payload["splits"].items()
            if len(values["scores"])
        ]
        fig, axes = plt.subplots(1, len(split_payloads), figsize=(5.2 * len(split_payloads), 4.6),
                                 squeeze=False)
        score_edges = np.linspace(0.0, 1.0, 41)
        for axis, (split_name, values) in zip(axes[0], split_payloads):
            for label, color, name in ((0, "#4E79A7", "Background"), (1, "#59A14F", "Signal")):
                mask = values["labels"] == label
                weights = values["weights"][mask]
                total = float(np.sum(weights, dtype=np.float64))
                normalized = weights / total if total > 0.0 else weights
                axis.hist(values["scores"][mask], bins=score_edges, weights=normalized,
                          histtype="step", linewidth=1.7, color=color, label=name)
            thresholds = [float(value) for value in payload.get("thresholds", ())]
            for threshold_index, threshold in enumerate(thresholds):
                axis.axvline(
                    threshold,
                    color="#b64a3a",
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.45,
                    label="Frozen fold cuts" if threshold_index == 0 else None,
                )
            axis.set_xlim(0.0, 1.0)
            axis.set_xlabel("XGBoost signal score")
            axis.set_ylabel("Weighted fraction / bin")
            axis.set_title(split_name.capitalize())
            _place_plot_legend(axis, outside=False, location="upper right")
        fig.suptitle(f"{channel.capitalize()} classifier-score distributions", fontweight="semibold")
        fig.tight_layout()
        score_png = run_dir / base / "score_distributions.png"
        score_pdf = run_dir / base / "score_distributions.pdf"
        _save_figure_exclusive(fig, score_png, dpi=170, bbox_inches="tight")
        _save_figure_exclusive(fig, score_pdf, bbox_inches="tight")
        plt.close(fig)
        records.append({
            "strategy": "xgboost", "observable": "xgboost_score",
            "title": f"{channel.capitalize()} score distributions", "channel": channel,
            "stage": "diagnostic", "kind": "diagnostic", "luminosity_fb": None,
            "png": score_png.relative_to(run_dir).as_posix(),
            "pdf": score_pdf.relative_to(run_dir).as_posix(),
        })

        names = list(xgbtools.FEATURE_NAMES)
        values = np.asarray([payload["feature_importance"].get(name, 0.0) for name in names])
        order = np.argsort(values)
        fig, ax = plt.subplots()
        ax.barh(np.asarray(names)[order], values[order], color="#275dad")
        ax.set_xlabel("XGBoost gain")
        ax.set_title(f"{channel.capitalize()} feature importance", loc="left",
                     fontweight="semibold")
        fig.tight_layout()
        importance_png = run_dir / base / "feature_importance.png"
        importance_pdf = run_dir / base / "feature_importance.pdf"
        _save_figure_exclusive(fig, importance_png, dpi=170, bbox_inches="tight")
        _save_figure_exclusive(fig, importance_pdf, bbox_inches="tight")
        plt.close(fig)
        records.append({
            "strategy": "xgboost", "observable": "xgboost_feature_importance",
            "title": f"{channel.capitalize()} feature importance", "channel": channel,
            "stage": "diagnostic", "kind": "diagnostic", "luminosity_fb": None,
            "png": importance_png.relative_to(run_dir).as_posix(),
            "pdf": importance_pdf.relative_to(run_dir).as_posix(),
        })
    return records


def _html_cutflow_tables(
    results: Sequence[SampleResult], luminosities: Sequence[float]
) -> str:
    sections: List[str] = []
    for strategy in ANALYSIS_STRATEGIES:
        if not any(result.strategy == strategy for result in results):
            continue
        for channel in ("higgs", "z"):
            channel_results = sorted(
                (
                    result for result in results
                    if result.spec.channel == channel and result.strategy == strategy
                ),
                key=lambda result: result.spec.stack_order,
            )
            for luminosity in luminosities:
                header = "".join(
                    f"<th>{html.escape(result.spec.label)}</th>" for result in channel_results
                )
                body_rows = []
                for step in cutflow_steps(channel, strategy):
                    cells = []
                    for result in channel_results:
                        factor = normalization_factor(
                            luminosity, result.spec.cross_section_pb, result.generated_sumw
                        )
                        cells.append(f"<td>{factor * result.cutflow[step].sumw:,.4g}</td>")
                    body_rows.append(
                        f"<tr><th>{html.escape(CUT_LABELS[step])}</th>{''.join(cells)}</tr>"
                    )
                sections.append(
                    f"<details><summary>{strategy.capitalize()} · {channel.capitalize()} cutflow · "
                    f"{luminosity:g} fb<sup>−1</sup></summary>"
                    f"<div class=\"table-wrap\"><table><thead><tr><th>Selection</th>{header}</tr></thead>"
                    f"<tbody>{''.join(body_rows)}</tbody></table></div></details>"
                )
    return "".join(sections)


def generate_run_index(
    run_dir: Path,
    run_metadata: Mapping[str, Any],
    results: Sequence[SampleResult],
    summaries: Sequence[Mapping[str, Any]],
    plot_records: Sequence[Mapping[str, Any]],
    luminosities: Sequence[float],
    xgboost_metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    cards: List[str] = []
    for record in plot_records:
        lumi = "shape" if record["luminosity_fb"] is None else _format_luminosity(float(record["luminosity_fb"]))
        strategy = str(record.get("strategy", "cutbased"))
        alt = f"{record['title']} ({record['channel']}, {record['stage']}, {record['kind']})"
        cards.append(
            "<article class=\"plot-card\" "
            f"data-strategy=\"{html.escape(strategy)}\" "
            f"data-channel=\"{html.escape(str(record['channel']))}\" "
            f"data-stage=\"{html.escape(str(record['stage']))}\" "
            f"data-kind=\"{html.escape(str(record['kind']))}\" "
            f"data-lumi=\"{html.escape(lumi)}\" "
            f"data-search=\"{html.escape((str(record['title']) + ' ' + str(record['observable'])).lower())}\">"
            f"<a class=\"thumb-link\" href=\"{html.escape(str(record['png']))}\"><img loading=\"lazy\" src=\"{html.escape(str(record['png']))}\" alt=\"{html.escape(alt)}\"></a>"
            "<div class=\"plot-copy\">"
            f"<div class=\"badges\"><span>{html.escape(strategy)}</span><span>{html.escape(str(record['channel']).upper())}</span><span>{html.escape(str(record['stage']))}</span><span>{html.escape(lumi)}</span></div>"
            f"<h3>{html.escape(str(record['title']))}</h3>"
            f"<p><a href=\"{html.escape(str(record['png']))}\">PNG</a><a href=\"{html.escape(str(record['pdf']))}\">PDF</a></p>"
            "</div></article>"
        )
    summary_cards: List[str] = []
    strategies = [name for name in ANALYSIS_STRATEGIES if any(r.strategy == name for r in results)]
    for strategy in strategies:
        for channel in ("higgs", "z"):
            channel_summary = channel_pull_summary(channel, results, luminosities, strategy)
            if channel_summary["f_beam"] is None:
                continue
            fbeam = f"{channel_summary['f_beam']:.5f}"
            mc_error = f"±{channel_summary['f_beam_mc_statistical_error']:.4g}"
            stat_rows = "".join(
                f"<div><dt>{luminosity:g} fb<sup>−1</sup> data stat.</dt>"
                f"<dd>±{channel_summary['f_beam_statistical_error'][str(float(luminosity))]:.4g}</dd></div>"
                for luminosity in luminosities
            )
            summary_cards.append(
                "<article class=\"metric total\">"
                f"<div class=\"badges\"><span>{strategy}</span><span>{channel}</span></div>"
                f"<h3>Total {channel.capitalize()} prediction</h3><dl>"
                f"<div><dt>Selected cross section</dt><dd>{channel_summary['selected_cross_section_pb']:.6g} pb</dd></div>"
                f"<div><dt>f<sub>beam</sub></dt><dd>{fbeam}</dd></div>"
                f"<div><dt>MC statistical</dt><dd>{mc_error}</dd></div>{stat_rows}"
                f"<div><dt>ΣR<sub>i</sub></dt><dd>{channel_summary['differential']['sum_R']:.8f}</dd></div>"
                f"<div><dt>L/R asymmetry</dt><dd>{channel_summary['left_right_asymmetry']:+.4f}</dd></div>"
                "</dl></article>"
            )
    partial_class = "partial" if run_metadata.get("partial") else "full"
    if run_metadata.get("derived_from_run"):
        status_label = f"derived {partial_class} run"
        derivation_html = (
            "<br><strong>Derived from:</strong> "
            f"{html.escape(str(run_metadata['derived_from_run']['run_id']))} "
            "(stored histogram replot; no event reread)"
        )
    elif run_metadata.get("resumed_from_checkpoint"):
        status_label = f"resumed {partial_class} run"
        derivation_html = (
            "<br><strong>Resumed from:</strong> ROOT-pass checkpoint of "
            f"{html.escape(str(run_metadata['resumed_from_checkpoint']['source_run_id']))} "
            "(no ROOT event reread)"
        )
    else:
        status_label = f"{partial_class} run"
        derivation_html = ""
    raw_scenario = run_metadata.get("scenario")
    scenario_html = ""
    if raw_scenario:
        parameters = ", ".join(
            f"{key}={value}" for key, value in raw_scenario.get("parameters", {}).items()
        ) or "default generator settings"
        scenario_html = (
            "<br><strong>Scenario:</strong> "
            f"{html.escape(str(raw_scenario['label']))} "
            f"(<code>{html.escape(str(raw_scenario['identifier']))}</code>; "
            f"{html.escape(parameters)})"
        )
    cutflow_html = _html_cutflow_tables(results, luminosities)
    cutflow_links = " · ".join(
        f'<a href="cutflows/{strategy}/{channel}.csv">{strategy} {channel} CSV</a>'
        for strategy in strategies for channel in ("higgs", "z")
    )
    xgb_html = ""
    if xgboost_metadata:
        rows = []
        xgb_scopes = {
            str(values.get("application_scope", ""))
            for values in xgboost_metadata.get("channels", {}).values()
        }
        for channel, values in xgboost_metadata.get("channels", {}).items():
            performance = values.get(
                "out_of_fold", values.get("application", values.get("test", {}))
            )
            if values.get("models"):
                threshold_summary = values["score_threshold_summary"]
                threshold_text = (
                    f"{threshold_summary['minimum']:.4g}–{threshold_summary['maximum']:.4g} "
                    f"(mean {threshold_summary['mean']:.4g})"
                )
                model_links = " ".join(
                    f'<a href="{html.escape(str(model["model_path"]))}">fold {int(model["fold"]) + 1}</a>'
                    for model in values["models"]
                )
            else:
                threshold_text = f"{values['score_threshold']:.6g}"
                model_links = f'<a href="{html.escape(values["model_path"])}">model</a>'
            rows.append(
                f"<tr><th>{html.escape(channel.capitalize())}</th>"
                f"<td>{threshold_text}</td>"
                f"<td>{values['validation_optimum']['significance']:.5g}</td>"
                f"<td>{performance['weighted_auc']:.5f}</td>"
                f"<td>{performance['signal_efficiency']:.5f}</td>"
                f"<td>{performance['background_efficiency']:.5f}</td>"
                f"<td>{model_links}</td></tr>"
            )
        if xgb_scopes == {"five_fold_out_of_fold_all_events"}:
            application_description = (
                "Nominal physics plots use rotating five-fold out-of-fold predictions: every "
                "event is tested once by a model and threshold that used neither that event nor "
                "its fold."
            )
        elif xgb_scopes == {"five_fold_routed_independent_events"}:
            application_description = (
                "Independent events are deterministically routed once through the frozen "
                "five-model, five-threshold nominal ensemble."
            )
        elif xgb_scopes == {"held_out_test_only"}:
            application_description = (
                "This legacy nominal run uses only its held-out test subset with inverse weights."
            )
        else:
            application_description = "The complete application scope is recorded in the metadata."
        xgb_html = (
            '<section class="panel"><h2>XGBoost models</h2>'
            '<p>Features: <code>' + html.escape(", ".join(xgboost_metadata["feature_names"])) + '</code>. '
            + application_description + '</p>'
            '<div class="table-wrap"><table><thead><tr><th>Channel</th><th>Score cut</th>'
            '<th>Validation S/√(S+B)</th><th>Physics AUC</th><th>Signal eff.</th><th>Background eff.</th><th>Artifacts</th>'
            '</tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
            '<p><a href="summaries/xgboost.json">Complete XGBoost metadata</a></p></section>'
        )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PullPheno · {html.escape(str(run_metadata['run_id']))}</title>
<style>
:root{{--ink:#172033;--muted:#657086;--line:#dce2eb;--paper:#fff;--wash:#f3f6fa;--accent:#275dad;--accent2:#0d8274}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
a{{color:var(--accent);text-decoration:none}} a:hover,a:focus-visible{{text-decoration:underline}} header{{background:linear-gradient(125deg,#12213c,#244b78 62%,#11776d);color:white;padding:3.2rem max(1.2rem,calc((100vw - 1280px)/2)) 2.8rem}}
header p{{max-width:70rem;color:#dce9f8}} .eyebrow{{font-size:.76rem;letter-spacing:.12em;text-transform:uppercase;font-weight:750}} h1{{font-size:clamp(2rem,5vw,4rem);line-height:1.04;margin:.35rem 0}} main{{max-width:1320px;margin:auto;padding:1.5rem}}
.status{{display:inline-flex;border:1px solid #ffffff55;border-radius:999px;padding:.3rem .7rem;font-weight:700;text-transform:uppercase;font-size:.72rem}} .status.partial{{background:#9b5d08}} .status.full{{background:#087568}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:1rem;margin:1.5rem 0}} .metric,.panel,.plot-card{{background:var(--paper);border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #2435510c}}
.metric{{padding:1rem}} .metric h3{{margin:.1rem 0 .8rem}} dl{{margin:0}} dl div{{display:flex;justify-content:space-between;gap:1rem;border-top:1px solid var(--line);padding:.38rem 0}} dt{{color:var(--muted)}} dd{{font-variant-numeric:tabular-nums;margin:0;font-weight:700}}
.panel{{padding:1rem;margin:1rem 0}} details{{border-top:1px solid var(--line);padding:.75rem 0}} summary{{cursor:pointer;font-weight:750}} .table-wrap{{overflow:auto;margin-top:.7rem}} table{{border-collapse:collapse;width:100%;font-size:.86rem}} th,td{{border-bottom:1px solid var(--line);padding:.48rem .6rem;text-align:right;white-space:nowrap}} th:first-child{{text-align:left;position:sticky;left:0;background:white}}
.controls{{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:.7rem;position:sticky;top:0;background:#f3f6faed;backdrop-filter:blur(8px);padding:.9rem 0;z-index:2}} label{{font-size:.78rem;color:var(--muted);font-weight:700}} select,input{{display:block;width:100%;margin-top:.2rem;border:1px solid #bfc8d5;border-radius:8px;background:white;padding:.55rem;color:var(--ink)}}
.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));gap:1rem}} .plot-card{{overflow:hidden}} .plot-card[hidden]{{display:none}} .thumb-link{{display:block;background:#e9edf3;aspect-ratio:1.38;overflow:hidden}} .thumb-link img{{width:100%;height:100%;object-fit:contain;background:white;transition:transform .18s ease}} .thumb-link:hover img{{transform:scale(1.012)}} .plot-copy{{padding:.9rem}} .plot-copy h3{{margin:.5rem 0 .25rem;font-size:1rem}} .plot-copy p{{margin:0;display:flex;gap:1rem}} .badges{{display:flex;gap:.35rem;flex-wrap:wrap}} .badges span{{font-size:.68rem;text-transform:uppercase;letter-spacing:.06em;border-radius:999px;background:#e7eef8;color:#274b79;padding:.18rem .45rem;font-weight:750}} #empty{{padding:2rem;text-align:center;color:var(--muted)}} footer{{padding:2.5rem 1.5rem;text-align:center;color:var(--muted)}}
@media(max-width:760px){{.controls{{grid-template-columns:1fr 1fr;position:static}}.controls label:last-child{{grid-column:1/-1}}}}
@media print{{header{{background:white;color:black;padding:1rem}}.controls{{display:none}}.plot-card{{break-inside:avoid}}body{{background:white}}}}
</style>
</head>
<body>
<header><div class="eyebrow">PullPheno particle-level analysis</div><h1>Signed pull-angle results</h1>
<span class="status {partial_class}">{status_label}</span>
<p><strong>Run:</strong> {html.escape(str(run_metadata['run_id']))}<br><strong>Completed:</strong> {html.escape(str(run_metadata['completed_utc']))}{scenario_html}{derivation_html}<br><strong>Configuration:</strong> anti-kT R=0.4, leading-pT tagging jets; cut-based and XGBoost branches share the common selection.</p></header>
<main>
<section><h2>Numerical summary</h2><div class="metrics">{''.join(summary_cards)}</div>
<p class="method-note"><strong>Uncertainties:</strong> both projected-data and current-MC errors use event-level six-bin covariance matrices, retaining the correlation between the two tagging jets. The MC term does not decrease with displayed luminosity.
<strong>Zero-|t| jets:</strong> raw tagging jets whose pull-vector magnitude is exactly zero. Their angle is undefined, so this analysis maps it to zero and reports the count explicitly.</p></section>
{xgb_html}
<section class="panel"><h2>Cutflows and data</h2>{cutflow_html}<p>{cutflow_links} · <a href="summaries/analysis.json">Complete JSON summary</a> · <a href="summaries/histograms.npz">Histogram arrays</a> · <a href="summaries/ri.csv">R<sub>i</sub> CSV</a> · <a href="summaries/ri_covariances.npz">R<sub>i</sub> covariances</a> · <a href="{RUN_LOG_NAME}">Run log</a></p></section>
<section><h2>Plot gallery</h2><div class="controls">
<label>Analysis<select id="strategy"><option value="all">All</option>{''.join(f'<option value="{name}">{name}</option>' for name in strategies)}</select></label>
<label>Channel<select id="channel"><option value="all">All</option><option value="higgs">Higgs</option><option value="z">Z</option></select></label>
<label>Stage<select id="stage"><option value="all">All</option><option value="common">Common</option><option value="vbf">VBF cuts</option><option value="xgboost">XGBoost</option><option value="diagnostic">Diagnostic</option><option value="cutbased">Cut based</option></select></label>
<label>Kind<select id="kind"><option value="all">All</option><option value="yield">Expected yields</option><option value="shape">Unit-area shapes</option><option value="differential">Differential Rᵢ</option><option value="diagnostic">Diagnostics</option></select></label>
<label>Luminosity<select id="lumi"><option value="all">All</option>{''.join(f'<option value="{_format_luminosity(value)}">{value:g} fb⁻¹</option>' for value in luminosities)}<option value="shape">Shapes</option></select></label>
<label>Search<input id="search" type="search" placeholder="mjj, pull, photon…"></label>
</div><div class="gallery">{''.join(cards)}</div><p id="empty" hidden>No plots match these filters.</p></section>
</main><footer>Self-contained PullPheno result bundle · existing run artifacts are immutable.</footer>
<script>
const controls=["strategy","channel","stage","kind","lumi"].map(id=>document.getElementById(id));const search=document.getElementById("search");const cards=[...document.querySelectorAll(".plot-card")];const empty=document.getElementById("empty");function applyFilters(){{const values=Object.fromEntries(controls.map(el=>[el.id,el.value]));const query=search.value.trim().toLowerCase();let shown=0;for(const card of cards){{const match=(values.strategy==="all"||card.dataset.strategy===values.strategy)&&(values.channel==="all"||card.dataset.channel===values.channel)&&(values.stage==="all"||card.dataset.stage===values.stage)&&(values.kind==="all"||card.dataset.kind===values.kind)&&(values.lumi==="all"||card.dataset.lumi===values.lumi)&&(!query||card.dataset.search.includes(query));card.hidden=!match;if(match)shown++;}}empty.hidden=shown!==0;}}controls.forEach(el=>el.addEventListener("change",applyFilters));search.addEventListener("input",applyFilters);
</script></body></html>
"""
    destination = run_dir / "index.html"
    write_text_exclusive(destination, document)
    return destination


def generate_comparison_index(
    run_dir: Path,
    run_metadata: Mapping[str, Any],
    sources: Sequence[ComparisonSource],
    comparison: Mapping[str, Any],
    plot_records: Sequence[Mapping[str, Any]],
    luminosities: Sequence[float],
    score_pull_diagnostic: Optional[Mapping[str, Any]] = None,
) -> Path:
    def format_optional(value: Any, format_spec: str) -> str:
        return "—" if value is None else format(float(value), format_spec)

    def format_p_value(distance: Mapping[str, Any]) -> str:
        probability = float(distance["p_value"])
        log10_probability = float(distance["log10_p_value"])
        if probability >= 1.0e-3:
            return f"{probability:.4g}"
        if probability > 0.0:
            return f"{probability:.3e}"
        return f"10<sup>{log10_probability:.2f}</sup>"

    source_rows = []
    for index, source in enumerate(sources):
        role = "Reference" if index == 0 else "Variation"
        parameters = ", ".join(
            f"{key}={value}" for key, value in source.scenario.parameters.items()
        ) or "—"
        source_rows.append(
            f"<tr><th>{role}</th><td><span class=\"swatch\" "
            f"style=\"background:{html.escape(str(source.scenario.color))}\"></span>"
            f"{html.escape(source.scenario.label)}</td>"
            f"<td>{html.escape(parameters)}</td>"
            f"<td><a href=\"../{html.escape(str(source.metadata['run_id']))}/index.html\">"
            f"{html.escape(str(source.metadata['run_id']))}</a></td></tr>"
        )
    cross_section_rows = "".join(
        f"<tr><th>{html.escape(str(process))}</th><td>{float(value):.10g}</td></tr>"
        for process, value in comparison["normalization"][
            "reference_cross_sections_pb"
        ].items()
    )
    summary_rows = []
    for strategy, strategy_values in comparison["analyses"].items():
        for channel, channel_values in strategy_values.items():
            reference_id = comparison["reference_scenario_id"]
            reference_fbeam = channel_values["scenarios"][reference_id]["f_beam"]
            for scenario_id, difference in channel_values[
                "differences_from_reference"
            ].items():
                varied_fbeam = channel_values["scenarios"][scenario_id]["f_beam"]
                for luminosity in luminosities:
                    lumi_key = str(float(luminosity))
                    nominal_truth = difference["luminosities"][lumi_key][
                        "nominal_truth"
                    ]
                    variation_truth = difference["luminosities"][lumi_key][
                        "variation_truth"
                    ]
                    nominal_z = nominal_truth[
                        "directional_f_beam_significance_data_plus_mc_stat"
                    ]
                    variation_z = variation_truth[
                        "directional_f_beam_significance_data_plus_mc_stat"
                    ]
                    d2 = nominal_truth["six_bin_data_plus_mc_stat"]
                    summary_rows.append(
                        f"<tr><td>{html.escape(strategy)}</td><td>{html.escape(channel)}</td>"
                        f"<td>{html.escape(difference['label'])}</td><td>{luminosity:g}</td>"
                        f"<td>{reference_fbeam:.6f}</td><td>{varied_fbeam:.6f}</td>"
                        f"<td>{difference['delta_f_beam']:+.6f}</td>"
                        f"<td>{format_optional(nominal_z, '+.3f')}</td>"
                        f"<td>{format_optional(variation_z, '+.3f')}</td>"
                        f"<td>{d2['D2']:.3f} (rank {d2['covariance_rank']})</td>"
                        f"<td>{format_p_value(d2)}</td></tr>"
                    )
    registry = {
        (spec.channel, spec.key): spec for spec in plot_registry()
    }
    hypothesis_rows: Dict[str, List[str]] = {
        "shape_only": [],
        "rate_and_shape": [],
    }
    for strategy, strategy_values in comparison["analyses"].items():
        for channel, channel_values in strategy_values.items():
            for observable, observable_values in channel_values["observables"].items():
                observable_title = registry[(channel, observable)].title
                for scenario_id, scenario_values in observable_values[
                    "comparisons_to_reference"
                ].items():
                    for luminosity_key, tests in scenario_values["luminosities"].items():
                        for test_scope in ("shape_only", "rate_and_shape"):
                            values = tests[test_scope]
                            nominal_data = values["nominal_truth"]["data_stat_only"]
                            nominal_combined = values["nominal_truth"][
                                "data_plus_mc_stat"
                            ]
                            variation_data = values["variation_truth"]["data_stat_only"]
                            variation_combined = values["variation_truth"][
                                "data_plus_mc_stat"
                            ]
                            mc_only = values["mc_stat_only"]
                            hypothesis_rows[test_scope].append(
                                f"<tr><td>{html.escape(strategy)}</td>"
                                f"<td>{html.escape(channel)}</td>"
                                f"<td>{html.escape(observable_title)}</td>"
                                f"<td>{html.escape(str(scenario_values['label']))}</td>"
                                f"<td>{float(luminosity_key):g}</td>"
                                f"<td>{nominal_combined['D2']:.3f} / "
                                f"{nominal_combined['covariance_rank']}</td>"
                                f"<td>{format_p_value(nominal_data)}</td>"
                                f"<td>{format_p_value(nominal_combined)}</td>"
                                f"<td>{format_p_value(variation_data)}</td>"
                                f"<td>{format_p_value(variation_combined)}</td>"
                                f"<td>{format_p_value(mc_only)}</td></tr>"
                            )

    ranking = comparison["observable_hypothesis_test_rankings"]["shape_only"]
    best = ranking["most_significant_local_test"]
    best_label = registry[(best["channel"], best["observable"])].title
    best_distance = {
        "p_value": best["p_value"],
        "log10_p_value": best["log10_p_value"],
    }
    bonferroni = float(ranking["bonferroni_upper_bound_for_selected_minimum_p"])
    bonferroni_log10 = float(ranking["log10_bonferroni_upper_bound"])
    bonferroni_text = (
        f"{bonferroni:.3e}"
        if bonferroni > 0.0
        else f"10<sup>{bonferroni_log10:.2f}</sup>"
    )
    hypothesis_header = (
        "<thead><tr><th>Analysis</th><th>Channel</th><th>Observable</th>"
        "<th>Variation</th><th>fb⁻¹</th><th>D² / rank (ref., data+MC)</th>"
        "<th>p data (ref.)</th><th>p data+MC (ref.)</th>"
        "<th>p data (var.)</th><th>p data+MC (var.)</th>"
        "<th>p MC only</th></tr></thead>"
    )
    hypothesis_html = (
        '<section class="panel"><h2>Pairwise Herwig–CR histogram p-values</h2>'
        '<p>These are local expected (Asimov) upper-tail χ² p-values using the full '
        'event-level bin covariance. The primary comparison is shape only, with each '
        'scenario normalized independently; rate+shape tests correspond to the absolute '
        'event-count panels. “Data+MC” adds the two statistically independent scenario-MC '
        'covariances. No modelling or experimental systematic uncertainty is included.</p>'
        f'<p><strong>Most significant local shape test:</strong> {html.escape(str(best["strategy"]))} '
        f'{html.escape(str(best["channel"]))}, {html.escape(best_label)}, '
        f'{float(best["luminosity_fb"]):g} fb⁻¹: D²={float(best["D2"]):.3f} with '
        f'rank {int(best["covariance_rank"])}, p={format_p_value(best_distance)}. '
        f'There are {int(ranking["test_count"])} correlated tests; the conservative '
        f'Bonferroni bound is p≤{bonferroni_text}.</p>'
        '<h3>Shape-only tests (primary)</h3><div class="table-wrap"><table>'
        + hypothesis_header
        + '<tbody>' + ''.join(hypothesis_rows["shape_only"]) + '</tbody></table></div>'
        '<details><summary>Rate-and-shape tests matching the absolute-yield plots</summary>'
        '<div class="table-wrap"><table>' + hypothesis_header + '<tbody>'
        + ''.join(hypothesis_rows["rate_and_shape"])
        + '</tbody></table></div></details></section>'
    )
    score_pull_html = ""
    if score_pull_diagnostic is not None:
        recommendation_rows = []
        for channel, channel_values in score_pull_diagnostic["channels"].items():
            for luminosity_key, luminosity_values in channel_values["luminosities"].items():
                for category_count, scan in luminosity_values["category_scans"].items():
                    recommended = scan["recommended"]
                    boundaries = ", ".join(
                        f"{value:.5g}" for value in recommended["boundary_scores"]
                    )
                    recommendation_rows.append(
                        f"<tr><td>{html.escape(channel)}</td>"
                        f"<td>{float(luminosity_key):g}</td>"
                        f"<td>{html.escape(category_count)}</td>"
                        f"<td>{html.escape(boundaries)}</td>"
                        f"<td>{recommended['minimum_D2_data_plus_mc_stat']:.4g}</td>"
                        f"<td>{recommended['minimum_D2_data_stat_only']:.4g}</td></tr>"
                    )
        score_pull_html = (
            '<section class="panel"><h2>Exploratory XGBoost score × pull diagnostic</h2>'
            '<p>The ten score quantiles are frozen by the nominal run and use all events '
            'after common selection. Pull shapes are normalized independently inside each '
            'candidate score category, so the scan is not driven by category-rate changes. '
            'The listed boundaries maximize the worst-case conditional-shape D² across all '
            'supplied CR variations. They require confirmation with independent simulations.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Channel</th><th>fb⁻¹</th>'
            '<th>Categories</th><th>Score boundaries</th><th>Worst D² data+MC</th>'
            '<th>Worst D² data only</th></tr></thead><tbody>'
            + ''.join(recommendation_rows)
            + '</tbody></table></div><p><a href="summaries/score_pull_diagnostic.json">JSON</a> · '
            '<a href="summaries/score_pull_diagnostic.csv">CSV</a> · '
            '<a href="summaries/score_pull_diagnostic.npz">NPZ moments</a></p></section>'
        )
    cards = []
    for record in plot_records:
        lumi = _format_luminosity(float(record["luminosity_fb"]))
        cards.append(
            "<article class=\"plot-card\" "
            f"data-strategy=\"{html.escape(str(record['strategy']))}\" "
            f"data-channel=\"{html.escape(str(record['channel']))}\" "
            f"data-kind=\"{html.escape(str(record['kind']))}\" "
            f"data-lumi=\"{html.escape(lumi)}\" "
            f"data-search=\"{html.escape((str(record['title']) + ' ' + str(record['observable'])).lower())}\">"
            f"<a class=\"thumb-link\" href=\"{html.escape(str(record['png']))}\">"
            f"<img loading=\"lazy\" src=\"{html.escape(str(record['png']))}\" "
            f"alt=\"{html.escape(str(record['title']))}\"></a>"
            "<div class=\"plot-copy\"><div class=\"badges\">"
            f"<span>{html.escape(str(record['strategy']))}</span>"
            f"<span>{html.escape(str(record['channel']))}</span>"
            f"<span>{html.escape(str(record['kind']))}</span><span>{html.escape(lumi)}</span>"
            f"</div><h3>{html.escape(str(record['title']))}</h3>"
            f"<p><a href=\"{html.escape(str(record['png']))}\">PNG</a> "
            f"<a href=\"{html.escape(str(record['pdf']))}\">PDF</a></p></div></article>"
        )
    analyses = list(comparison["analyses"])
    kinds = sorted({str(record["kind"]) for record in plot_records})
    document = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PullPheno CR comparison · {html.escape(str(run_metadata['run_id']))}</title>
<style>
:root{{--ink:#172033;--muted:#657086;--line:#dce2eb;--paper:#fff;--wash:#f3f6fa;--accent:#275dad}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}header{{padding:3rem max(1rem,calc((100vw - 1280px)/2));background:linear-gradient(125deg,#12213c,#244b78 62%,#11776d);color:white}}header p{{color:#dce9f8}}h1{{font-size:clamp(2rem,5vw,3.7rem);margin:.3rem 0}}main{{max-width:1320px;margin:auto;padding:1.5rem}}section{{margin:1.2rem 0}}.panel,.plot-card{{background:white;border:1px solid var(--line);border-radius:14px;box-shadow:0 5px 18px #2435510c}}.panel{{padding:1rem}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.87rem}}th,td{{padding:.5rem .65rem;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}.swatch{{display:inline-block;width:.75rem;height:.75rem;border-radius:50%;margin-right:.4rem}}.controls{{display:grid;grid-template-columns:repeat(5,minmax(130px,1fr));gap:.7rem;position:sticky;top:0;z-index:2;background:#f3f6faed;backdrop-filter:blur(8px);padding:.8rem 0}}label{{font-size:.78rem;color:var(--muted);font-weight:700}}select,input{{display:block;width:100%;margin-top:.2rem;padding:.5rem;border:1px solid #bfc8d5;border-radius:8px;background:white}}.gallery{{display:grid;grid-template-columns:repeat(auto-fit,minmax(315px,1fr));gap:1rem}}.plot-card{{overflow:hidden}}.plot-card[hidden]{{display:none}}.thumb-link{{display:block;aspect-ratio:1.25;background:white;overflow:hidden}}.thumb-link img{{width:100%;height:100%;object-fit:contain}}.plot-copy{{padding:.85rem}}.plot-copy h3{{font-size:1rem;margin:.45rem 0}}.badges{{display:flex;gap:.35rem;flex-wrap:wrap}}.badges span{{font-size:.68rem;text-transform:uppercase;font-weight:750;border-radius:999px;padding:.18rem .45rem;background:#e7eef8;color:#274b79}}code{{overflow-wrap:anywhere}}footer{{padding:2.5rem;text-align:center;color:var(--muted)}}@media(max-width:760px){{.controls{{grid-template-columns:1fr 1fr;position:static}}}}
</style></head><body><header><div>PullPheno particle-level analysis</div>
<h1>Independent CR scenarios</h1><p><strong>Comparison run:</strong> {html.escape(str(run_metadata['run_id']))}<br>
In total-scenario plots, each point is the complete prediction from one independently generated scenario. Absolute signed-pull-angle stack overlays instead retain the first run's process stack and draw every CR variation only as total-yield error bars; companion plots add variation/reference ratios with independent-sample MC errors. All process samples, including backgrounds, are scenario-specific; no background sample is shared. The first run fixes the process cross sections and is the ratio reference.</p></header><main>
<section class="panel"><h2>Source runs</h2><div class="table-wrap"><table><thead><tr><th>Role</th><th>Scenario</th><th>Parameters</th><th>Immutable source</th></tr></thead><tbody>{''.join(source_rows)}</tbody></table></div><h3>Common reference cross sections</h3><p>Every scenario uses the first run's final-state cross sections; scenario-specific efficiencies and generated sums of weights remain independent.</p><div class="table-wrap"><table><thead><tr><th>Process</th><th>Cross section [pb]</th></tr></thead><tbody>{cross_section_rows}</tbody></table></div></section>
<section class="panel"><h2>Absolute signed-pull-angle numerical comparison</h2><p>Directional f<sub>beam</sub> values retain their signs. Six-bin D² values are Mahalanobis distances in the supported covariance subspace; √D² is not labelled as a one-dimensional Gaussian significance.</p><div class="table-wrap"><table><thead><tr><th>Analysis</th><th>Channel</th><th>Variation</th><th>fb⁻¹</th><th>f<sub>beam</sub> ref.</th><th>f<sub>beam</sub> var.</th><th>Δf<sub>beam</sub></th><th>Z (ref. truth)</th><th>Z (var. truth)</th><th>D² (ref. truth)</th><th>p (ref., data+MC)</th></tr></thead><tbody>{''.join(summary_rows)}</tbody></table></div><p><a href="summaries/comparison.json">JSON</a> · <a href="summaries/comparison.csv">CSV</a> · <a href="summaries/comparison.npz">NPZ arrays, tests and covariances</a></p></section>
{hypothesis_html}
{score_pull_html}
<section><h2>Pull-observable comparisons, reference stacks and diagnostics</h2><div class="controls">
<label>Analysis<select id="strategy"><option value="all">All</option>{''.join(f'<option value="{name}">{name}</option>' for name in analyses)}</select></label>
<label>Channel<select id="channel"><option value="all">All</option><option value="higgs">Higgs</option><option value="z">Z</option></select></label>
<label>Plot type<select id="kind"><option value="all">All</option>{''.join(f'<option value="{html.escape(name)}">{html.escape(name)}</option>' for name in kinds)}</select></label>
<label>Luminosity<select id="lumi"><option value="all">All</option>{''.join(f'<option value="{_format_luminosity(value)}">{value:g} fb⁻¹</option>' for value in luminosities)}</select></label>
<label>Search<input id="search" type="search" placeholder="angle, t_phi…"></label></div>
<div class="gallery">{''.join(cards)}</div><p id="empty" hidden>No plots match these filters.</p></section></main>
<footer>Portable immutable comparison bundle · source runs remain untouched.</footer>
<script>const ids=["strategy","channel","kind","lumi"],controls=ids.map(id=>document.getElementById(id)),search=document.getElementById("search"),cards=[...document.querySelectorAll(".plot-card")],empty=document.getElementById("empty");function filter(){{let n=0;for(const card of cards){{const ok=controls.every(el=>el.value==="all"||card.dataset[el.id]===el.value)&&(!search.value.trim()||card.dataset.search.includes(search.value.trim().toLowerCase()));card.hidden=!ok;if(ok)n++;}}empty.hidden=n!==0;}}controls.forEach(el=>el.addEventListener("change",filter));search.addEventListener("input",filter);</script></body></html>"""
    destination = run_dir / "index.html"
    write_text_exclusive(destination, document)
    return destination


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attribute_name = "src" if tag in ("img", "script") else "href"
        for key, value in attrs:
            if key == attribute_name and value:
                self.links.append(value)


def validate_html_links(index_path: Path) -> None:
    parser = _LinkCollector()
    parser.feed(index_path.read_text(encoding="utf-8"))
    missing = []
    for link in parser.links:
        if link.startswith(("#", "http://", "https://", "mailto:", "javascript:")):
            continue
        target = (index_path.parent / link.split("#", 1)[0]).resolve()
        if not target.exists():
            missing.append(link)
    if missing:
        raise RuntimeError(f"HTML index contains missing relative links: {missing}")


def _top_level_document(runs: Sequence[Mapping[str, Any]]) -> str:
    cards = []
    for run in runs:
        badge = "partial" if run.get("partial") else "full"
        if run.get("run_type") == "comparison":
            badge_label = f"comparison · {badge}"
        else:
            badge_label = f"derived {badge}" if run.get("derived_from_run") else badge
        name = html.escape(str(run.get("run_name") or "unnamed"))
        if run.get("run_type") == "comparison":
            detail = (
                f"{len(run.get('source_runs', []))} scenarios · "
                f"{html.escape(', '.join(run.get('analyses', ['cutbased'])))}"
            )
            action = "Open comparison →"
        else:
            scenario_suffix = (
                f" · {html.escape(str(run['scenario']['label']))}"
                if run.get("scenario")
                else ""
            )
            detail = (
                f"{len(run.get('samples', []))} samples · "
                f"{html.escape(', '.join(run.get('analyses', ['cutbased'])))}"
                f"{scenario_suffix}"
            )
            action = "Open plots and cutflows →"
        cards.append(
            "<article>"
            f"<div><span class=\"badge {badge}\">{badge_label}</span><span class=\"name\">{name}</span></div>"
            f"<h2><a href=\"runs/{html.escape(str(run['run_id']))}/index.html\">{html.escape(str(run['run_id']))}</a></h2>"
            f"<p>Completed {html.escape(str(run.get('completed_utc', 'unknown')))} · {detail}</p>"
            f"<p><a href=\"runs/{html.escape(str(run['run_id']))}/index.html\">{action}</a></p>"
            "</article>"
        )
    if not cards:
        cards.append("<article><h2>No completed runs yet</h2><p>Incomplete runs are intentionally omitted.</p></article>")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PullPheno result runs</title><style>
:root{{--ink:#172033;--muted:#657086;--line:#dce2eb;--paper:#fff;--wash:#f3f6fa;--accent:#275dad}}*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font:15px/1.5 Inter,system-ui,sans-serif}}header{{padding:3rem max(1rem,calc((100vw - 1100px)/2));background:#142844;color:white}}main{{max-width:1130px;margin:auto;padding:1.5rem;display:grid;gap:1rem}}article{{background:white;border:1px solid var(--line);border-radius:14px;padding:1.2rem;box-shadow:0 5px 18px #2435510c}}h1{{font-size:clamp(2rem,5vw,3.8rem);margin:.2rem 0}}h2{{font-size:1.05rem;overflow-wrap:anywhere}}a{{color:var(--accent);text-decoration:none}}a:hover,a:focus-visible{{text-decoration:underline}}.badge{{font-size:.7rem;text-transform:uppercase;font-weight:800;border-radius:999px;padding:.2rem .5rem;background:#087568;color:white}}.badge.partial{{background:#9b5d08}}.name{{margin-left:.5rem;color:var(--muted);font-weight:700}}p{{color:var(--muted)}}
</style></head><body><header><div>PullPheno</div><h1>Versioned analysis runs</h1><p>Completed runs are immutable and listed newest first.</p></header><main>{''.join(cards)}</main></body></html>"""


def update_top_level_catalog(output_root: Path) -> Tuple[Path, Path]:
    runs_root = output_root / "runs"
    runs: List[Dict[str, Any]] = []
    if runs_root.exists():
        for child in runs_root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            metadata_path = child / "run.json"
            index_path = child / "index.html"
            if not metadata_path.is_file() or not index_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if metadata.get("status") != "complete":
                continue
            runs.append(metadata)
    runs.sort(key=lambda item: str(item.get("completed_utc", "")), reverse=True)
    runs_json = output_root / "runs.json"
    top_index = output_root / "index.html"
    atomic_write_text(runs_json, json.dumps(runs, indent=2, sort_keys=True, allow_nan=False) + "\n")
    atomic_write_text(top_index, _top_level_document(runs))
    validate_html_links(top_index)
    return top_index, runs_json


def reduction_roundoff_tolerance(
    reference: float,
    term_count: int,
    additions_per_term: int = 1,
    relative_floor: float = 1.0e-12,
    absolute_floor: float = 1.0e-10,
) -> float:
    """Bound harmless differences between positive floating-point reductions.

    ROOT shards, the full XGBoost application loop and joint score-pull bins
    reduce the same positive event weights in different orders. A fixed
    relative tolerance eventually rejects valid closures as the event count
    grows. The standard ``gamma_n`` summation bound scales with the number of
    additions, while the explicit floors retain the historical tolerance for
    small samples.
    """
    value = float(reference)
    terms = int(term_count)
    multiplicity = int(additions_per_term)
    if not math.isfinite(value):
        raise ValueError("Roundoff reference must be finite")
    if terms < 0 or multiplicity <= 0:
        raise ValueError("Roundoff term counts must be non-negative and positive")
    operation_count = max(1, terms * multiplicity)
    accumulated_epsilon = operation_count * np.finfo(np.float64).eps
    if accumulated_epsilon >= 1.0:
        raise ValueError("Too many floating-point additions for a finite gamma_n bound")
    gamma_n = accumulated_epsilon / (1.0 - accumulated_epsilon)
    scale = abs(value)
    return float(
        max(
            float(absolute_floor),
            float(relative_floor) * scale,
            4.0 * gamma_n * scale,
        )
    )


def reductions_close(
    first: float,
    second: float,
    term_count: int,
    additions_per_term: int = 1,
    relative_floor: float = 1.0e-12,
    absolute_floor: float = 1.0e-10,
) -> bool:
    """Compare two positive reductions with an event-count-aware tolerance."""
    left = float(first)
    right = float(second)
    if not math.isfinite(left) or not math.isfinite(right):
        return False
    tolerance = reduction_roundoff_tolerance(
        max(abs(left), abs(right)),
        term_count,
        additions_per_term,
        relative_floor,
        absolute_floor,
    )
    return bool(abs(left - right) <= tolerance)


def validate_results(results: Sequence[SampleResult], partial: bool) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    failures = []
    for result in results:
        steps = cutflow_steps(result.spec.channel, result.strategy)
        cut_values = [result.cutflow[step].sumw for step in steps]
        positive_weights = all(stat.sumw2 >= 0.0 for stat in result.cutflow.values())
        if result.strategy == "xgboost":
            monotonic = (
                all(first + 1.0e-9 >= second for first, second in zip(cut_values[:-2], cut_values[1:-2]))
                and cut_values[-2] + 1.0e-9 >= cut_values[-1]
            )
        else:
            monotonic = all(
                first + 1.0e-9 >= second for first, second in zip(cut_values, cut_values[1:])
            )
        finite = all(
            np.all(np.isfinite(hist.sumw)) and np.all(np.isfinite(hist.sumw2))
            for hist in result.histograms.values()
        )
        selected = result.cutflow[steps[-1]].sumw
        pull_integral = result.histograms["signed_pull_angle"].integral
        folded_pull_integral = result.histograms["folded_pull_angle"].integral
        integral_match = math.isclose(selected, pull_integral, rel_tol=1.0e-10, abs_tol=1.0e-10)
        folded_integral_match = math.isclose(
            selected, folded_pull_integral, rel_tol=1.0e-10, abs_tol=1.0e-10
        )
        pull_moments_valid = True
        exact_pull_moments = result.pull_moment_model == PULL_MOMENT_MODEL
        selected_sumw2 = result.cutflow[steps[-1]].sumw2
        for observable in PULL_OBSERVABLE_KEYS:
            moments = result.pull_observable_moments.get(observable)
            histogram = result.histograms[observable]
            if moments is None:
                pull_moments_valid = False
                continue
            tolerance = max(1.0e-10, 1.0e-10 * abs(selected))
            sumw2_tolerance = max(1.0e-10, 1.0e-10 * abs(selected_sumw2))
            pull_moments_valid = pull_moments_valid and (
                np.array_equal(moments.edges, histogram.edges)
                and np.all(np.isfinite(moments.bin_sumw))
                and np.all(np.isfinite(moments.event_second_sumw))
                and np.all(np.isfinite(moments.mc_second_sumw2))
                and np.allclose(
                    moments.event_second_sumw,
                    moments.event_second_sumw.T,
                    rtol=0.0,
                    atol=tolerance,
                )
                and np.allclose(
                    moments.mc_second_sumw2,
                    moments.mc_second_sumw2.T,
                    rtol=0.0,
                    atol=sumw2_tolerance,
                )
                and np.allclose(
                    moments.bin_sumw, histogram.sumw, rtol=1.0e-10, atol=tolerance
                )
                and math.isclose(
                    float(np.sum(moments.bin_sumw)),
                    selected,
                    rel_tol=1.0e-10,
                    abs_tol=tolerance,
                )
                and math.isclose(
                    float(np.sum(moments.event_second_sumw)),
                    selected,
                    rel_tol=1.0e-10,
                    abs_tol=tolerance,
                )
                and (
                    not exact_pull_moments
                    or math.isclose(
                        float(np.sum(moments.mc_second_sumw2)),
                        selected_sumw2,
                        rel_tol=1.0e-10,
                        abs_tol=sumw2_tolerance,
                    )
                )
            )
        bin_integral_match = math.isclose(
            selected,
            float(np.sum(result.pull_bin_sumw, dtype=np.float64)),
            rel_tol=1.0e-10,
            abs_tol=1.0e-10,
        )
        differential = differential_pull_statistics(
            result.pull_bin_sumw,
            result.pull_event_second_sumw,
            result.pull_mc_second_sumw2,
            (300.0,),
        )
        ri_closure = differential["R"] is None or (
            math.isclose(differential["sum_R"], 1.0, rel_tol=0.0, abs_tol=1.0e-12)
            and math.isclose(
                differential["f_beam"],
                float(np.sum(np.asarray(differential["R"])[:3])),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        )
        covariance_valid = True
        if differential["R"] is not None:
            covariances = [np.asarray(differential["mc_statistical_covariance"])]
            covariances.extend(
                np.asarray(value)
                for value in differential["expected_statistical_covariance"].values()
            )
            covariance_valid = all(
                np.all(np.isfinite(covariance))
                and np.allclose(covariance, covariance.T, rtol=0.0, atol=1.0e-12)
                and np.allclose(np.sum(covariance, axis=1), 0.0, rtol=0.0, atol=1.0e-10)
                and float(np.min(np.linalg.eigvalsh(covariance))) >= -1.0e-10
                for covariance in covariances
            )
        generated_match = partial or math.isclose(result.processed_sumw, result.generated_sumw, rel_tol=1.0e-10, abs_tol=1.0e-8)
        full_xgboost_application = True
        score_pull_moments_valid = True
        if result.strategy == "xgboost" and result.application_scope in {
            "five_fold_out_of_fold_all_events",
            "five_fold_routed_independent_events",
            "legacy_single_model_independent_events",
            "all_independent_events",
        }:
            common = result.cutflow["opposite_hemispheres"]
            application = result.cutflow["xgboost_application_sample"]
            full_xgboost_application = (
                common.raw_count == application.raw_count
                and reductions_close(common.sumw, application.sumw, common.raw_count)
                and reductions_close(common.sumw2, application.sumw2, common.raw_count)
            )
            if result.score_pull_moments is not None:
                joint = result.score_pull_moments
                joint_bin_tolerance = reduction_roundoff_tolerance(
                    application.sumw,
                    joint.event_count,
                    additions_per_term=2,
                    relative_floor=1.0e-10,
                )
                joint_event_tolerance = reduction_roundoff_tolerance(
                    application.sumw,
                    joint.event_count,
                    additions_per_term=4,
                    relative_floor=1.0e-10,
                )
                joint_sumw2_tolerance = reduction_roundoff_tolerance(
                    application.sumw2,
                    joint.event_count,
                    additions_per_term=4,
                    relative_floor=1.0e-10,
                )
                score_pull_moments_valid = (
                    result.score_pull_moment_model == SCORE_PULL_MOMENT_MODEL
                    and joint.event_count == application.raw_count
                    and np.array_equal(joint.pull_edges, PULL_BIN_EDGES)
                    and len(joint.score_edges) == SCORE_PULL_BIN_COUNT + 1
                    and np.all(np.isfinite(joint.bin_sumw))
                    and np.all(np.isfinite(joint.event_second_sumw))
                    and np.all(np.isfinite(joint.mc_second_sumw2))
                    and np.allclose(
                        joint.event_second_sumw,
                        joint.event_second_sumw.T,
                        rtol=0.0,
                        atol=joint_event_tolerance,
                    )
                    and np.allclose(
                        joint.mc_second_sumw2,
                        joint.mc_second_sumw2.T,
                        rtol=0.0,
                        atol=joint_sumw2_tolerance,
                    )
                    and math.isclose(
                        float(np.sum(joint.bin_sumw, dtype=np.float64)),
                        application.sumw,
                        rel_tol=0.0,
                        abs_tol=joint_bin_tolerance,
                    )
                    and math.isclose(
                        float(np.sum(joint.event_second_sumw, dtype=np.float64)),
                        application.sumw,
                        rel_tol=0.0,
                        abs_tol=joint_event_tolerance,
                    )
                    and math.isclose(
                        float(np.sum(joint.mc_second_sumw2, dtype=np.float64)),
                        application.sumw2,
                        rel_tol=0.0,
                        abs_tol=joint_sumw2_tolerance,
                    )
                )
        item = {
            "sample": result.spec.name,
            "strategy": result.strategy,
            "no_invalid_events": result.invalid_events == 0,
            "monotonic_cutflow": monotonic,
            "finite_histograms": finite,
            "pull_integral_matches_selected": integral_match,
            "folded_pull_integral_matches_selected": folded_integral_match,
            "all_pull_observable_moments_valid": pull_moments_valid,
            "ri_integral_matches_selected": bin_integral_match,
            "ri_and_fbeam_closure": ri_closure,
            "covariances_valid": covariance_valid,
            "processed_sumw_matches_generated": generated_match,
            "nonnegative_sumw2": positive_weights,
            "xgboost_application_uses_all_common_events": full_xgboost_application,
            "score_pull_moments_valid_if_present": score_pull_moments_valid,
        }
        checks.append(item)
        if not all(value for key, value in item.items() if key not in ("sample", "strategy")):
            failures.append(item)
    if failures:
        raise RuntimeError(f"Analysis validation failed: {failures}")
    return {"status": "passed", "checks": checks}


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path, help="JSON sample manifest")
    source.add_argument(
        "--from-run",
        type=Path,
        help="Completed run to replot from stored histograms without rereading ROOT events",
    )
    source.add_argument(
        "--resume-incomplete",
        type=Path,
        help=(
            "Failed run containing a complete ROOT-pass checkpoint; resumes XGBoost "
            "and artifact production without reopening ROOT events"
        ),
    )
    source.add_argument(
        "--compare-runs",
        type=Path,
        nargs="+",
        metavar="RUN",
        help=(
            "Completed scenario runs to compare without ROOT input; the first run is the "
            "cross-section and ratio reference"
        ),
    )
    parser.add_argument("--output-root", type=Path, default=Path("results"), help="Root directory containing immutable runs")
    parser.add_argument("--run-name", help="Optional human-readable label included in the unique run ID")
    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=ANALYSIS_STRATEGIES,
        default=("cutbased",),
        help="Analysis branches to produce in one immutable run (default: cutbased)",
    )
    parser.add_argument(
        "--xgb-model-run",
        type=Path,
        help="Completed nominal run supplying frozen XGBoost models and thresholds",
    )
    parser.add_argument("--max-events", type=int, help="Maximum events per sample (marks the run as partial)")
    parser.add_argument("--luminosities", type=float, nargs="+", help="Override manifest luminosities in fb^-1")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum concurrent ROOT event-range workers (default: 1)",
    )
    parser.add_argument(
        "--event-shards",
        type=int,
        help=(
            "Contiguous event ranges per sample; combine with --workers for within-sample "
            "parallelism (default for new ROOT runs: 1)"
        ),
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    args = parser.parse_args(argv)
    if args.max_events is not None and args.max_events <= 0:
        parser.error("--max-events must be positive")
    if args.from_run is not None and args.max_events is not None:
        parser.error("--max-events cannot be combined with --from-run")
    if args.resume_incomplete is not None and args.max_events is not None:
        parser.error("--max-events cannot be combined with --resume-incomplete")
    if args.from_run is not None and args.event_shards is not None:
        parser.error("--event-shards cannot be combined with --from-run")
    if args.resume_incomplete is not None and args.event_shards is not None:
        parser.error("--event-shards is restored from the checkpoint when resuming")
    if args.compare_runs is not None and args.event_shards is not None:
        parser.error("--event-shards cannot be combined with --compare-runs")
    if args.compare_runs is not None and len(args.compare_runs) < 2:
        parser.error("--compare-runs requires a reference and at least one variation")
    if args.compare_runs is not None and args.max_events is not None:
        parser.error("--max-events cannot be combined with --compare-runs")
    if len(set(args.analyses)) != len(args.analyses):
        parser.error("--analyses contains a duplicate strategy")
    if args.xgb_model_run is not None and "xgboost" not in args.analyses:
        parser.error("--xgb-model-run requires --analyses to include xgboost")
    if args.from_run is not None and args.xgb_model_run is not None:
        parser.error("--xgb-model-run cannot be combined with --from-run")
    if args.resume_incomplete is not None and args.xgb_model_run is not None:
        parser.error("--xgb-model-run is restored from the checkpoint when resuming")
    if args.compare_runs is not None and args.xgb_model_run is not None:
        parser.error("--xgb-model-run cannot be combined with --compare-runs")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.event_shards is not None and args.event_shards <= 0:
        parser.error("--event-shards must be positive")
    return args


def run_comparison(args: argparse.Namespace, started: datetime) -> int:
    analyses = tuple(
        strategy for strategy in ANALYSIS_STRATEGIES if strategy in args.analyses
    )
    output_root = args.output_root.expanduser().resolve()
    sources = tuple(
        resolve_comparison_source(path, output_root, args.luminosities, index)
        for index, path in enumerate(args.compare_runs)
    )
    reference_cross_sections = validate_comparison_sources(sources, analyses)
    luminosities = tuple(
        float(value)
        for value in (args.luminosities or sources[0].config.luminosities_fb)
    )
    payload = {
        "analysis_version": ANALYSIS_VERSION,
        "run_type": "comparison",
        "analyses": list(analyses),
        "luminosities_fb": list(luminosities),
        "source_runs": [
            {
                "run_id": source.metadata["run_id"],
                "configuration_hash": source.metadata.get("configuration_hash"),
                "scenario": asdict(source.scenario),
            }
            for source in sources
        ],
        "reference_cross_sections_pb": reference_cross_sections,
        "pull_moment_model": PULL_MOMENT_MODEL,
    }
    reservation = reserve_run_directory(
        output_root, make_run_id(payload, args.run_name, started)
    )
    run_log_handler = attach_run_file_logger(reservation.incomplete_dir, args.log_level)
    phase = "comparison-statistics"
    logging.info("Reserved immutable comparison run %s", reservation.run_id)
    try:
        comparison, numerical = build_comparison_statistics(
            sources,
            analyses,
            luminosities,
            reference_cross_sections,
        )
        score_pull_payload: Optional[Dict[str, Any]] = None
        score_pull_numerical: Dict[Tuple[str, float, str], Dict[str, np.ndarray]] = {}
        if "xgboost" in analyses:
            score_pull_payload, score_pull_numerical = build_score_pull_diagnostic(
                sources,
                luminosities,
                reference_cross_sections,
            )
            comparison["score_pull_diagnostic"] = {
                "available": True,
                "json": "summaries/score_pull_diagnostic.json",
                "csv": "summaries/score_pull_diagnostic.csv",
                "npz": "summaries/score_pull_diagnostic.npz",
                "moment_model": SCORE_PULL_MOMENT_MODEL,
            }
        write_comparison_artifacts(reservation.incomplete_dir, comparison, numerical)
        if score_pull_payload is not None:
            write_score_pull_diagnostic_artifacts(
                reservation.incomplete_dir,
                score_pull_payload,
                score_pull_numerical,
            )
        phase = "comparison-plots"
        plot_records = generate_comparison_plots(
            reservation.incomplete_dir,
            sources,
            analyses,
            luminosities,
            numerical,
        )
        if score_pull_payload is not None:
            plot_records.extend(
                generate_score_pull_diagnostic_plots(
                    reservation.incomplete_dir,
                    sources,
                    luminosities,
                    score_pull_payload,
                    score_pull_numerical,
                )
            )
        write_json_exclusive(
            reservation.incomplete_dir / "summaries" / "plots.json", plot_records
        )
        completed = datetime.now(timezone.utc)
        partial = bool(sources[0].metadata.get("partial"))
        run_metadata = {
            "status": "complete",
            "run_type": "comparison",
            "run_id": reservation.run_id,
            "run_name": sanitize_run_name(args.run_name),
            "analysis_version": ANALYSIS_VERSION,
            "started_utc": started.isoformat(),
            "completed_utc": completed.isoformat(),
            "duration_seconds": (completed - started).total_seconds(),
            "partial": partial,
            "max_events_per_sample": sources[0].metadata.get("max_events_per_sample"),
            "workers": 0,
            "analyses": list(analyses),
            "luminosities_fb": list(luminosities),
            "configuration_hash": config_digest(payload),
            "configuration": payload,
            "git": git_provenance(Path(__file__).resolve().parent),
            "samples": [],
            "scenario": None,
            "source_runs": [
                {
                    "run_id": source.metadata["run_id"],
                    "run_name": source.metadata.get("run_name"),
                    "scenario": asdict(source.scenario),
                    "configuration_hash": source.metadata.get("configuration_hash"),
                }
                for source in sources
            ],
            "reference_run_id": sources[0].metadata["run_id"],
            "artifacts": {
                "run_log": RUN_LOG_NAME,
                "plots": len(plot_records),
                "comparison_json": "summaries/comparison.json",
                "comparison_csv": "summaries/comparison.csv",
                "comparison_npz": "summaries/comparison.npz",
                **(
                    {
                        "score_pull_diagnostic_json": "summaries/score_pull_diagnostic.json",
                        "score_pull_diagnostic_csv": "summaries/score_pull_diagnostic.csv",
                        "score_pull_diagnostic_npz": "summaries/score_pull_diagnostic.npz",
                    }
                    if score_pull_payload is not None
                    else {}
                ),
            },
            "validation": {
                "status": "passed",
                "source_compatibility": True,
                "source_count": len(sources),
                "pull_moment_model": PULL_MOMENT_MODEL,
                "normalization_uses_reference_cross_sections": True,
                "score_pull_moment_model": (
                    SCORE_PULL_MOMENT_MODEL if score_pull_payload is not None else None
                ),
            },
        }
        generate_comparison_index(
            reservation.incomplete_dir,
            run_metadata,
            sources,
            comparison,
            plot_records,
            luminosities,
            score_pull_payload,
        )
        write_json_exclusive(reservation.incomplete_dir / "run.json", run_metadata)
        phase = "final-link-validation"
        validate_html_links(reservation.incomplete_dir / "index.html")
        reservation.incomplete_dir.rename(reservation.final_dir)
        update_top_level_catalog(reservation.output_root)
        logging.info("Completed comparison run: %s", reservation.final_dir)
        print(reservation.final_dir)
        return 0
    except Exception as error:
        write_failure_record(reservation, started, phase, error)
        logging.exception(
            "Comparison failed; partial artifacts remain in %s",
            reservation.incomplete_dir,
        )
        return 1
    finally:
        detach_run_file_logger(run_log_handler)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_arguments(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    started = datetime.now(timezone.utc)
    if args.compare_runs is not None:
        return run_comparison(args, started)

    source_metadata: Optional[Dict[str, Any]] = None
    preloaded_results: Optional[List[SampleResult]] = None
    checkpoint_results: Optional[List[SampleResult]] = None
    resume_metadata: Optional[Dict[str, Any]] = None
    xgboost_metadata: Optional[Dict[str, Any]] = None
    xgboost_diagnostics: Dict[str, Any] = {}
    effective_xgb_model_run: Optional[Path] = args.xgb_model_run
    event_sharding_metadata: Optional[Dict[str, Any]] = None

    if args.from_run is not None:
        config, preloaded_results, source_metadata = load_completed_run(
            args.from_run, args.luminosities
        )
        effective_max_events = source_metadata.get("max_events_per_sample")
        effective_event_shards = int(
            source_metadata.get("configuration", {})
            .get("event_processing", {})
            .get("event_shards_per_sample", 1)
        )
        analyses = tuple(
            strategy
            for strategy in ANALYSIS_STRATEGIES
            if any(result.strategy == strategy for result in preloaded_results)
        )
        source_xgb_summary = (
            args.from_run.expanduser().resolve() / "summaries" / "xgboost.json"
        )
        if source_xgb_summary.is_file():
            xgboost_metadata = json.loads(source_xgb_summary.read_text(encoding="utf-8"))
    elif args.resume_incomplete is not None:
        config, checkpoint_results, resume_metadata = load_root_pass_checkpoint(
            args.resume_incomplete, args.luminosities
        )
        checkpoint_configuration = resume_metadata["configuration"]
        stored_analyses = tuple(
            strategy
            for strategy in ANALYSIS_STRATEGIES
            if strategy in checkpoint_configuration.get("analyses", ("cutbased",))
        )
        requested_analyses = tuple(
            strategy for strategy in ANALYSIS_STRATEGIES if strategy in args.analyses
        )
        if requested_analyses != stored_analyses:
            raise ValueError(
                "--analyses must exactly match the ROOT-pass checkpoint: "
                f"requested {requested_analyses}, stored {stored_analyses}"
            )
        analyses = stored_analyses
        effective_max_events = checkpoint_configuration.get("max_events_per_sample")
        effective_event_shards = int(
            checkpoint_configuration.get("event_processing", {}).get(
                "event_shards_per_sample", 1
            )
        )
        xgb_configuration = checkpoint_configuration.get("xgboost") or {}
        stored_model_run = xgb_configuration.get("model_run")
        effective_xgb_model_run = (
            None if stored_model_run is None else Path(str(stored_model_run))
        )
    else:
        config = read_manifest(args.samples, args.luminosities)
        effective_max_events = args.max_events
        effective_event_shards = args.event_shards or 1
        analyses = tuple(
            strategy for strategy in ANALYSIS_STRATEGIES if strategy in args.analyses
        )
    if "xgboost" in analyses:
        for channel in ("higgs", "z"):
            roles = {sample.role for sample in config.samples if sample.channel == channel}
            if roles != {"signal", "background"}:
                raise ValueError(
                    f"XGBoost channel {channel} requires manifest signal and background roles"
                )
    payload = resolved_config_payload(
        config,
        effective_max_events,
        analyses,
        effective_xgb_model_run,
        effective_event_shards,
    )
    if source_metadata is not None:
        payload["derived_from_run_id"] = source_metadata["run_id"]
        payload["event_source"] = "stored_histograms"
    if resume_metadata is not None:
        payload["resumed_from_checkpoint"] = {
            "source_run_id": resume_metadata["source_run_id"],
            "configuration_hash": resume_metadata["configuration_hash"],
            "checkpoint_created_utc": resume_metadata["created_utc"],
            "path": str(args.resume_incomplete.expanduser().resolve()),
        }
        payload["event_source"] = "root_pass_checkpoint"
    base_run_id = make_run_id(payload, args.run_name, started)
    reservation = reserve_run_directory(args.output_root.expanduser(), base_run_id)
    run_log_handler = attach_run_file_logger(reservation.incomplete_dir, args.log_level)
    phase = "initialization"
    logging.info("Reserved immutable run %s", reservation.run_id)
    try:
        if preloaded_results is not None:
            results = preloaded_results
            worker_count = 0
            partial = bool(source_metadata and source_metadata.get("partial"))
            logging.info(
                "Replotting stored histograms from completed run %s",
                source_metadata["run_id"],
            )
            if xgboost_metadata is not None:
                for values in xgboost_metadata.get("channels", {}).values():
                    relative_models = (
                        [Path(str(model["model_path"])) for model in values["models"]]
                        if values.get("models")
                        else [Path(str(values["model_path"]))]
                    )
                    for relative_model in relative_models:
                        source_model = args.from_run.resolve() / relative_model
                        destination_model = reservation.incomplete_dir / relative_model
                        if not source_model.is_file():
                            raise FileNotFoundError(
                                f"Source model artifact is missing: {source_model}"
                            )
                        destination_model.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source_model, destination_model)
        elif checkpoint_results is not None:
            worker_count = 0
            partial = effective_max_events is not None
            cut_results = checkpoint_results
            logging.info(
                "Skipping ROOT input; resuming from checkpoint created by %s",
                resume_metadata["source_run_id"],
            )
            write_json_exclusive(
                reservation.incomplete_dir / "resumed-from-checkpoint.json",
                payload["resumed_from_checkpoint"],
            )
        else:
            partial = args.max_events is not None
            collect_common_events = "xgboost" in analyses
        if preloaded_results is None and checkpoint_results is None:
            phase = "root-event-processing"
            (
                cut_results,
                event_sharding_metadata,
                worker_count,
            ) = process_root_samples(
                config,
                args.max_events,
                effective_event_shards,
                args.workers,
                args.log_level,
                collect_common_events,
            )

        checkpoint_created = False
        if preloaded_results is None and checkpoint_results is None:
            phase = "root-pass-checkpoint"
            write_root_pass_checkpoint(
                reservation.incomplete_dir,
                payload,
                cut_results,
                reservation.run_id,
            )
            checkpoint_created = True

        if preloaded_results is None:
            results = list(cut_results) if "cutbased" in analyses else []
            if "xgboost" in analyses:
                phase = "xgboost-training-and-application"
                xgb_results, xgboost_metadata, xgboost_diagnostics = build_xgboost_results(
                    cut_results,
                    reservation.incomplete_dir,
                    model_run=effective_xgb_model_run,
                )
                results.extend(xgb_results)
                for result in cut_results:
                    result.common_events = None

        phase = "numerical-summaries-and-validation"
        summaries = [result_summary(result, config.luminosities_fb) for result in results]
        validation = validate_results(results, partial)
        summary_payload = {
            "analysis_version": ANALYSIS_VERSION,
            "partial": partial,
            "luminosities_fb": list(config.luminosities_fb),
            "samples": summaries,
            "channels": {
                strategy: {
                    channel: channel_pull_summary(
                        channel, results, config.luminosities_fb, strategy
                    )
                    for channel in ("higgs", "z")
                }
                for strategy in analyses
            },
            "validation": validation,
        }
        if source_metadata is not None:
            summary_payload["derived_from_run_id"] = source_metadata["run_id"]
        if resume_metadata is not None:
            summary_payload["resumed_from_checkpoint_run_id"] = resume_metadata[
                "source_run_id"
            ]
        write_json_exclusive(
            reservation.incomplete_dir / "summaries" / "analysis.json",
            summary_payload,
        )
        if xgboost_metadata is not None:
            write_json_exclusive(
                reservation.incomplete_dir / "summaries" / "xgboost.json",
                xgboost_metadata,
            )
        write_histogram_npz(reservation.incomplete_dir, results)
        write_ri_artifacts(reservation.incomplete_dir, results, config.luminosities_fb)
        for strategy in analyses:
            for channel in ("higgs", "z"):
                write_cutflow_files(
                    reservation.incomplete_dir,
                    channel,
                    results,
                    config.luminosities_fb,
                    strategy,
                )
        phase = "plot-generation"
        plot_records = generate_plots(
            reservation.incomplete_dir,
            results,
            config.luminosities_fb,
            partial,
        )
        plot_records.extend(
            generate_ri_plots(
                reservation.incomplete_dir, results, config.luminosities_fb, partial
            )
        )
        if xgboost_diagnostics:
            plot_records.extend(
                generate_xgboost_diagnostic_plots(
                    reservation.incomplete_dir, xgboost_diagnostics
                )
            )
        write_json_exclusive(reservation.incomplete_dir / "summaries" / "plots.json", plot_records)
        completed = datetime.now(timezone.utc)
        run_metadata = {
            "status": "complete",
            "run_type": "analysis",
            "run_id": reservation.run_id,
            "run_name": sanitize_run_name(args.run_name),
            "analysis_version": ANALYSIS_VERSION,
            "started_utc": started.isoformat(),
            "completed_utc": completed.isoformat(),
            "duration_seconds": (completed - started).total_seconds(),
            "partial": partial,
            "max_events_per_sample": effective_max_events,
            "workers": worker_count,
            "event_shards_per_sample": effective_event_shards,
            "event_sharding": event_sharding_metadata,
            "analyses": list(analyses),
            "configuration_hash": config_digest(payload),
            "configuration": payload,
            "scenario": None if config.scenario is None else asdict(config.scenario),
            "git": git_provenance(Path(__file__).resolve().parent),
            "samples": [
                {
                    "name": next(result for result in results if result.spec.name == spec.name).spec.name,
                    "channel": spec.channel,
                    "role": spec.role,
                    "cross_section_pb": spec.cross_section_pb,
                    "cross_section_unc_pb": spec.cross_section_unc_pb,
                    "cross_section_source": spec.cross_section_source,
                    "generator_cross_section_pb": spec.generator_cross_section_pb,
                    "generator_cross_section_unc_pb": spec.generator_cross_section_unc_pb,
                    "total_entries": next(result for result in results if result.spec.name == spec.name).total_entries,
                    "processed_entries": next(result for result in results if result.spec.name == spec.name).processed_entries,
                    "generated_sumw": next(result for result in results if result.spec.name == spec.name).generated_sumw,
                    "processed_sumw": next(result for result in results if result.spec.name == spec.name).processed_sumw,
                    "files": next(result for result in results if result.spec.name == spec.name).files,
                }
                for spec in config.samples
            ],
            "artifacts": {
                "run_log": RUN_LOG_NAME,
                "plots": len(plot_records),
                "summary": "summaries/analysis.json",
                "histograms": "summaries/histograms.npz",
                "ri_csv": "summaries/ri.csv",
                "ri_covariances": "summaries/ri_covariances.npz",
                "cutflows": [
                    f"cutflows/{strategy}/{channel}.csv"
                    for strategy in analyses for channel in ("higgs", "z")
                ],
            },
            "validation": validation,
        }
        if source_metadata is not None:
            run_metadata["derived_from_run"] = {
                "run_id": source_metadata["run_id"],
                "configuration_hash": source_metadata.get("configuration_hash"),
                "analysis_version": source_metadata.get("analysis_version"),
            }
            run_metadata["event_processing"] = "reused_stored_histograms"
        if resume_metadata is not None:
            run_metadata["resumed_from_checkpoint"] = payload[
                "resumed_from_checkpoint"
            ]
            run_metadata["event_processing"] = "reused_root_pass_checkpoint"
        elif preloaded_results is None:
            run_metadata["root_pass_checkpoint"] = {
                "created": checkpoint_created,
                "retained_after_success": False,
                "format_version": ROOT_PASS_CHECKPOINT_VERSION,
            }
        generate_run_index(
            reservation.incomplete_dir,
            run_metadata,
            results,
            summaries,
            plot_records,
            config.luminosities_fb,
            xgboost_metadata,
        )
        write_json_exclusive(reservation.incomplete_dir / "run.json", run_metadata)
        phase = "final-link-validation"
        validate_html_links(reservation.incomplete_dir / "index.html")
        if checkpoint_created:
            phase = "checkpoint-cleanup"
            remove_root_pass_checkpoint(reservation.incomplete_dir)
            logging.info("Removed ROOT-pass recovery checkpoint after successful validation")
        phase = "finalization"
        reservation.incomplete_dir.rename(reservation.final_dir)
        update_top_level_catalog(reservation.output_root)
        logging.info("Completed run: %s", reservation.final_dir)
        print(reservation.final_dir)
        return 0
    except Exception as error:
        write_failure_record(reservation, started, phase, error)
        logging.exception("Run failed; partial artifacts remain in %s", reservation.incomplete_dir)
        return 1
    finally:
        detach_run_file_logger(run_log_handler)


if __name__ == "__main__":
    sys.exit(main())
