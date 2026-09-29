"""Event-level moments for the beam-oriented pull asymmetry.

The observable is |x| times the positive-minus-negative density in x,
normalized to the selected cross section.  Each selected event contributes
both tagging jets with weight 1/2; the normalization includes exact zeros
and values outside the displayed range.  No pull-magnitude cut is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


ASYMMETRY_KEYS = ("pull_asymmetry_beam", "pull_asymmetry_phi")
ASYMMETRY_EDGES = np.linspace(0.0, 0.03, 61, dtype=np.float64)
ASYMMETRY_MODEL = "event_level_signed_projection_with_selected_denominator_v1"


@dataclass
class BeamAsymmetryMoments:
    """Weighted first/second moments of an augmented per-event vector.

    The vector holds signed finite bins, one signed overflow bin, and the
    selected-event denominator. ``event_second_sumw`` is sum(w f f^T), for
    projected data, while ``mc_second_sumw2`` is sum(w^2 f f^T), for finite
    Monte Carlo. In particular, the two tagging jets are not independent.
    """

    edges: np.ndarray = field(default_factory=lambda: ASYMMETRY_EDGES.copy())
    sumw: np.ndarray = field(init=False)
    event_second_sumw: np.ndarray = field(init=False)
    mc_second_sumw2: np.ndarray = field(init=False)
    positive_sumw: np.ndarray = field(init=False)
    negative_sumw: np.ndarray = field(init=False)
    zero_sumw: float = 0.0
    event_count: int = 0
    zero_entries: int = 0

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=np.float64).copy()
        if (self.edges.ndim != 1 or len(self.edges) < 2
                or not np.all(np.isfinite(self.edges)) or self.edges[0] != 0.0
                or np.any(np.diff(self.edges) <= 0.0)):
            raise ValueError("Asymmetry edges must increase strictly from zero")
        dimension = len(self.edges) + 1
        self.sumw = np.zeros(dimension, dtype=np.float64)
        self.event_second_sumw = np.zeros((dimension, dimension), dtype=np.float64)
        self.mc_second_sumw2 = np.zeros((dimension, dimension), dtype=np.float64)
        self.positive_sumw = np.zeros(dimension - 1, dtype=np.float64)
        self.negative_sumw = np.zeros(dimension - 1, dtype=np.float64)

    def fill(self, values: Sequence[float], event_weight: float) -> None:
        """Fill the two unfiltered signed projections of one selected event."""
        projections = np.asarray(values, dtype=np.float64)
        weight = float(event_weight)
        if projections.shape != (2,) or not np.all(np.isfinite(projections)):
            raise ValueError("An asymmetry event requires two finite tagging-jet projections")
        if not math.isfinite(weight):
            raise ValueError("An asymmetry event weight must be finite")
        vector = np.zeros_like(self.sumw)
        vector[-1] = 1.0
        for projection in projections:
            if projection == 0.0:
                self.zero_sumw += 0.5 * weight
                self.zero_entries += 1
                continue
            magnitude = abs(float(projection))
            # The last finite edge belongs to overflow, as do larger values.
            index = min(int(np.searchsorted(self.edges, magnitude, side="right") - 1),
                        len(self.edges) - 1)
            sign = 1.0 if projection > 0.0 else -1.0
            vector[index] += 0.5 * sign
            if sign > 0.0:
                self.positive_sumw[index] += 0.5 * weight
            else:
                self.negative_sumw[index] += 0.5 * weight
        self.sumw += weight * vector
        # Use sparse event support: at most two projection bins plus denominator.
        nonzero = np.flatnonzero(vector)
        support = np.ix_(nonzero, nonzero)
        outer = np.outer(vector[nonzero], vector[nonzero])
        self.event_second_sumw[support] += weight * outer
        self.mc_second_sumw2[support] += weight * weight * outer
        self.event_count += 1

    def add(self, other: "BeamAsymmetryMoments") -> None:
        """Merge another independent shard, requiring identical bin edges."""
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("Cannot merge asymmetry moments with different edges")
        for attribute in ("sumw", "event_second_sumw", "mc_second_sumw2",
                          "positive_sumw", "negative_sumw"):
            getattr(self, attribute)[:] += getattr(other, attribute)
        self.zero_sumw += other.zero_sumw
        self.event_count += other.event_count
        self.zero_entries += other.zero_entries

    def validate(self) -> None:
        """Validate persisted moments and normalization bookkeeping."""
        dimension = len(self.edges) + 1
        shapes = {"sumw": (dimension,),
                  "event_second_sumw": (dimension, dimension),
                  "mc_second_sumw2": (dimension, dimension),
                  "positive_sumw": (dimension - 1,),
                  "negative_sumw": (dimension - 1,)}
        for attribute, shape in shapes.items():
            values = getattr(self, attribute)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError("Invalid asymmetry moment array: " + attribute)
        if not math.isfinite(self.zero_sumw):
            raise ValueError("Nonfinite asymmetry zero weight")
        if (self.event_count < 0 or self.zero_entries < 0
                or self.zero_entries > 2 * self.event_count):
            raise ValueError("Invalid asymmetry event/zero counts")
        if not np.allclose(self.sumw[:-1], self.positive_sumw - self.negative_sumw,
                           rtol=1e-9, atol=1e-10):
            raise ValueError("Asymmetry sign bookkeeping does not close")
        accounted = float(np.sum(self.positive_sumw + self.negative_sumw)) + self.zero_sumw
        if not math.isclose(float(self.sumw[-1]), accounted, rel_tol=1e-9, abs_tol=1e-10):
            raise ValueError("Asymmetry selected normalization does not close")
        for attribute in ("event_second_sumw", "mc_second_sumw2"):
            values = getattr(self, attribute)
            if not np.allclose(values, values.T, rtol=1e-9, atol=1e-10):
                raise ValueError("Asymmetry second moments are not symmetric")

    def to_arrays(self, prefix: str) -> Dict[str, np.ndarray]:
        """Return an NPZ/checkpoint-compatible mapping with ``prefix__`` keys."""
        return {
            prefix + "__" + attribute: np.asarray(getattr(self, attribute)).copy()
            for attribute in ("edges", "sumw", "event_second_sumw", "mc_second_sumw2",
                              "positive_sumw", "negative_sumw", "zero_sumw",
                              "event_count", "zero_entries")
        }

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], prefix: str) -> "BeamAsymmetryMoments":
        """Restore serialized moments; missing state is a clear archive error."""
        result = cls(np.asarray(arrays[prefix + "__edges"], dtype=np.float64))
        for attribute in ("sumw", "event_second_sumw", "mc_second_sumw2",
                          "positive_sumw", "negative_sumw"):
            setattr(result, attribute,
                    np.asarray(arrays[prefix + "__" + attribute], dtype=np.float64).copy())
        result.zero_sumw = float(np.asarray(arrays[prefix + "__zero_sumw"]).item())
        result.event_count = int(np.asarray(arrays[prefix + "__event_count"]).item())
        result.zero_entries = int(np.asarray(arrays[prefix + "__zero_entries"]).item())
        result.validate()
        return result


def _ratio_jacobian(numerator: np.ndarray, denominator: float,
                    factors: np.ndarray) -> np.ndarray:
    bins = len(numerator)
    jacobian = np.zeros((bins, bins + 2), dtype=np.float64)
    jacobian[np.arange(bins), np.arange(bins)] = factors / denominator
    jacobian[:, -1] = -factors * numerator / (denominator * denominator)
    return jacobian


def _propagate(jacobian: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    result = jacobian @ covariance @ jacobian.T
    return 0.5 * (result + result.T)


def asymmetry_statistics(results: Iterable[Any], luminosities: Sequence[float],
                         cross_sections_pb: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """Combine processes with one selected-channel denominator.

    ``results`` have ``spec.name``, ``spec.cross_section_pb``,
    ``generated_sumw`` and a ``pull_asymmetry_moments`` mapping. Cross-section
    overrides allow generator comparisons to use the same process rates.
    Output is JSON-safe, including signed per-process contributions whose
    sum is precisely the total asymmetry.
    """
    samples = list(results.values()) if isinstance(results, Mapping) else list(results)
    if not samples:
        raise ValueError("Asymmetry statistics require at least one sample")
    luminosities = tuple(float(value) for value in luminosities)
    if any(not math.isfinite(value) or value <= 0.0 for value in luminosities):
        raise ValueError("Asymmetry luminosities must be finite and positive")
    names = [result.spec.name for result in samples]
    if len(set(names)) != len(names):
        raise ValueError("Asymmetry process names must be unique within a channel")

    output: Dict[str, Any] = {}
    for key in ASYMMETRY_KEYS:
        first = None
        event_second = None
        mc_second = None
        process_first: Dict[str, np.ndarray] = {}
        process_positive: Dict[str, np.ndarray] = {}
        process_negative: Dict[str, np.ndarray] = {}
        process_zero: Dict[str, float] = {}
        edges = None
        events = 0
        zero_entries = 0
        for result in samples:
            name = result.spec.name
            moments = getattr(result, "pull_asymmetry_moments", {}).get(key)
            if moments is None:
                raise ValueError("Sample %s has no %s moments; reanalyse its events" % (name, key))
            moments.validate()
            if edges is None:
                edges = moments.edges.copy()
                first = np.zeros_like(moments.sumw)
                event_second = np.zeros_like(moments.event_second_sumw)
                mc_second = np.zeros_like(moments.mc_second_sumw2)
            elif not np.array_equal(edges, moments.edges):
                raise ValueError("Asymmetry sample bin edges differ")
            generated_sumw = float(result.generated_sumw)
            sigma = float(result.spec.cross_section_pb if cross_sections_pb is None
                          else cross_sections_pb[name])
            if generated_sumw == 0.0 or not math.isfinite(generated_sumw):
                raise ValueError("Asymmetry generated normalization must be finite and nonzero")
            if not math.isfinite(sigma) or sigma < 0.0:
                raise ValueError("Asymmetry process cross sections must be finite and nonnegative")
            scale = sigma / generated_sumw
            process_first[name] = moments.sumw * scale
            process_positive[name] = moments.positive_sumw * scale
            process_negative[name] = moments.negative_sumw * scale
            process_zero[name] = moments.zero_sumw * scale
            first += process_first[name]
            event_second += moments.event_second_sumw * scale
            mc_second += moments.mc_second_sumw2 * scale * scale
            events += moments.event_count
            zero_entries += moments.zero_entries

        denominator = float(first[-1])
        if not math.isfinite(denominator) or denominator < 0.0:
            raise ValueError("Asymmetry selected cross section must be finite and nonnegative")
        bins = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        factors = centers / np.diff(edges)
        payload: Dict[str, Any] = {
            "bin_edges": edges.tolist(), "centers": centers.tolist(),
            "A": None, "mc_covariance": None, "mc_errors": None,
            "data_covariances": {str(value): None for value in luminosities},
            "data_errors": {str(value): None for value in luminosities},
            "denominator_pb": denominator, "numerator_pb": first[:bins].tolist(),
            "process_contributions": {name: None for name in names},
            "process_denominators_pb": {name: float(value[-1])
                                         for name, value in process_first.items()},
            "process_numerators_pb": {name: value[:bins].tolist()
                                       for name, value in process_first.items()},
            "event_count": events,
            "overflow": {
                "signed_numerator_pb": float(first[bins]),
                "positive_pb": sum(float(value[-1]) for value in process_positive.values()),
                "negative_pb": sum(float(value[-1]) for value in process_negative.values()),
                "included_in_denominator": True,
            },
            "zero": {"cross_section_pb": sum(process_zero.values()),
                     "entries": zero_entries, "included_in_denominator": True},
            "definition": {
                "model": ASYMMETRY_MODEL,
                "projection": "t_beam" if key == ASYMMETRY_KEYS[0] else "t_phi",
                "formula": "A_i=(center_i/width_i)*(W_i_positive-W_i_negative)/W_selected",
                "jet_weight_fraction": 0.5,
                "sign_rule": "strict positive/negative; exact zero contributes only to denominator",
                "denominator": "all selected events, including zeros and overflow",
                "overflow_rule": "abs(projection)>=last_bin_edge; never folded into finite bins",
                "pull_magnitude_cut_applied": False,
                "mc_covariance": "event-level sum(w^2 f f^T), including denominator correlations",
                "data_covariance": "event-level sum(w f f^T)/(1000*luminosity_fb), including denominator correlations",
            },
        }
        if denominator > 0.0:
            jacobian = _ratio_jacobian(first[:bins], denominator, factors)
            covariance = _propagate(jacobian, mc_second)
            payload["A"] = (factors * first[:bins] / denominator).tolist()
            payload["mc_covariance"] = covariance.tolist()
            payload["mc_errors"] = np.sqrt(np.maximum(np.diag(covariance), 0.0)).tolist()
            for luminosity in luminosities:
                covariance = _propagate(jacobian, event_second) / (1000.0 * luminosity)
                payload["data_covariances"][str(luminosity)] = covariance.tolist()
                payload["data_errors"][str(luminosity)] = np.sqrt(
                    np.maximum(np.diag(covariance), 0.0)).tolist()
            payload["process_contributions"] = {
                name: (factors * values[:bins] / denominator).tolist()
                for name, values in process_first.items()
            }
        output[key] = payload
    return output


def ratio_statistics(variation: Mapping[str, Any], reference: Mapping[str, Any],
                     error_model: str, luminosity: Optional[float] = None) -> Dict[str, Any]:
    """Ratios to nominal Herwig, masking only zero/nonfinite denominators.

    Projected-data ratios hold the reference prediction fixed. Finite-MC
    ratios propagate both independent simulations, including bin covariance.
    Masked rows/columns are exported as null, rather than fabricated zeros.
    """
    if not np.array_equal(variation["bin_edges"], reference["bin_edges"]):
        raise ValueError("Asymmetry ratio bin edges differ")
    bins = len(reference["bin_edges"]) - 1
    if variation["A"] is None or reference["A"] is None:
        return {"ratio": [None] * bins, "valid": [False] * bins,
                "denominator_valid": [False] * bins, "numerical_valid": [False] * bins,
                "covariance": [[None] * bins for _ in range(bins)],
                "errors": [None] * bins}
    numerator = np.asarray(variation["A"], dtype=np.float64)
    denominator = np.asarray(reference["A"], dtype=np.float64)
    denominator_valid = np.isfinite(denominator) & (denominator != 0.0)
    valid = np.isfinite(numerator) & denominator_valid
    if error_model in ("mc", "finite_mc"):
        numerator_cov = np.asarray(variation["mc_covariance"], dtype=np.float64)
        denominator_cov = np.asarray(reference["mc_covariance"], dtype=np.float64)
    elif error_model in ("data", "expected_data"):
        if luminosity is None:
            raise ValueError("Projected-data asymmetry ratios require a luminosity")
        numerator_cov = np.asarray(variation["data_covariances"][str(float(luminosity))],
                                   dtype=np.float64)
        denominator_cov = None
    else:
        raise ValueError("Unknown asymmetry ratio error model: " + error_model)
    ratios = np.full(bins, np.nan)
    covariance = np.full((bins, bins), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        ratios[valid] = numerator[valid] / denominator[valid]
        # Divide covariances sequentially. Squaring a very small denominator
        # underflows even when the ratio and its variance are representable.
        indices = np.flatnonzero(valid & np.isfinite(ratios))
        support = np.ix_(indices, indices)
        selected_denominator = denominator[indices]
        selected_covariance = (numerator_cov[support]
                               / selected_denominator[:, None]
                               / selected_denominator[None, :])
        if denominator_cov is not None:
            reference_term = (denominator_cov[support]
                              / selected_denominator[:, None]
                              / selected_denominator[None, :])
            # Avoid multiplying an exact-zero covariance by an overflowing
            # ratio product; expected-data ratios omit this term altogether.
            reference_term = (reference_term * ratios[indices, None]
                              * ratios[None, indices])
            selected_covariance += reference_term
        covariance[support] = selected_covariance
    numerical_valid = valid & np.isfinite(ratios) & np.isfinite(np.diag(covariance))
    return {
        "ratio": [float(value) if accepted and math.isfinite(value) else None
                  for value, accepted in zip(ratios, valid)],
        "valid": valid.tolist(),
        "denominator_valid": denominator_valid.tolist(),
        "numerical_valid": numerical_valid.tolist(),
        "covariance": [[float(covariance[i, j]) if valid[i] and valid[j]
                        and math.isfinite(covariance[i, j]) else None
                        for j in range(bins)] for i in range(bins)],
        "errors": [math.sqrt(max(float(covariance[i, i]), 0.0)) if numerical_valid[i] else None
                   for i in range(bins)],
        "error_model": error_model,
    }
