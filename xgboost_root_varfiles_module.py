"""Small, import-safe XGBoost helpers for the PullPheno analysis.

ROOT event reconstruction intentionally lives in :mod:`HwSimPythonAnalysis`.
This module only accepts NumPy arrays, which keeps splitting, weighting and
threshold selection independently testable even when XGBoost is unavailable.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np


FEATURE_NAMES: Tuple[str, ...] = (
    "mjj",
    "abs_delta_yjj",
    "leading_jet_pt",
    "subleading_jet_pt",
    "boson_pt",
    "zstar",
)

MODEL_PARAMETERS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 1,
    "n_jobs": 1,
}

CROSS_FIT_FOLDS = 5
CROSS_FIT_SEED = 1


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }


@dataclass(frozen=True)
class CrossFitSplit:
    """One leakage-free outer-fold pipeline.

    ``test`` is the physics-application fold, ``validation`` fixes this
    pipeline's score threshold, and ``train`` contains the remaining folds.
    """

    fold: int
    validation_fold: int
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def as_split_indices(self) -> SplitIndices:
        return SplitIndices(
            train=self.train,
            validation=self.validation,
            test=self.test,
        )


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    significance: float
    signal_weight: float
    background_weight: float
    selected_count: int


def _sample_seed(sample_key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{sample_key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def deterministic_split(
    number_of_events: int,
    sample_key: str,
    seed: int = 1,
) -> SplitIndices:
    """Return an exact, deterministic 60/20/20 split for one process."""
    number_of_events = int(number_of_events)
    if number_of_events < 0:
        raise ValueError("number_of_events must be non-negative")
    generator = np.random.default_rng(_sample_seed(sample_key, seed))
    permutation = generator.permutation(number_of_events).astype(np.int64, copy=False)
    train_end = int(math.floor(0.60 * number_of_events))
    validation_end = train_end + int(math.floor(0.20 * number_of_events))
    return SplitIndices(
        train=permutation[:train_end],
        validation=permutation[train_end:validation_end],
        test=permutation[validation_end:],
    )


def deterministic_crossfit_splits(
    number_of_events: int,
    sample_key: str,
    folds: int = CROSS_FIT_FOLDS,
    seed: int = CROSS_FIT_SEED,
) -> Tuple[CrossFitSplit, ...]:
    """Return rotating, deterministic train/validation/test fold pipelines.

    The shuffled population is partitioned exactly once into ``folds`` outer
    folds.  For pipeline ``k``, fold ``k`` is used only for physics testing,
    fold ``(k + 1) % folds`` fixes the score threshold, and every other fold
    trains the classifier.  Consequently each event is tested once,
    validated once and trained on ``folds - 2`` times.  With five folds this
    is the requested rotating 60/20/20 construction, up to unavoidable
    one-event differences when the population is not divisible by five.
    """
    number_of_events = int(number_of_events)
    folds = int(folds)
    if number_of_events < 0:
        raise ValueError("number_of_events must be non-negative")
    if folds < 3:
        raise ValueError("Cross-fitting requires at least three folds")
    if number_of_events < folds:
        raise ValueError(
            f"Cross-fitting {number_of_events} events into {folds} folds would create an empty fold"
        )
    generator = np.random.default_rng(_sample_seed(sample_key, seed))
    permutation = generator.permutation(number_of_events).astype(np.int64, copy=False)
    partitions = tuple(
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(permutation, folds)
    )
    pipelines = []
    for test_fold in range(folds):
        validation_fold = (test_fold + 1) % folds
        training_folds = [
            partitions[index]
            for index in range(folds)
            if index not in (test_fold, validation_fold)
        ]
        pipelines.append(
            CrossFitSplit(
                fold=test_fold,
                validation_fold=validation_fold,
                train=np.concatenate(training_folds).astype(np.int64, copy=False),
                validation=partitions[validation_fold],
                test=partitions[test_fold],
            )
        )
    return tuple(pipelines)


def inverse_split_probability(number_of_events: int, selected_count: int) -> float:
    """Horvitz--Thompson correction for an exact random subset."""
    number_of_events = int(number_of_events)
    selected_count = int(selected_count)
    if number_of_events <= 0 or selected_count <= 0 or selected_count > number_of_events:
        raise ValueError("Invalid split population or subset size")
    return number_of_events / selected_count


def balanced_training_weights(
    physical_weights: Sequence[float],
    labels: Sequence[int],
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Balance signal/background totals while retaining within-class weights.

    The balanced weights have mean one and each class carries half of the
    total weight.  Keeping the absolute total near the event count avoids
    changing XGBoost regularisation scales such as ``min_child_weight``.
    """
    weights = np.asarray(physical_weights, dtype=np.float64)
    classes = np.asarray(labels, dtype=np.int8)
    if weights.ndim != 1 or classes.shape != weights.shape:
        raise ValueError("Weights and labels must be one-dimensional arrays of equal length")
    if len(weights) == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("XGBoost training weights must be finite and strictly positive")
    if not np.all(np.isin(classes, (0, 1))):
        raise ValueError("Labels must be binary with signal=1 and background=0")
    signal = classes == 1
    background = ~signal
    signal_total = float(np.sum(weights[signal], dtype=np.float64))
    background_total = float(np.sum(weights[background], dtype=np.float64))
    if signal_total <= 0.0 or background_total <= 0.0:
        raise ValueError("Both signal and background are required for training")
    target = 0.5 * len(weights)
    signal_factor = target / signal_total
    background_factor = target / background_total
    balanced = weights.copy()
    balanced[signal] *= signal_factor
    balanced[background] *= background_factor
    return balanced, {
        "signal_factor": signal_factor,
        "background_factor": background_factor,
        "signal_sum": float(np.sum(balanced[signal], dtype=np.float64)),
        "background_sum": float(np.sum(balanced[background], dtype=np.float64)),
        "total_sum": float(np.sum(balanced, dtype=np.float64)),
    }


def optimize_significance_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    physical_weights: Sequence[float],
) -> ThresholdResult:
    """Maximise S/sqrt(S+B) over distinct score thresholds.

    Events pass when ``score >= threshold``.  Since scores are traversed from
    high to low, ``argmax`` resolves exact ties in favour of the tighter cut.
    """
    score_array = np.asarray(scores, dtype=np.float64)
    classes = np.asarray(labels, dtype=np.int8)
    weights = np.asarray(physical_weights, dtype=np.float64)
    if score_array.ndim != 1 or classes.shape != score_array.shape or weights.shape != score_array.shape:
        raise ValueError("Scores, labels and weights must be equal-length one-dimensional arrays")
    if len(scores) == 0:
        raise ValueError("Cannot optimise a threshold on an empty validation sample")
    if not np.all(np.isfinite(score_array)) or not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Validation scores and weights must be finite with non-negative weights")
    if not np.all(np.isin(classes, (0, 1))) or not np.any(classes == 1) or not np.any(classes == 0):
        raise ValueError("Validation data must contain signal and background")

    order = np.argsort(-score_array, kind="mergesort")
    ordered_scores = score_array[order]
    ordered_labels = classes[order]
    ordered_weights = weights[order]
    cumulative_signal = np.cumsum(ordered_weights * (ordered_labels == 1), dtype=np.float64)
    cumulative_background = np.cumsum(ordered_weights * (ordered_labels == 0), dtype=np.float64)
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0.0), len(ordered_scores) - 1]
    signal = cumulative_signal[group_ends]
    background = cumulative_background[group_ends]
    denominator = np.sqrt(np.maximum(signal + background, 0.0))
    significance = np.divide(signal, denominator, out=np.zeros_like(signal), where=denominator > 0.0)
    best = int(np.argmax(significance))
    end = int(group_ends[best])
    return ThresholdResult(
        threshold=float(ordered_scores[end]),
        significance=float(significance[best]),
        signal_weight=float(signal[best]),
        background_weight=float(background[best]),
        selected_count=end + 1,
    )


def weighted_roc_curve(
    labels: Sequence[int],
    scores: Sequence[float],
    weights: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return a physical-weight ROC curve and trapezoidal AUC."""
    classes = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=np.float64)
    weight_array = np.asarray(weights, dtype=np.float64)
    if classes.shape != score_array.shape or classes.shape != weight_array.shape or classes.ndim != 1:
        raise ValueError("ROC inputs must be equal-length one-dimensional arrays")
    positive_total = float(np.sum(weight_array[classes == 1], dtype=np.float64))
    negative_total = float(np.sum(weight_array[classes == 0], dtype=np.float64))
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("ROC data must contain positive-weight signal and background")
    order = np.argsort(-score_array, kind="mergesort")
    ordered_scores = score_array[order]
    ordered_labels = classes[order]
    ordered_weights = weight_array[order]
    cumulative_positive = np.cumsum(ordered_weights * (ordered_labels == 1), dtype=np.float64)
    cumulative_negative = np.cumsum(ordered_weights * (ordered_labels == 0), dtype=np.float64)
    group_ends = np.r_[np.flatnonzero(np.diff(ordered_scores) != 0.0), len(ordered_scores) - 1]
    tpr = np.r_[0.0, cumulative_positive[group_ends] / positive_total]
    fpr = np.r_[0.0, cumulative_negative[group_ends] / negative_total]
    thresholds = np.r_[math.inf, ordered_scores[group_ends]]
    auc = float(np.sum(np.diff(fpr) * 0.5 * (tpr[:-1] + tpr[1:]), dtype=np.float64))
    return fpr, tpr, thresholds, auc


def weighted_confusion(
    labels: Sequence[int],
    scores: Sequence[float],
    weights: Sequence[float],
    threshold: float,
) -> np.ndarray:
    classes = np.asarray(labels, dtype=np.int8)
    predictions = np.asarray(scores, dtype=np.float64) >= float(threshold)
    weight_array = np.asarray(weights, dtype=np.float64)
    matrix = np.zeros((2, 2), dtype=np.float64)
    for truth in (0, 1):
        for prediction in (0, 1):
            mask = (classes == truth) & (predictions == bool(prediction))
            matrix[truth, prediction] = np.sum(weight_array[mask], dtype=np.float64)
    return matrix


def create_classifier(parameters: Mapping[str, Any] | None = None) -> Any:
    try:
        import xgboost as xgb  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "XGBoost analysis requested but xgboost is unavailable; install requirements-xgboost.txt"
        ) from error
    settings = dict(MODEL_PARAMETERS)
    if parameters:
        settings.update(parameters)
    return xgb.XGBClassifier(**settings)


def train_classifier(features: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> Any:
    matrix = np.asarray(features, dtype=np.float64)
    classes = np.asarray(labels, dtype=np.int8)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"Training matrix must have {len(FEATURE_NAMES)} columns in the fixed feature order")
    if len(matrix) != len(classes) or len(matrix) != len(sample_weights):
        raise ValueError("Training feature, label and weight lengths differ")
    classifier = create_classifier()
    classifier.fit(matrix, classes, sample_weight=sample_weights, verbose=False)
    return classifier


def signal_scores(classifier: Any, features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    probabilities = np.asarray(classifier.predict_proba(matrix), dtype=np.float64)
    if probabilities.shape != (len(matrix), 2):
        raise RuntimeError(f"Unexpected XGBoost probability shape: {probabilities.shape}")
    scores = probabilities[:, 1]
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise RuntimeError("XGBoost returned invalid signal probabilities")
    return scores


def save_classifier(classifier: Any, destination: Path) -> str:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite model: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    classifier.save_model(str(destination))
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    return digest


def load_classifier(source: Path) -> Any:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"XGBoost model does not exist: {source}")
    classifier = create_classifier()
    classifier.load_model(str(source))
    return classifier


def runtime_versions() -> Dict[str, str]:
    import numpy

    try:
        import xgboost  # type: ignore

        xgboost_version = str(xgboost.__version__)
    except ImportError:
        xgboost_version = "unavailable"
    return {"numpy": str(numpy.__version__), "xgboost": xgboost_version}
