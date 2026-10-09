"""Plots and portable numerical exports for beam-oriented pull asymmetries.

The two signs of each process are stacked separately, while every process
contribution uses the same selected channel denominator.  Ratios retain negative
reference predictions and mask only unavailable or exactly zero denominators.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import plot_style
from beam_asymmetry import ASYMMETRY_KEYS, asymmetry_statistics, ratio_statistics as moment_ratio_statistics

VIEWS = {"full": (0.0, 0.03), "near-zero": (0.0, 0.008)}
OBSERVABLE_LABELS = {
    "pull_asymmetry_beam": ("Beam pull asymmetry", r"$|t_{\mathrm{beam}}|$", r"$A_{\mathrm{beam}}$"),
    "pull_asymmetry_phi": ("Azimuthal pull asymmetry", r"$|t_\phi|$", r"$A_\phi$"),
}


def _plain(value: Any) -> Any:
    """Convert arrays and nonfinite entries to strict, portable JSON values."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def ratio_statistics(
    prediction: Mapping[str, Any], reference: Mapping[str, Any],
    luminosities: Sequence[float],
) -> dict[str, Any]:
    """Propagate independent MC totals; projected data holds the reference fixed.

    Covariances preserve correlations between bins. Invalid rows and columns
    are exported as null, accompanied by an explicit bin validity mask.
    """
    finite_mc = moment_ratio_statistics(prediction, reference, "mc")
    projected_data = {str(float(lumi)): moment_ratio_statistics(prediction, reference, "data", lumi)
                      for lumi in luminosities}
    count = len(reference["centers"])
    reference_error = np.full(count, np.nan)
    valid = np.asarray(finite_mc["valid"], dtype=bool)
    if reference.get("A") is not None and reference.get("mc_covariance") is not None:
        denominator = _array(reference["A"])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            reference_error[valid] = np.sqrt(np.maximum(
                np.diag(_array(reference["mc_covariance"]))[valid], 0.0,
            )) / np.abs(denominator[valid])
    return {
        "central": finite_mc["ratio"], "valid": finite_mc["valid"],
        "denominator_valid": finite_mc["denominator_valid"],
        "numerical_valid": {"mc": finite_mc["numerical_valid"],
                            "data": {lumi: statistics["numerical_valid"]
                                     for lumi, statistics in projected_data.items()}},
        "mc_covariance": finite_mc["covariance"],
        "data_covariances": {lumi: statistics["covariance"]
                             for lumi, statistics in projected_data.items()},
        "reference_mc_relative_error": _plain(reference_error),
        "mc_reference": "independent finite-MC reference covariance included",
        "data_reference": "reference central prediction fixed",
    }


def signed_stack(contributions: Mapping[str, Any]) -> dict[str, Any]:
    """Return positive and negative stack positions and their algebraic total."""
    if not contributions:
        raise ValueError("An asymmetry stack needs at least one process")
    count = len(next(iter(contributions.values())))
    positive = np.zeros(count)
    negative = np.zeros(count)
    bottoms, values = {}, {}
    for name, contribution in contributions.items():
        vector = _array(contribution)
        if vector.shape != (count,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid asymmetry process contribution: {name}")
        bottoms[name] = np.where(vector >= 0.0, positive, negative).copy()
        values[name] = vector
        positive += np.maximum(vector, 0.0)
        negative += np.minimum(vector, 0.0)
    return {"bottoms": bottoms, "values": values, "positive": positive,
            "negative": negative, "total": positive + negative}


def build_sample_payload(results: Sequence[Any], luminosities: Sequence[float]) -> dict[str, Any]:
    predictions = []
    for result in results:
        predictions.append({
            "strategy": result.strategy, "channel": result.spec.channel,
            "sample": result.spec.name,
            "observables": asymmetry_statistics([result], luminosities),
        })
    return _plain({"schema_version": 1, "mode": "sample", "predictions": predictions,
                   "luminosities_fb": list(luminosities), "ratios": [], "plots": []})


def build_comparison_payload(
    sources: Sequence[Any], analyses: Sequence[str], luminosities: Sequence[float],
    reference_cross_sections: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if len(sources) < 2:
        raise ValueError("Asymmetry comparisons need a reference and at least one variation")
    predictions, ratios = [], []
    for strategy in analyses:
        channels = sorted({result.spec.channel for result in sources[0].results if result.strategy == strategy})
        for channel in channels:
            channel_statistics = []
            for source in sources:
                results = [result for result in source.results
                           if result.strategy == strategy and result.spec.channel == channel]
                if not results:
                    raise ValueError(f"Missing asymmetry results: {source.scenario.identifier}/{strategy}/{channel}")
                observables = asymmetry_statistics(results, luminosities, reference_cross_sections)
                predictions.append({"strategy": strategy, "channel": channel,
                                    "scenario": source.scenario.identifier, "observables": observables})
                channel_statistics.append(observables)
            for source, observables in zip(sources[1:], channel_statistics[1:]):
                for key in ASYMMETRY_KEYS:
                    if not np.array_equal(observables[key]["bin_edges"], channel_statistics[0][key]["bin_edges"]):
                        raise ValueError(f"Asymmetry bin edges differ: {key}/{source.scenario.identifier}")
                    ratios.append({"strategy": strategy, "channel": channel, "observable": key,
                                   "scenario": source.scenario.identifier,
                                   "reference": sources[0].scenario.identifier,
                                   **ratio_statistics(observables[key], channel_statistics[0][key], luminosities)})
    return _plain({"schema_version": 1, "mode": "comparison", "predictions": predictions,
                   "luminosities_fb": list(luminosities), "ratios": ratios, "plots": []})


def write_artifacts(run_dir: Path, payload: Mapping[str, Any]) -> None:
    """Write strict JSON and lossless numeric arrays with explicit validity masks."""
    directory = Path(run_dir) / "summaries"
    directory.mkdir(parents=True, exist_ok=True)
    json_path, npz_path = directory / "asymmetries.json", directory / "asymmetries.npz"
    if json_path.exists() or npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite asymmetry artifacts in {directory}")
    arrays = {}
    for index, prediction in enumerate(payload["predictions"]):
        for key, statistics in prediction["observables"].items():
            prefix = f"prediction_{index}__{key}"
            for field in ("bin_edges", "centers", "A", "mc_covariance", "numerator_pb"):
                if statistics.get(field) is not None:
                    arrays[f"{prefix}__{field}"] = _array(statistics[field])
            for lumi, covariance in statistics.get("data_covariances", {}).items():
                if covariance is not None:
                    arrays[f"{prefix}__data_covariance__{lumi}"] = _array(covariance)
            for process, contribution in statistics.get("process_contributions", {}).items():
                if contribution is not None:
                    arrays[f"{prefix}__process__{process}"] = _array(contribution)
            arrays[f"{prefix}__denominator_pb"] = np.asarray(statistics["denominator_pb"])
    for index, ratio in enumerate(payload.get("ratios", [])):
        prefix = f"ratio_{index}"
        arrays[f"{prefix}__valid"] = np.asarray(ratio["valid"], dtype=bool)
        arrays[f"{prefix}__denominator_valid"] = np.asarray(ratio["denominator_valid"], dtype=bool)
        arrays[f"{prefix}__mc_numerical_valid"] = np.asarray(ratio["numerical_valid"]["mc"], dtype=bool)
        for lumi, valid in ratio["numerical_valid"]["data"].items():
            arrays[f"{prefix}__data_numerical_valid__{lumi}"] = np.asarray(valid, dtype=bool)
        arrays[f"{prefix}__central"] = _array(ratio["central"])
        arrays[f"{prefix}__reference_mc_relative_error"] = _array(ratio["reference_mc_relative_error"])
        if ratio.get("mc_covariance") is not None:
            arrays[f"{prefix}__mc_covariance"] = _array(ratio["mc_covariance"])
        for lumi, covariance in ratio.get("data_covariances", {}).items():
            if covariance is not None:
                arrays[f"{prefix}__data_covariance__{lumi}"] = _array(covariance)
    # Keep the JSON index in the archive so the numerical arrays are self-describing.
    serialized = json.dumps(_plain(payload), indent=2, allow_nan=False) + "\n"
    arrays["metadata_json"] = np.asarray(serialized)
    with json_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    with npz_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def _pyplot() -> Any:
    return plot_style.pyplot()


def _slug(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")


def _errors(covariance: Any, count: int) -> np.ndarray:
    return np.zeros(count) if covariance is None else np.sqrt(np.maximum(np.diag(_array(covariance)), 0.0))


def _window_limits(axis: Any, centers: np.ndarray, window: tuple[float, float],
                   vectors: Sequence[np.ndarray], baseline: float = 0.0) -> None:
    """Fit the vertical range to the bins inside the horizontal window, with errors."""
    lower, upper = window
    visible = (centers >= lower) & (centers <= upper)
    finite_values = [baseline]
    for vector in vectors:
        entries = _array(vector)[visible]
        finite_values.extend(entries[np.isfinite(entries)].tolist())
    ymin, ymax = min(finite_values), max(finite_values)
    span = ymax - ymin
    padding = 0.10 * span if span else max(abs(ymin) * 0.10, 0.01)
    axis.set_ylim(ymin - padding, ymax + padding)


def _visible_limits(axis: Any, centers: np.ndarray, view: str,
                    vectors: Sequence[np.ndarray], baseline: float = 0.0) -> None:
    """Give the near-zero view its own vertical range, including error bars."""
    if view != "near-zero":
        return
    _window_limits(axis, centers, VIEWS[view], vectors, baseline)


def _view_window(view: str, edges: np.ndarray, vectors: Sequence[np.ndarray]) -> tuple[float, float]:
    """The near-zero view is fixed; the full view ends after the last populated bin."""
    lower, upper = VIEWS[view]
    if view != "full":
        return lower, upper
    return lower, min(upper, plot_style.populated_limits(edges, vectors)[1])


def _decorate(axis: Any, key: str, window: tuple[float, float], ratio: bool = False) -> None:
    _, xlabel, ylabel = OBSERVABLE_LABELS[key]
    axis.set_xlim(*window)
    axis.axhline(0.0, color=plot_style.GUIDE, linewidth=0.6, zorder=4)
    plot_style.scaled_axis(axis, ylabel, max(abs(value) for value in axis.get_ylim()))
    if not ratio:
        axis.set_xlabel(xlabel)


def _header(channel: str, strategy: str, cuts: Mapping[str, float] | None, luminosity: float | None,
            prefix: str | None, particle_pt_min_gev: float | None, partial: bool) -> str:
    # Asymmetries always use both tagging jets, whatever pull-magnitude cut the run applies.
    return plot_style.header(prefix=prefix, channel=channel, cuts=cuts, luminosity=luminosity,
                             threshold=None, particle_pt_min_gev=particle_pt_min_gev,
                             notes=["XGBoost selection"] if strategy == "xgboost" else [],
                             partial=partial)


def _save(figure: Any, run_dir: Path, relative: Path, particle_pt_min_gev: float | None) -> dict[str, str]:
    """Write PNG and PDF copies; the header already states ``particle_pt_min_gev``."""
    paths = {}
    for extension in ("png", "pdf"):
        path = Path(run_dir) / relative.with_suffix(f".{extension}")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite plot: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=plot_style.PNG_DPI, bbox_inches="tight")
        paths[extension] = path.relative_to(run_dir).as_posix()
    return paths


@plot_style.paper_figures
def generate_sample_plots(
    run_dir: Path, results: Sequence[Any], luminosities: Sequence[float],
    partial: bool = False, scenario_label: str | None = None,
    particle_pt_min_gev: float | None = None, payload: dict[str, Any] | None = None,
    cuts: Mapping[str, float] | None = None,
) -> list[dict[str, Any]]:
    payload = build_sample_payload(results, luminosities) if payload is None else payload
    plt, records = _pyplot(), []
    lookup = {(item["strategy"], item["sample"]): item for item in payload["predictions"]}
    modes = [("mc-stat", None)] + [("data-stat", float(lumi)) for lumi in luminosities]
    for result in results:
        label = plot_style.paper_label(result.spec.label)
        for key, statistics in lookup[(result.strategy, result.spec.name)]["observables"].items():
            defined = statistics.get("A") is not None
            centers, edges = map(_array, (statistics["centers"], statistics["bin_edges"]))
            values = _array(statistics["A"]) if defined else None
            for uncertainty, luminosity in modes:
                data = uncertainty == "data-stat"
                covariance = (statistics["data_covariances"].get(str(float(luminosity)))
                              if data else statistics["mc_covariance"])
                errors = _errors(covariance, len(centers)) if defined else None
                for view in VIEWS:
                    figure, axis = plt.subplots(figsize=plot_style.SINGLE_FIGSIZE, constrained_layout=True)
                    handle = plot_style.reference_handle(result.spec.color)
                    if defined:
                        axis.stairs(values, edges, color=result.spec.color, linewidth=1.1, zorder=6)
                        axis.errorbar(centers, values, yerr=errors, fmt="none", color=result.spec.color,
                                      elinewidth=plot_style.ERROR_LINE_WIDTH, capsize=plot_style.CAP_SIZE,
                                      zorder=7)
                        entry = (handle, f"{label}, {'expected stat.' if data else 'MC stat.'}")
                        window = _view_window(view, edges, [values, errors])
                        _window_limits(axis, centers, window, [values-errors, values+errors])
                    else:
                        entry = (handle, label)
                        axis.text(0.5, 0.55, "Asymmetry undefined: no selected events", ha="center", va="center",
                                  transform=axis.transAxes, color=plot_style.MUTED)
                        axis.set_ylim(-0.1, 0.1)
                        window = VIEWS[view]
                    _decorate(axis, key, window)
                    plot_style.legend_above(axis, [[entry]], title=_header(
                        result.spec.channel, result.strategy, cuts, luminosity, scenario_label,
                        particle_pt_min_gev, partial))
                    relative = (Path("plots") / result.strategy / "asymmetries" / result.spec.channel /
                                _slug(result.spec.name))
                    if data:
                        relative = relative / "data-stat" / f"{luminosity:g}fb"
                    relative = relative / view / key
                    records.append({"strategy": result.strategy, "observable": key,
                                    "title": f"{OBSERVABLE_LABELS[key][0]} · {result.spec.label} · {uncertainty} · {view}",
                                    "channel": result.spec.channel,
                                    "stage": "xgboost" if result.strategy == "xgboost" else "vbf",
                                    "kind": "asymmetry-shape-data-stat" if data else "asymmetry-shape",
                                    "sample": result.spec.name, "view": view, "uncertainty": uncertainty,
                                    "error_mode": "data" if data else "mc", "defined": defined,
                                    "luminosity_fb": luminosity,
                                    **_save(figure, run_dir, relative, particle_pt_min_gev)})
                    plt.close(figure)
    payload["plots"] = records
    write_artifacts(run_dir, payload)
    return records


@plot_style.paper_figures
def generate_comparison_plots(
    run_dir: Path, sources: Sequence[Any], analyses: Sequence[str], luminosities: Sequence[float],
    payload: dict[str, Any] | None = None, *,
    reference_cross_sections: Mapping[str, float] | None = None,
    include_reference_stacks: bool = True,
) -> list[dict[str, Any]]:
    payload = build_comparison_payload(sources, analyses, luminosities, reference_cross_sections) if payload is None else payload
    plt, records = _pyplot(), []
    predictions = {(item["strategy"], item["channel"], item["scenario"]): item["observables"]
                   for item in payload["predictions"]}
    ratios = {(item["strategy"], item["channel"], item["observable"], item["scenario"]): item
              for item in payload["ratios"]}
    reference_id = sources[0].scenario.identifier
    label_prefix, scenario_labels = plot_style.scenario_labels([source.scenario.label for source in sources])
    reference_label = scenario_labels[0]
    cuts = getattr(sources[0].config, "cuts", None)
    particle_pt_min_gev = sources[0].config.particle_pt_min_gev
    partial = bool((getattr(sources[0], "metadata", None) or {}).get("partial"))
    for strategy in analyses:
        channels = sorted({channel for key_strategy, channel, scenario in predictions
                           if key_strategy == strategy and scenario == reference_id})
        for channel in channels:
            processes = sorted([result for result in sources[0].results
                                if result.strategy == strategy and result.spec.channel == channel],
                               key=lambda result: result.spec.stack_order)
            for key in ASYMMETRY_KEYS:
                reference = predictions[(strategy, channel, reference_id)][key]
                if reference.get("A") is None:
                    continue
                edges, centers, total = map(_array, (reference["bin_edges"], reference["centers"], reference["A"]))
                widths = np.diff(edges)
                stack = signed_stack({result.spec.name: reference["process_contributions"][result.spec.name]
                                      for result in processes})
                if not np.allclose(stack["total"], total, rtol=1e-11, atol=1e-14):
                    raise RuntimeError(f"Reference asymmetry stack does not close: {strategy}/{channel}/{key}")
                for luminosity in luminosities:
                    for uncertainty in ("mc-stat", "data-stat"):
                        data = uncertainty == "data-stat"
                        covariance = reference["data_covariances"].get(str(float(luminosity))) if data else reference["mc_covariance"]
                        errors = _errors(covariance, len(total))
                        for view in VIEWS:
                            figure, (axis, ratio_axis) = plt.subplots(
                                2, 1, figsize=plot_style.RATIO_FIGSIZE, sharex=True, layout="constrained",
                                gridspec_kw={"height_ratios": plot_style.RATIO_HEIGHT_RATIOS})
                            figure.get_layout_engine().set(hspace=0.0, h_pad=0.01)
                            visible_values = [total-errors, total+errors]
                            populated = [total, errors]
                            process_entries = []
                            if include_reference_stacks:
                                visible_values.extend([stack["positive"], stack["negative"]])
                                for result in processes:
                                    name = result.spec.name
                                    label = plot_style.paper_label(result.spec.label)
                                    handle = plot_style.stacked_fill(axis, edges, stack["values"][name],
                                                                     stack["bottoms"][name], result.spec.color, label)
                                    process_entries.append((handle, label))
                                    populated.append(stack["values"][name])
                            band = plot_style.uncertainty_band(axis, edges, total-errors, total+errors)
                            axis.stairs(total, edges, color=plot_style.REFERENCE_LINE, linewidth=1.1, zorder=6)
                            reference_entries = [((band, plot_style.reference_handle()),
                                                  f"{reference_label} total, {'expected stat.' if data else 'MC stat.'}")]
                            variation_entries, ratio_points = [], []
                            for index, source in enumerate(sources[1:]):
                                scenario = source.scenario.identifier
                                statistics = predictions[(strategy, channel, scenario)][key]
                                if statistics.get("A") is None:
                                    continue
                                values = _array(statistics["A"])
                                variation_covariance = statistics["data_covariances"].get(str(float(luminosity))) if data else statistics["mc_covariance"]
                                variation_errors = _errors(variation_covariance, len(values))
                                visible_values.extend([values-variation_errors, values+variation_errors])
                                populated.append(values)
                                shift = (index - 0.5 * (len(sources)-2)) * min(0.12, 0.55/(len(sources)-1))
                                positions = centers + shift * widths
                                color = source.scenario.color or ("#D05A47", "#0D8274", "#8A5FBF")[index % 3]
                                marker = plot_style.scenario_marker(scenario, index + 1)
                                container = plot_style.points(axis, positions, values, variation_errors,
                                                              color=color, marker=marker, zorder=10)
                                variation_entries.append((container, scenario_labels[index + 1]))
                                ratio = ratios[(strategy, channel, key, scenario)]
                                validity = (ratio["numerical_valid"]["data"][str(float(luminosity))]
                                            if data else ratio["numerical_valid"]["mc"])
                                valid = np.asarray(validity, dtype=bool)
                                central = _array(ratio["central"])
                                ratio_covariance = ratio["data_covariances"].get(str(float(luminosity))) if data else ratio["mc_covariance"]
                                ratio_errors = _errors(ratio_covariance, len(central))
                                valid &= np.isfinite(central) & np.isfinite(ratio_errors)
                                plot_style.points(ratio_axis, positions[valid], central[valid], ratio_errors[valid],
                                                  color=color, marker=marker, zorder=10)
                                ratio_points.append((positions, central, ratio_errors, valid, color))
                            window = _view_window(view, edges, populated)
                            _window_limits(axis, centers, window, visible_values)
                            ratio_axis.axhline(1.0, color=plot_style.GUIDE, linewidth=0.7)
                            ratio_axis.set_ylabel(f"Ratio to\n{reference_label}")
                            ratio_axis.set_xlabel(OBSERVABLE_LABELS[key][1])
                            in_window = (centers >= window[0]) & (centers <= window[1])
                            ratio_range = plot_style.ratio_window(
                                [(central[valid & in_window], ratio_errors[valid & in_window])
                                 for _, central, ratio_errors, valid, _ in ratio_points])
                            if ratio_range is not None:
                                ratio_axis.set_ylim(*ratio_range)
                            for positions, central, _, valid, color in ratio_points:
                                plot_style.mark_clipped(ratio_axis, positions[valid & in_window],
                                                        central[valid & in_window], color)
                            _decorate(axis, key, window, ratio=True)
                            plot_style.legend_above(
                                axis, [reference_entries + process_entries[::-1], variation_entries],
                                title=_header(channel, strategy, cuts, float(luminosity) if data else None,
                                              label_prefix, particle_pt_min_gev, partial))
                            figure.align_ylabels((axis, ratio_axis))
                            kind = ("asymmetry-reference-stack-plus-total-ratio" if include_reference_stacks
                                    else "asymmetry-total-ratio") + ("-data-stat" if data else "")
                            relative = (Path("plots") / "comparison" / strategy / "asymmetries" /
                                        f"{luminosity:g}fb" / channel / uncertainty / view / key)
                            records.append({"strategy": strategy, "observable": key,
                                            "title": f"{OBSERVABLE_LABELS[key][0]} · {'reference process stack + scenario totals' if include_reference_stacks else 'scenario totals'} · {uncertainty} · {view}",
                                            "channel": channel, "stage": "comparison", "kind": kind,
                                            "view": view, "sample": None, "uncertainty": uncertainty,
                                            "error_mode": "data" if data else "mc", "luminosity_fb": float(luminosity),
                                            **_save(figure, run_dir, relative, particle_pt_min_gev)})
                            plt.close(figure)
    payload["plots"] = records
    write_artifacts(run_dir, payload)
    return records
