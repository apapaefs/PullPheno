"""Shared publication style for PullPheno figures.

Figures are drawn at the printed width of one elsarticle 5p column
(252 pt = 3.5 in) with Times-like STIX text, so they can be included at
``\\columnwidth`` without rescaling their fonts.  Run context (channel,
acceptance, luminosity and selections) is written as a compact header above
the legend instead of a figure title; captions belong to the manuscript.
Only presentation is defined here; no plotted number depends on this module.
"""
from __future__ import annotations

import functools
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

COLUMN_WIDTH_IN = 3.5
SINGLE_FIGSIZE = (COLUMN_WIDTH_IN, 3.2)
RATIO_FIGSIZE = (COLUMN_WIDTH_IN, 4.1)
RATIO_HEIGHT_RATIOS = (2.6, 1.0)
PNG_DPI = 300

INK = "#172033"
MUTED = "#4d5668"
GUIDE = "#6b7385"
BAND_COLOR = "#2b303a"
REFERENCE_LINE = "#111418"
PARTIAL_COLOR = "#b3261e"
FILL_LIGHTEN = 0.42

MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
# Fixed generator markers survive when other scenarios are omitted from a plot.
SCENARIO_MARKERS = {
    "herwig-default": "o", "herwig-preco025": "s", "mg5-pythia8": "^",
    "mg5-pythia8-dipolerecoil": "^",
}
MARKER_SIZE = 3.6
MARKER_EDGE_WIDTH = 0.9
ERROR_LINE_WIDTH = 0.8
CAP_SIZE = 1.4

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "STIX Two Text", "TeX Gyre Termes", "Nimbus Roman",
                   "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "legend.title_fontsize": 7.0,
    "figure.figsize": SINGLE_FIGSIZE,
    "figure.dpi": 100,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.labelpad": 2.5,
    "axes.formatter.use_mathtext": True,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.size": 1.8,
    "ytick.minor.size": 1.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.pad": 2.5,
    "ytick.major.pad": 2.5,
    "lines.linewidth": 1.0,
    "hatch.linewidth": 0.45,
    "axes.grid": False,
    "savefig.dpi": PNG_DPI,
    "savefig.facecolor": "white",
    "savefig.pad_inches": 0.03,
    # TrueType embedding keeps PDF text editable and avoids Type 3 fonts.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# Process-global defaults that the unrestyled diagnostic plots were drawn with.
LEGACY_RC_PARAMS = {
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
    "figure.figsize": (7.2, 5.2),
    "savefig.facecolor": "white",
}

CHANNEL_LABELS = {"z": r"$Zjj$, $Z\to\ell\ell$", "higgs": r"$Hjj$, $H\to\gamma\gamma$"}

_LABEL_TOKENS = (
    (re.compile(r"\bpRecoB\s*=\s*([0-9.]+)"), r"$p_{\\mathrm{reco}}^{\\mathrm{B}}=\1$"),
    (re.compile(r"\bpReco\s*=\s*([0-9.]+)"), r"$p_{\\mathrm{reco}}=\1$"),
    (re.compile(r"\b([ZWH])jj\b"), r"$\1jj$"),
    (re.compile(r"\baajj\b"), r"$\\gamma\\gamma jj$"),
)


def pyplot() -> Any:
    cache = Path(tempfile.gettempdir()) / "pullpheno-matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def paper_figures(function: Callable[..., Any]) -> Callable[..., Any]:
    """Draw every figure of ``function`` with the publication rc settings.

    The settings are scoped, so later plotting code in the same process keeps
    its own defaults.
    """
    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        plt = pyplot()
        with plt.rc_context(RC_PARAMS):
            return function(*args, **kwargs)

    return wrapper


def lighten(color: Any, amount: float = FILL_LIGHTEN) -> Tuple[float, float, float]:
    """Blend a process colour towards white so overlaid markers stay legible."""
    from matplotlib.colors import to_rgb

    rgb = np.asarray(to_rgb(color), dtype=np.float64)
    return tuple(float(value) for value in rgb + (1.0 - rgb) * amount)


def scenario_marker(identifier: str, index: int) -> str:
    return SCENARIO_MARKERS.get(identifier, MARKERS[index % len(MARKERS)])


def paper_label(text: Optional[str]) -> str:
    """Typeset known parameter and process tokens; labels with math are kept."""
    text = "" if text is None else str(text)
    if "$" in text:
        return text
    for pattern, replacement in _LABEL_TOKENS:
        text = pattern.sub(replacement, text)
    return text


def split_common_prefix(labels: Sequence[str]) -> Tuple[str, List[str]]:
    """Move leading words shared by every label (e.g. a generator) to a prefix."""
    words = [str(label).split() for label in labels]
    if len(words) < 2 or any(not item for item in words):
        return "", [str(label) for label in labels]
    shared = 0
    for column in zip(*words):
        if len(set(column)) != 1:
            break
        shared += 1
    shared = min(shared, min(len(item) for item in words) - 1)
    if shared <= 0:
        return "", [str(label) for label in labels]
    return " ".join(words[0][:shared]), [" ".join(item[shared:]) for item in words]


def scenario_labels(labels: Sequence[str]) -> Tuple[str, List[str]]:
    """Shared generator prefix for headers and short typeset scenario labels."""
    prefix, short = split_common_prefix(labels)
    return prefix, [paper_label(label) for label in short]


def acceptance_label(cuts: Optional[Mapping[str, float]]) -> Optional[str]:
    if not cuts:
        return None
    if "jet_abs_eta_max" in cuts:
        return rf"$|\eta_j|<{cuts['jet_abs_eta_max']:g}$"
    if "jet_abs_y_max" in cuts:
        return rf"$|y_j|<{cuts['jet_abs_y_max']:g}$"
    return None


def luminosity_label(luminosity: Optional[float]) -> Optional[str]:
    if luminosity is None:
        return None
    return rf"${float(luminosity):g}\,\mathrm{{fb}}^{{-1}}$"


def pull_cut_label(threshold: Optional[float]) -> str:
    return rf"$|\vec t\,|>{threshold:g}$" if threshold is not None else r"no $|\vec t\,|$ cut"


def header(
    *,
    prefix: Optional[str] = None,
    channel: Optional[str] = None,
    cuts: Optional[Mapping[str, float]] = None,
    luminosity: Optional[float] = None,
    threshold: Optional[float] = None,
    show_threshold: bool = True,
    particle_pt_min_gev: Optional[float] = None,
    notes: Sequence[str] = (),
    partial: bool = False,
) -> str:
    """Two compact lines: sample and acceptance, then selections and notes."""
    first = [
        paper_label(prefix) if prefix else None,
        CHANNEL_LABELS.get(channel, channel) if channel else None,
        acceptance_label(cuts),
        luminosity_label(luminosity),
    ]
    second = [pull_cut_label(threshold) if show_threshold else None]
    if particle_pt_min_gev is not None:
        second.append(rf"particle $p_T>{particle_pt_min_gev:g}$ GeV")
    second.extend(notes)
    if partial:
        second.append("PARTIAL RUN")
    lines = [", ".join(item for item in group if item) for group in (first, second)]
    return "\n".join(line for line in lines if line)


def _blank_handle() -> Any:
    from matplotlib.patches import Rectangle

    return Rectangle((0, 0), 1, 1, fill=False, edgecolor="none", visible=False)


def legend_above(
    axis: Any,
    groups: Sequence[Sequence[Tuple[Any, str]]],
    *,
    title: str = "",
    max_columns: int = 2,
) -> Any:
    """Place the header and legend above ``axis`` so they never cover data.

    Each group fills its own column when the columns fit the figure width;
    otherwise every entry is listed in one column.
    """
    from matplotlib.legend_handler import HandlerTuple

    groups = [list(group) for group in groups if group]
    figure = axis.figure
    columns = min(max_columns, len(groups)) if len(groups) > 1 else (
        min(max_columns, 2) if groups and len(groups[0]) > 3 else 1
    )
    legend = None
    while True:
        if len(groups) > 1 and columns == len(groups):
            rows = max(len(group) for group in groups)
            entries = [entry for group in groups
                       for entry in group + [(_blank_handle(), "")] * (rows - len(group))]
        else:
            entries = [entry for group in groups for entry in group]
        if legend is not None:
            legend.remove()
        legend = axis.legend(
            [handle for handle, _ in entries], [label for _, label in entries],
            loc="lower left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0,
            ncol=max(1, columns), frameon=False, title=title or None, alignment="left",
            handlelength=1.6, handleheight=0.8, handletextpad=0.5, columnspacing=1.0,
            labelspacing=0.28, borderpad=0.0,
            handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
        )
        legend.get_title().set_color(MUTED)
        legend.get_title().set_linespacing(1.15)
        if "PARTIAL RUN" in (title or ""):
            legend.get_title().set_color(PARTIAL_COLOR)
        if columns <= 1:
            break
        # Laid-out axes span about 86% of a column-wide figure; drawing first to
        # measure them would collapse the layout around an oversized legend.
        renderer = figure.canvas.get_renderer()
        if legend.get_window_extent(renderer).width <= 0.86 * figure.bbox.width:
            break
        columns = 1
    legend.set_zorder(20)
    return legend


def stacked_fill(
    axis: Any, edges: np.ndarray, values: np.ndarray, bottom: np.ndarray, color: Any, label: str,
) -> Any:
    """Draw one stack layer as a light bar fill with a step outline in its colour."""
    widths = np.diff(edges)
    axis.bar(edges[:-1], values, width=widths, align="edge", bottom=bottom,
             color=lighten(color), edgecolor="none", linewidth=0.0, label=label, zorder=2)
    axis.stairs(bottom + values, edges, baseline=None, color=color, linewidth=0.7, zorder=3)
    from matplotlib.patches import Patch

    return Patch(facecolor=lighten(color), edgecolor=color, linewidth=0.7)


def uncertainty_band(axis: Any, edges: np.ndarray, low: np.ndarray, high: np.ndarray,
                     label: Optional[str] = None, zorder: float = 5) -> Any:
    """Hatched band for a reference uncertainty, legible over light fills."""
    axis.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post",
                      facecolor="none", edgecolor=BAND_COLOR, hatch="//////", linewidth=0.0,
                      label=label, zorder=zorder)
    from matplotlib.patches import Patch

    return Patch(facecolor="none", edgecolor=BAND_COLOR, hatch="//////", linewidth=0.0)


def reference_handle(color: Any = REFERENCE_LINE) -> Any:
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, linewidth=1.1)


def points(axis: Any, x: np.ndarray, y: np.ndarray, errors: Optional[np.ndarray], *, color: Any,
           marker: str, filled: bool = False, label: Optional[str] = None, zorder: float = 12,
           scale: float = 1.0) -> Any:
    """Open (variation) or filled (reference) markers with thin error bars."""
    return axis.errorbar(
        x, y, yerr=errors, fmt=marker, linestyle="none", color=color,
        markersize=MARKER_SIZE * scale, markerfacecolor=color if filled else "white",
        markeredgecolor=color, markeredgewidth=MARKER_EDGE_WIDTH, elinewidth=ERROR_LINE_WIDTH,
        capsize=CAP_SIZE, capthick=ERROR_LINE_WIDTH, label=label, zorder=zorder,
    )


def ratio_window(
    pairs: Sequence[Tuple[np.ndarray, np.ndarray]], *, minimum: float = 0.05,
    precision: float = 0.25, margin: float = 1.15,
) -> Optional[Tuple[float, float]]:
    """Symmetric ratio range about one set by the informative points.

    Points whose ratio error exceeds ``precision`` (e.g. sparsely populated
    tails) do not widen the range; ``mark_clipped`` flags them instead.
    """
    deviations, fallback = [], []
    for ratios, errors in pairs:
        ratios, errors = np.asarray(ratios, dtype=np.float64), np.asarray(errors, dtype=np.float64)
        finite = np.isfinite(ratios) & np.isfinite(errors)
        spread = np.maximum(np.abs(ratios - errors - 1.0), np.abs(ratios + errors - 1.0))
        fallback.extend(spread[finite].tolist())
        deviations.extend(spread[finite & (errors <= precision)].tolist())
    values = deviations or fallback
    if not values:
        return None
    deviation = max(minimum, margin * max(values))
    return 1.0 - deviation, 1.0 + deviation


def mark_clipped(axis: Any, x: np.ndarray, y: np.ndarray, color: Any) -> None:
    """Arrows at the panel edge for central values outside the ratio range."""
    low, high = axis.get_ylim()
    span = high - low
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    for position, value in zip(x, y):
        if not np.isfinite(value) or low <= value <= high:
            continue
        tip = high - 0.03 * span if value > high else low + 0.03 * span
        tail = tip - 0.22 * span if value > high else tip + 0.22 * span
        axis.annotate("", xy=(position, tip), xytext=(position, tail),
                      arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 0.8,
                                  "mutation_scale": 5, "shrinkA": 0, "shrinkB": 0},
                      zorder=13, annotation_clip=False)


def populated_limits(
    edges: np.ndarray, vectors: Sequence[np.ndarray], *, symmetric: bool = False,
    keep_lower: bool = True, relative_floor: float = 1.0e-5,
) -> Tuple[float, float]:
    """Axis range covering the bins with visible content plus one bin, on bin edges."""
    edges = np.asarray(edges, dtype=np.float64)
    magnitude = np.zeros(len(edges) - 1)
    for vector in vectors:
        values = np.abs(np.asarray(vector, dtype=np.float64))
        magnitude = np.maximum(magnitude, np.where(np.isfinite(values), values, 0.0))
    peak = float(magnitude.max()) if magnitude.size else 0.0
    occupied = np.flatnonzero(magnitude > relative_floor * peak) if peak > 0.0 else []
    if not len(occupied):
        return float(edges[0]), float(edges[-1])
    first, last = int(occupied[0]), int(occupied[-1])
    if symmetric:
        step = float(np.max(np.diff(edges)))
        half = max(abs(edges[first]), abs(edges[last + 1])) + step
        return float(max(edges[0], -half)), float(min(edges[-1], half))
    low = edges[0] if keep_lower else edges[max(first - 1, 0)]
    return float(low), float(edges[min(last + 2, len(edges) - 1)])


def scaled_axis(axis: Any, label: str, magnitude: float, *, plain: Tuple[int, int] = (-1, 3)) -> None:
    """Quote large or small values in units of a power of ten in the y label.

    A matplotlib offset text would collide with the legend above the axes.
    """
    from matplotlib.ticker import FuncFormatter

    exponent = 0
    if np.isfinite(magnitude) and magnitude > 0.0:
        exponent = int(np.floor(np.log10(magnitude)))
        if plain[0] <= exponent <= plain[1]:
            exponent = 0
    scale = 10.0 ** exponent

    def tick(value: float, _: Any) -> str:
        text = f"{value / scale:.6g}"
        return "0" if text in ("0", "-0") else text.replace("-", "\N{MINUS SIGN}")

    axis.yaxis.set_major_formatter(FuncFormatter(tick))
    axis.set_ylabel(label + (rf" [$10^{{{exponent}}}$]" if exponent else ""))
