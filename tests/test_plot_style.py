"""Publication-style helpers shared by the analysis and asymmetry plots."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pullpheno-matplotlib-tests"))

import plot_style
from beam_asymmetry import ASYMMETRY_KEYS, BeamAsymmetryMoments
import beam_asymmetry_plots


class PlotStyleTests(unittest.TestCase):
    def test_labels_are_typeset_and_shared_generator_moves_to_header(self):
        prefix, labels = plot_style.scenario_labels(
            ["Herwig Nominal", "Herwig pRecoB=0.30 compensated", "Herwig pReco=0.60 MPI shift"]
        )
        self.assertEqual(prefix, "Herwig")
        self.assertEqual(labels[0], "Nominal")
        self.assertEqual(labels[1], r"$p_{\mathrm{reco}}^{\mathrm{B}}=0.30$ compensated")
        self.assertEqual(labels[2], r"$p_{\mathrm{reco}}=0.60$ MPI shift")
        self.assertEqual(plot_style.paper_label("QCD Zjj"), "QCD $Zjj$")
        self.assertEqual(plot_style.paper_label("$x$ Zjj"), "$x$ Zjj")
        self.assertEqual(plot_style.split_common_prefix(["Herwig Nominal", "MG5 + Pythia"]),
                         ("", ["Herwig Nominal", "MG5 + Pythia"]))
        self.assertEqual(plot_style.split_common_prefix(["Herwig", "Herwig"]), ("", ["Herwig", "Herwig"]))

    def test_header_states_selection_and_partial_runs(self):
        text = plot_style.header(prefix="Herwig", channel="z", cuts={"jet_abs_eta_max": 3.0},
                                 luminosity=300.0, threshold=0.0015, particle_pt_min_gev=0.1,
                                 partial=True)
        first, second = text.split("\n")
        self.assertIn(r"$|\eta_j|<3$", first)
        self.assertIn(r"$300\,\mathrm{fb}^{-1}$", first)
        self.assertIn(r"$|\vec t\,|>0.0015$", second)
        self.assertTrue(second.endswith("PARTIAL RUN"))
        self.assertIn("no", plot_style.header(threshold=None))
        self.assertEqual(plot_style.header(show_threshold=False), "")

    def test_ratio_window_ignores_imprecise_tails(self):
        window = plot_style.ratio_window([(np.array([1.01, 0.99, 12.0]), np.array([0.003, 0.004, 9.0]))])
        self.assertEqual(window, (0.95, 1.05))
        low, high = plot_style.ratio_window([(np.array([1.3, 0.8]), np.array([0.05, 0.05]))])
        self.assertLess(low, 0.75)
        self.assertGreater(high, 1.35)
        self.assertIsNone(plot_style.ratio_window([(np.array([np.nan]), np.array([np.nan]))]))

    def test_populated_limits_trim_empty_tails_on_bin_edges(self):
        edges = np.linspace(-0.03, 0.03, 41)
        centers = 0.5 * (edges[1:] + edges[:-1])
        values = np.where(np.abs(centers) < 0.016, 1.0, 0.0)
        low, high = plot_style.populated_limits(edges, [values], symmetric=True)
        self.assertAlmostEqual(low, -high)
        self.assertTrue(np.any(np.isclose(edges, high)))
        self.assertGreater(high, 0.016)
        self.assertLess(high, 0.03)
        edges = np.linspace(0.0, 0.06, 41)
        values = np.where(edges[:-1] < 0.0195, 1.0, 0.0)
        self.assertEqual(plot_style.populated_limits(edges, [values])[0], 0.0)
        self.assertEqual(plot_style.populated_limits(edges, [np.zeros(40)]), (0.0, 0.06))

    def test_large_values_are_quoted_in_label_units(self):
        plt = plot_style.pyplot()
        with plt.rc_context(plot_style.RC_PARAMS):
            figure, axis = plt.subplots()
            axis.plot([0, 1], [0, 1.2e5])
            plot_style.scaled_axis(axis, "Events / bin", 1.2e5)
            self.assertEqual(axis.get_ylabel(), r"Events / bin [$10^{5}$]")
            self.assertEqual(axis.yaxis.get_major_formatter()(5.0e4, 0), "0.5")
            plot_style.scaled_axis(axis, "Events / bin", 120.0)
            self.assertEqual(axis.get_ylabel(), "Events / bin")
            plt.close(figure)

    def test_legend_and_header_sit_above_the_axes(self):
        plt = plot_style.pyplot()
        with plt.rc_context(plot_style.RC_PARAMS):
            figure, axis = plt.subplots(figsize=plot_style.SINGLE_FIGSIZE, constrained_layout=True)
            first, = axis.plot([0, 1], [0, 1], label="a")
            second, = axis.plot([0, 1], [1, 0])
            legend = plot_style.legend_above(axis, [[(first, "Nominal")], [(second, "Variation")]],
                                             title="header line")
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            self.assertGreaterEqual(legend.get_window_extent(renderer).y0,
                                    axis.get_window_extent(renderer).y1)
            self.assertEqual(legend.get_title().get_text(), "header line")
            self.assertEqual(legend._ncols, 2)
            long_label = "a very long scenario description that cannot share a row " * 2
            legend = plot_style.legend_above(axis, [[(first, long_label)], [(second, long_label)]])
            self.assertEqual(legend._ncols, 1)
            plt.close(figure)

    def test_publication_settings_are_scoped(self):
        plt = plot_style.pyplot()
        before = plt.rcParams["font.family"]

        @plot_style.paper_figures
        def draw():
            return plt.rcParams["font.family"], plt.rcParams["pdf.fonttype"]

        self.assertEqual(draw(), (["serif"], 42))
        self.assertEqual(plt.rcParams["font.family"], before)

    def test_asymmetry_markers_match_yield_comparison_markers(self):
        def sample():
            moments = {key: BeamAsymmetryMoments(np.array([0., .001, .002, .03])) for key in ASYMMETRY_KEYS}
            for accumulator in moments.values():
                accumulator.fill([.0005, -.0015], 1.)
            return SimpleNamespace(
                spec=SimpleNamespace(name="signal", channel="z", label="EW Zjj", color="#275DAD",
                                     stack_order=0, cross_section_pb=1.0),
                strategy="cutbased", generated_sumw=1.0, pull_asymmetry_moments=moments)

        identifiers = ["nominal", "a", "b", "c", "d"]
        sources = [SimpleNamespace(scenario=SimpleNamespace(identifier=name, label=f"Herwig {name}", color="#D05A47"),
                                   results=[sample()], config=SimpleNamespace(particle_pt_min_gev=.1), metadata={})
                   for name in identifiers]
        markers = []

        def inspect(figure, run_dir, relative, particle_pt_min_gev):
            markers.append([container.lines[0].get_marker() for container in figure.axes[0].containers
                            if hasattr(container, "lines")])
            return {"png": "x.png", "pdf": "x.pdf"}

        from unittest import mock
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(beam_asymmetry_plots, "_save", side_effect=inspect):
            beam_asymmetry_plots.generate_comparison_plots(Path(temp), sources, ["cutbased"], [300.])
        expected = [plot_style.scenario_marker(name, index) for index, name in enumerate(identifiers)][1:]
        self.assertEqual(expected, ["s", "^", "D", "v"])
        self.assertTrue(markers and all(item == expected for item in markers))


if __name__ == "__main__":
    unittest.main()
