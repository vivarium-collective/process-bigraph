import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from process_bigraph.visualization import Visualization


def teardown_function():
    Visualization.units_resolver = None     # never leak state across tests


def test_finalize_figure_appends_unit():
    Visualization.units_resolver = lambda path: "fg" if path == "mass" else None
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass")
    ax.set_xlabel("Time (min)")
    Visualization.finalize_figure(fig, [(ax, "y", "mass"), (ax, "x", "time")])
    assert ax.get_ylabel() == "Mass (fg)"
    assert ax.get_xlabel() == "Time (min)"      # no resolver hit -> unchanged
    plt.close(fig)


def test_finalize_figure_idempotent():
    Visualization.units_resolver = lambda path: "fg"
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass (fg)")
    Visualization.finalize_figure(fig, [(ax, "y", "mass")])
    assert ax.get_ylabel() == "Mass (fg)"
    plt.close(fig)


def test_no_resolver_is_noop():
    Visualization.units_resolver = None
    fig, ax = plt.subplots()
    ax.set_ylabel("Mass")
    Visualization.finalize_figure(fig, [(ax, "y", "mass")])
    assert ax.get_ylabel() == "Mass"
    plt.close(fig)


def test_figure_to_html_returns_img_tag():
    Visualization.units_resolver = lambda path: "fg"
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1]); ax.set_ylabel("Mass")
    html = Visualization.figure_to_html(fig, [(ax, "y", "mass")])
    assert html.startswith('<img src="data:image/png;base64,')
    assert html.rstrip().endswith("/>")
