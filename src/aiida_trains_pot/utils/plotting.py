"""Plotting utilities for TrainsPotWorkChain results."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.gridspec import GridSpec

_SETS = ("training", "validation", "test")
_SET_MARKERS = {"training": "o", "validation": "s", "test": "^"}
_SET_COLORS = {"training": "#4878CF", "validation": "#EF8536", "test": "#6ABE6B"}
_SET_KEY_MAP = {"training": "TRAINING", "validation": "VALIDATION", "test": "TEST", "all": "ALL"}
_PROP_UNITS = {"e": "eV", "f": "eV/Å", "s": "eV/Å³"}

_RC = {
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.grid": True,
    "grid.color": "#DDDDDD",
    "grid.linestyle": "-",
    "grid.linewidth": 0.6,
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFAFA",
    "legend.framealpha": 0.9,
    "legend.edgecolor": "#CCCCCC",
    "xtick.direction": "out",
    "ytick.direction": "out",
}


def _find_eval_nodes(trainspot_node):
    from aiida.orm import CalcJobNode

    nodes = [
        link.node
        for link in trainspot_node.base.links.get_outgoing().all()
        if isinstance(link.node, CalcJobNode)
        and link.node.process_label == "EvaluationCalculation"
        and link.node.is_finished_ok
    ]
    return sorted(nodes, key=lambda n: n.ctime)


def _add_diagonal(ax):
    xl, yl = ax.get_xlim(), ax.get_ylim()
    lo = min(xl[0], yl[0])
    hi = max(xl[1], yl[1])
    ax.plot([lo, hi], [lo, hi], color="#888888", lw=1.0, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)


def _rmse_subtitle(ax, rmse_dict, prop_key, unit):
    committee = rmse_dict.get("ALL", {}).get("committee", {})
    val = committee.get(f"rmse_{prop_key}")
    if val is not None:
        ax.annotate(
            f"RMSE = {val:.3f} {unit}",
            xy=(0.5, 1.01),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#555555",
        )


def _display_rmse_table(cycle_idx, rmse_dict, detailed=False):
    try:
        import pandas as pd

        from IPython.display import HTML, display
    except ImportError:
        return

    set_keys = [("ALL", "all"), ("TRAINING", "training"), ("VALIDATION", "validation"), ("TEST", "test")]
    prop_keys = [("e", "E (eV)"), ("f", "F (eV/Å)"), ("s", "S (eV/Å³)")]

    display(HTML(f"<b>Cycle {cycle_idx + 1} — RMSE</b>"))

    if not detailed:
        rows = {}
        for set_upper, _ in set_keys:
            set_data = rmse_dict.get(set_upper, {})
            committee = set_data.get("committee", {})
            pot_keys = sorted(k for k in set_data if k.startswith("pot_"))
            row = {}
            for p, pk in prop_keys:
                val = committee.get(f"rmse_{p}", float("nan"))
                pot_vals = [set_data[k].get(f"rmse_{p}", float("nan")) for k in pot_keys]
                err = np.std(pot_vals) if pot_vals else float("nan")
                row[pk] = f"{val:.4f} ± {err:.4f}"
            rows[set_upper] = row
        df = pd.DataFrame(rows).T
        df.index.name = "set"
        display(df.style.highlight_min(axis=0, props="font-weight:bold"))
    else:
        first_set_data = next((rmse_dict.get(sk, {}) for sk, _ in set_keys if rmse_dict.get(sk)), {})
        model_keys = sorted(k for k in first_set_data if k.startswith("pot_")) + ["committee"]
        columns = pd.MultiIndex.from_tuples(
            [(set_upper, pk) for set_upper, _ in set_keys for _, pk in prop_keys],
            names=["set", "property"],
        )
        rows = {}
        for model in model_keys:
            row = []
            for set_upper, _ in set_keys:
                model_data = rmse_dict.get(set_upper, {}).get(model, {})
                for p, _ in prop_keys:
                    row.append(model_data.get(f"rmse_{p}", float("nan")))
            rows[model] = row
        df = pd.DataFrame(rows, index=columns).T
        df.index.name = "model"
        display(df.style.format("{:.4f}").highlight_min(axis=0, props="font-weight:bold"))


def _density_colors(x, y, bins=200):
    """Return per-point density via 2D histogram + Gaussian smoothing, mapped to [0,1]."""
    from scipy.ndimage import gaussian_filter

    counts, xedges, yedges = np.histogram2d(x, y, bins=bins)
    counts = gaussian_filter(counts.astype(float), sigma=1.5)

    # Map each point to its bin
    xi = np.clip(np.searchsorted(xedges[1:-1], x), 0, bins - 1)
    yi = np.clip(np.searchsorted(yedges[1:-1], y), 0, bins - 1)
    density = counts[xi, yi]
    return (density - density.min()) / (density.max() - density.min() + 1e-12)


def _display_fig(fig, show, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        try:
            from IPython.display import display

            display(fig)
        except ImportError:
            plt.show()
        finally:
            plt.close(fig)


def plot_parity(node, cycles=None, sets=None, detailed=False, figsize_per_cycle=(14, 4), show=False, save_path=None):
    """Plot parity plots (E, F, S) for each evaluation cycle in a TrainsPotWorkChain.

    If more than one cycle is found/selected, also plots RMSE evolution across cycles.
    After the parity figure, displays an RMSE table per cycle (committee only by default).

    Args:
        node: WorkChainNode (TrainsPotWorkChain), e.g. load_node(pk)
        cycles: list of 1-based cycle indices to plot, or None for all completed
        sets: list of dataset sets to show, subset of ("training", "validation", "test"),
              or None for all three
        detailed: if True, RMSE table shows individual potentials in addition to committee
        figsize_per_cycle: (width, height) per cycle row in the parity figure
        show: call plt.show() (default False — in notebooks figures are displayed inline)
        save_path: optional file path to save the parity figure (evolution saved alongside)

    Returns:
        (fig_parity, fig_evolution) — fig_evolution is None if only one cycle
    """
    if sets is None:
        sets = list(_SETS)
    else:
        unknown = set(sets) - set(_SETS)
        if unknown:
            raise ValueError(f"Unknown sets: {unknown}. Choose from {_SETS}.")

    eval_nodes = _find_eval_nodes(node)
    if not eval_nodes:
        raise RuntimeError(f"No completed EvaluationCalculation found as CALL child of node pk={node.pk}.")

    if cycles is not None:
        eval_nodes = [eval_nodes[i - 1] for i in cycles if 0 < i <= len(eval_nodes)]

    n_cycles = len(eval_nodes)

    all_rmse = []
    all_parity = []
    for eval_node in eval_nodes:
        outputs = eval_node.outputs
        all_parity.append(outputs.parity_data.labelled)
        all_rmse.append(outputs.rmse.labelled.get_dict())

    with mpl.rc_context(_RC):
        # ------------------------------------------------------------------ #
        # Figure 1: parity plots                                               #
        # ------------------------------------------------------------------ #
        fig_w, fig_h = figsize_per_cycle
        fig_parity = plt.figure(figsize=(fig_w, fig_h * n_cycles))
        outer_gs = GridSpec(n_cycles, 1, figure=fig_parity, hspace=0.6)

        for cycle_idx, (parity, rmse_dict) in enumerate(zip(all_parity, all_rmse, strict=False)):
            inner_gs = outer_gs[cycle_idx].subgridspec(1, 3, wspace=0.38)
            ax_e = fig_parity.add_subplot(inner_gs[0])
            ax_f = fig_parity.add_subplot(inner_gs[1])
            ax_s = fig_parity.add_subplot(inner_gs[2])

            dft_s_all = np.concatenate([parity.get_array(f"{s}_dft_s") for s in _SETS])
            stresses_available = not np.allclose(dft_s_all, 0)

            for dset in sets:
                color = _SET_COLORS[dset]
                marker = _SET_MARKERS[dset]

                # Energy — few points: medium size, no edge
                ax_e.scatter(
                    parity.get_array(f"{dset}_dft_e"),
                    parity.get_array(f"{dset}_pot_e"),
                    color=color,
                    marker=marker,
                    s=20,
                    alpha=0.85,
                    label=dset,
                    edgecolors="none",
                    zorder=3,
                )
                # Forces — many points: visible but not overlapping
                ax_f.scatter(
                    parity.get_array(f"{dset}_dft_f").ravel(),
                    parity.get_array(f"{dset}_pot_f").ravel(),
                    color=color,
                    marker=".",
                    s=20,
                    alpha=0.4,
                    label=dset,
                    edgecolors="none",
                    rasterized=True,
                )
                if stresses_available:
                    ax_s.scatter(
                        parity.get_array(f"{dset}_dft_s"),
                        parity.get_array(f"{dset}_pot_s"),
                        color=color,
                        marker=".",
                        s=20,
                        alpha=0.4,
                        label=dset,
                        edgecolors="none",
                        rasterized=True,
                    )

            if not stresses_available:
                ax_s.text(
                    0.5,
                    0.5,
                    "DFT stresses\nnot available",
                    ha="center",
                    va="center",
                    transform=ax_s.transAxes,
                    fontsize=9,
                    color="#999999",
                )
                ax_s.set_axis_off()

            cycle_label = f"Cycle {cycle_idx + 1}"
            for ax, prop, unit, prop_key in [
                (ax_e, "Energy", "eV", "e"),
                (ax_f, "Forces", "eV/Å", "f"),
                (ax_s, "Stress", "eV/Å³", "s"),
            ]:
                if not ax.axison:
                    continue
                ax.set_xlabel(f"DFT {prop} ({unit})")
                ax.set_ylabel(f"Predicted {prop} ({unit})")
                ax.set_title(f"{cycle_label}  ·  {prop}", fontweight="bold", pad=18)
                _add_diagonal(ax)
                _rmse_subtitle(ax, rmse_dict, prop_key, unit)
                ax.tick_params(labelsize=8)
                ax.set_aspect("equal", adjustable="box")

            ax_e.legend(
                title="set", fontsize=8, title_fontsize=8, markerscale=1.0, loc="upper left", bbox_to_anchor=(0.0, 1.0)
            )

        fig_parity.suptitle("Parity plots", fontsize=12, fontweight="bold", y=1.02)
        _display_fig(fig_parity, show, save_path)

    for cycle_idx, rmse_dict in enumerate(all_rmse):
        _display_rmse_table(cycle_idx, rmse_dict, detailed=detailed)

    # ------------------------------------------------------------------ #
    # Figure 2: RMSE evolution (only when more than one cycle)            #
    # ------------------------------------------------------------------ #
    fig_evolution = None
    if n_cycles > 1:
        cycle_nums = np.arange(1, n_cycles + 1)
        props = [("e", "Energy RMSE", "eV"), ("f", "Forces RMSE", "eV/Å"), ("s", "Stress RMSE", "eV/Å³")]

        with mpl.rc_context(_RC):
            fig_evolution, axes = plt.subplots(1, 3, figsize=(13, 4))
            fig_evolution.suptitle("RMSE evolution", fontsize=12, fontweight="bold", y=1.02)

            for ax, (prop_key, prop_label, unit) in zip(axes, props, strict=False):
                for dset in sets:
                    color = _SET_COLORS[dset]
                    set_key = _SET_KEY_MAP[dset]
                    rmse_vals, rmse_errs = [], []
                    for rmse in all_rmse:
                        set_data = rmse.get(set_key, {})
                        committee_val = set_data.get("committee", {}).get(f"rmse_{prop_key}", np.nan)
                        pot_vals = [
                            v.get(f"rmse_{prop_key}", np.nan) for k, v in set_data.items() if k.startswith("pot_")
                        ]
                        rmse_vals.append(committee_val)
                        rmse_errs.append(np.std(pot_vals) if pot_vals else 0.0)

                    rmse_arr = np.array(rmse_vals)
                    err_arr = np.array(rmse_errs)
                    ax.fill_between(cycle_nums, rmse_arr - err_arr, rmse_arr + err_arr, color=color, alpha=0.15)
                    ax.plot(
                        cycle_nums,
                        rmse_arr,
                        color=color,
                        marker=_SET_MARKERS[dset],
                        label=dset,
                        lw=2,
                        markersize=7,
                        markeredgecolor="white",
                        markeredgewidth=0.8,
                    )

                ax.set_xlabel("Cycle")
                ax.set_ylabel(f"RMSE ({unit})")
                ax.set_title(prop_label, fontweight="bold")
                ax.set_xticks(cycle_nums)
                ax.legend(fontsize=8)

            fig_evolution.tight_layout()
            evo_save = None
            if save_path:
                from pathlib import Path

                p = Path(save_path)
                evo_save = str(p.with_stem(p.stem + "_evolution"))
            _display_fig(fig_evolution, show, evo_save)


def plot_error_calibration(node, cycles=None, sets=None, show=False, save_path=None):
    """Plot error calibration: actual error vs committee uncertainty from parity_data.

    For each cycle, shows 3 subplots (E, F, S):
      - X axis: |DFT − predicted| (actual error)
      - Y axis: committee std (uncertainty estimate)
      - Color:  local point density via Gaussian KDE
      - Line:   linear fit y = a·x through the origin

    A well-calibrated committee lies on y = x (uncertainty ≈ error).

    Args:
        node: WorkChainNode (TrainsPotWorkChain), e.g. load_node(pk)
        cycles: list of 1-based cycle indices, or None for all completed
        sets: list of sets to include, subset of ("training", "validation", "test"),
              or None for all three
        show: call plt.show() (default False — in notebooks figures are displayed inline)
        save_path: optional file path to save the figure

    Returns:
        matplotlib Figure
    """
    from scipy.optimize import curve_fit

    if sets is None:
        sets = list(_SETS)
    else:
        unknown = set(sets) - set(_SETS)
        if unknown:
            raise ValueError(f"Unknown sets: {unknown}. Choose from {_SETS}.")

    eval_nodes = _find_eval_nodes(node)
    if not eval_nodes:
        raise RuntimeError(f"No completed EvaluationCalculation found as CALL child of node pk={node.pk}.")

    if cycles is not None:
        eval_nodes = [eval_nodes[i - 1] for i in cycles if 0 < i <= len(eval_nodes)]

    n_cycles = len(eval_nodes)

    def _linear(x, a):
        return a * x

    with mpl.rc_context(_RC):
        fig_w, fig_h = 13, 4
        fig = plt.figure(figsize=(fig_w, fig_h * n_cycles))
        outer_gs = GridSpec(n_cycles, 1, figure=fig, hspace=0.65)

        for cycle_idx, eval_node in enumerate(eval_nodes):
            parity = eval_node.outputs.parity_data.labelled
            dft_s_all = np.concatenate([parity.get_array(f"{s}_dft_s") for s in _SETS])
            stresses_available = not np.allclose(dft_s_all, 0)

            inner_gs = outer_gs[cycle_idx].subgridspec(1, 3, wspace=0.38)
            ax_e = fig.add_subplot(inner_gs[0])
            ax_f = fig.add_subplot(inner_gs[1])
            ax_s = fig.add_subplot(inner_gs[2])

            cycle_label = f"Cycle {cycle_idx + 1}"

            for prop_key, ax, unit, label in [
                ("e", ax_e, "eV", "Energy"),
                ("f", ax_f, "eV/Å", "Forces"),
                ("s", ax_s, "eV/Å³", "Stress"),
            ]:
                if prop_key == "s" and not stresses_available:
                    ax.text(
                        0.5,
                        0.5,
                        "DFT stresses\nnot available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="#999999",
                    )
                    ax.set_axis_off()
                    continue

                # Collect |error| and std across all requested sets
                err_all, std_all = [], []
                for dset in sets:
                    dft = parity.get_array(f"{dset}_dft_{prop_key}").ravel()
                    pot = parity.get_array(f"{dset}_pot_{prop_key}").ravel()
                    std = parity.get_array(f"{dset}_std_pot_{prop_key}").ravel()
                    err_all.append(np.abs(dft - pot))
                    std_all.append(std)

                err = np.concatenate(err_all)
                std = np.concatenate(std_all)

                # Remove zeros/nans to avoid KDE issues
                mask = (err > 0) & (std > 0) & np.isfinite(err) & np.isfinite(std)
                err, std = err[mask], std[mask]

                # Density color
                colors = _density_colors(err, std)
                sc = ax.scatter(err, std, c=colors, cmap="plasma", s=10, alpha=0.6, edgecolors="none", rasterized=True)
                cb = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.03, label="density")
                cb.solids.set_alpha(1)
                cb.ax.tick_params(labelsize=6)

                # Fit y = a·x
                try:
                    (a,), _ = curve_fit(_linear, err, std, p0=[1.0])
                    x_fit = np.linspace(0, err.max(), 200)
                    ax.plot(x_fit, _linear(x_fit, a), color="#CC3333", lw=1.8, ls="-", label=f"fit:  σ = {a:.2f}·|ε|")
                    # y = x reference
                    ax.plot(x_fit, x_fit, color="#888888", lw=1.0, ls="--", label="y = x")
                    ax.legend(fontsize=7.5, framealpha=0.85)
                except Exception:
                    pass

                ax.set_xlabel(f"DFT error ({unit})")
                ax.set_ylabel(f"Committee σ ({unit})")
                ax.set_title(f"{cycle_label}  ·  {label}", fontweight="bold", pad=8)
                ax.tick_params(labelsize=8)

        fig.suptitle("Error calibration", fontsize=12, fontweight="bold", y=1.02)

        cal_save = None
        if save_path:
            from pathlib import Path

            p = Path(save_path)
            cal_save = str(p.with_stem(p.stem + "_calibration"))
        _display_fig(fig, show, cal_save)
