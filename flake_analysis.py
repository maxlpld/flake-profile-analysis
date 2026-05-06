#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Only user-adjustable height-detection parameter.
# Number of consecutive decreasing points required after the edge maximum.
EDGE_DECREASE_RUN = 3

ERROR_BAR_MODE = "std"

UNIBAS = {
    "mint": "#A5D7D2",
    "mint_hell": "#D2EBE9",
    "rot": "#D20537",
    "anthrazit": "#2D373C",
    "anthrazit_hell": "#46505A",
    "black": "#000000",
}

PROFILE_COLORS = [
    UNIBAS["mint"],
    UNIBAS["rot"],
    UNIBAS["anthrazit_hell"],
    UNIBAS["anthrazit"],
    UNIBAS["mint_hell"],
]


@dataclass
class Profile:
    name: str
    x_nm: np.ndarray
    z_nm: np.ndarray


@dataclass
class HeightResult:
    profile: str
    height_left_nm: float
    height_right_nm: float
    height_left_err_nm: float = 0.0
    height_right_err_nm: float = 0.0
    left_marker_x_nm: float = np.nan
    left_marker_z_nm: float = np.nan
    right_marker_x_nm: float = np.nan
    right_marker_z_nm: float = np.nan


@dataclass
class StripeResult:
    profile: str
    mean_spacing_nm: float
    std_spacing_nm: float
    n_peaks: int
    n_spacings: int


@dataclass
class PairedStripeResult:
    profile: str
    forward_mean_spacing_nm: float
    forward_std_spacing_nm: float
    forward_n_peaks: int
    forward_n_spacings: int
    backward_mean_spacing_nm: float
    backward_std_spacing_nm: float
    backward_n_peaks: int
    backward_n_spacings: int


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AFM/Gwyddion line profiles across a flake.")
    parser.add_argument("--data", type=str, default=None, help="Backward-compatible alias for --forward-data.")
    parser.add_argument("--forward-data", type=str, default=None, help="Path to the forward Gwyddion profile data file.")
    parser.add_argument("--backward-data", type=str, default=None, help="Optional path to the backward Gwyddion profile data file.")
    parser.add_argument("--transparent", type=str, default=None, help="yes/no for transparent plot background.")
    parser.add_argument("--height-inset", type=str, default=None, help="yes/no: put the height plot as an inset in the profile plot.")
    parser.add_argument("--usetex", action="store_true", help="Use external LaTeX installation for text rendering.")
    return parser.parse_args()


def yes_no(value):
    return value.lower() in {"y", "yes", "true", "1"}


def setup_matplotlib(use_tex=False):
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "text.usetex": bool(use_tex),
    })


def ask(msg, default="y"):
    raw = input(msg).strip().lower()
    if raw == "":
        raw = default
    return yes_no(raw)


def ask_data_path(initial_path=None, label="forward", optional=False) -> Optional[Path]:
    while True:
        if initial_path is not None:
            raw = str(initial_path).strip()
        else:
            if optional:
                raw = input(f"Path to the {label} Gwyddion profile data file [press Enter to skip]: ").strip()
            else:
                raw = input(f"Path to the {label} Gwyddion profile data file: ").strip()

        if optional and raw == "":
            print(f"No {label} data file selected. The analysis will use only the available forward data.\n")
            return None

        path = Path(raw.strip('"').strip("'")).expanduser().resolve()
        print(f"\nSelected {label} file:\n{path}")

        if not path.exists():
            print("This file does not exist. Please enter another path.\n")
            initial_path = None
            continue

        if ask("Use this file? [Y/n]: "):
            return path

        initial_path = None


def ask_transparent(arg_value=None) -> bool:
    if arg_value is not None:
        return yes_no(arg_value)
    return ask("Transparent background? [Y/n]: ")


def ask_height_inset(arg_value=None) -> bool:
    if arg_value is not None:
        return yes_no(arg_value)
    return ask("Put the apparent-height plot as an inset in the mean-profile plot? [Y/n]: ")


def sniff_delimiter(first_line):
    if ";" in first_line:
        return ";"
    if "\t" in first_line:
        return "\t"
    return ","


def unit_to_nm_factor(unit):
    u = unit.strip().lower().replace("[", "").replace("]", "")
    if u in {"m", "meter", "metre", "meters", "metres"}:
        return 1e9
    if u in {"nm", "nanometer", "nanometre", "nanometers", "nanometres"}:
        return 1.0
    if u in {"um", "µm", "μm", "micrometer", "micrometre", "micrometers", "micrometres"}:
        return 1e3
    if u in {"a", "å", "angstrom", "angstroem", "ångström", "angstroms"}:
        return 0.1
    return 1.0


def parse_float(value):
    try:
        return float(value.strip().replace(",", "."))
    except Exception:
        return None


def read_profiles(path):
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(text) < 4:
        raise ValueError("The file is too short. Expected a Gwyddion profile text export.")

    delimiter = sniff_delimiter(text[0])
    rows = list(csv.reader(text, delimiter=delimiter))

    names = rows[0]
    axes = rows[1]
    units = rows[2]

    profiles = []

    for col in range(0, len(axes)-1, 2):
        x_values = []
        z_values = []

        x_factor = unit_to_nm_factor(units[col])
        z_factor = unit_to_nm_factor(units[col+1])

        for row in rows[3:]:
            if len(row) <= col+1:
                continue

            x = parse_float(row[col])
            z = parse_float(row[col+1])

            if x is None or z is None:
                continue

            x_values.append(x * x_factor)
            z_values.append(z * z_factor)

        if len(x_values) > 5:
            x = np.asarray(x_values, dtype=float)
            z = np.asarray(z_values, dtype=float)

            order = np.argsort(x)
            x = x[order]
            z = z[order]
            x = x - x[0]

            profiles.append(Profile(
                name=f"Profile {len(profiles)+1}",
                x_nm=x,
                z_nm=z
            ))

    if not profiles:
        raise ValueError("No profiles found. Check the Gwyddion export format.")

    return profiles


def moving_average(y, window):
    if window < 3:
        return y.copy()

    if window % 2 == 0:
        window += 1

    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 3:
        return y.copy()

    pad = window // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window

    return np.convolve(yp, kernel, mode="valid")


def detect_edges(profile):
    """Threshold-based flake region detection used only for stripe spacing."""
    z = profile.z_nm
    n = len(z)

    window = max(5, int(0.05 * n))
    z_s = moving_average(z, window)

    low = np.percentile(z_s, 15)
    high = np.percentile(z_s, 85)
    threshold = low + 0.30 * (high - low)

    mask = z_s > threshold
    idx = np.where(mask)[0]

    if len(idx) < 2:
        raise RuntimeError(f"Could not detect the flake plateau for {profile.name}.")

    return idx[0], idx[-1]


def detect_edge_height_from_endpoint(profile, side):
    """Detect one edge height from one endpoint using the requested rule.

    Left edge:
        start from the first available point and move to the right. The selected
        point is the last point reached while the profile was increasing before
        EDGE_DECREASE_RUN consecutive decreases are observed.

    Right edge:
        start from the last available point and move to the left. The same rule
        is applied in the reversed scan direction.

    The height is |z(selected point) - z(endpoint)| and the marker is placed on
    the selected point used for this height measurement.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    x = profile.x_nm
    z = profile.z_nm
    finite = np.isfinite(x) & np.isfinite(z)
    idx = np.where(finite)[0]

    if len(idx) < EDGE_DECREASE_RUN + 2:
        return np.nan, np.nan, np.nan

    if side == "right":
        idx = idx[::-1]

    base_idx = int(idx[0])
    z_scan = z[idx]

    last_increasing_pos = None
    selected_pos = None

    for pos in range(1, len(idx)):
        if z_scan[pos] > z_scan[pos - 1]:
            last_increasing_pos = pos

        if last_increasing_pos is None:
            continue

        j = last_increasing_pos
        if j + EDGE_DECREASE_RUN < len(idx):
            after_max = z_scan[j:j + EDGE_DECREASE_RUN + 1]
            if np.all(np.diff(after_max) < 0):
                selected_pos = j
                break

    if selected_pos is None:
        # Fallback: if no complete decreasing run is found, use the last point
        # that was part of the initial increase.  This avoids jumping to the
        # opposite edge of the flake.
        if last_increasing_pos is not None:
            selected_pos = last_increasing_pos
        else:
            selected_pos = int(np.nanargmax(z_scan))

    max_idx = int(idx[selected_pos])
    height = abs(float(z[max_idx]) - float(z[base_idx]))

    return height, float(x[max_idx]), float(z[max_idx])


def analyze_height(profile):
    left_h, left_marker_x, left_marker_z = detect_edge_height_from_endpoint(profile, side="left")
    right_h, right_marker_x, right_marker_z = detect_edge_height_from_endpoint(profile, side="right")

    return HeightResult(
        profile=profile.name,
        height_left_nm=left_h,
        height_right_nm=right_h,
        left_marker_x_nm=left_marker_x,
        left_marker_z_nm=left_marker_z,
        right_marker_x_nm=right_marker_x,
        right_marker_z_nm=right_marker_z,
    )


def analyze_stripes(profile):
    left_idx, right_idx = detect_edges(profile)

    x = profile.x_nm[left_idx:right_idx + 1]
    z = profile.z_nm[left_idx:right_idx + 1]

    if len(z) < 10:
        return StripeResult(profile.name, np.nan, np.nan, 0, 0)

    # Remove the slow background from the flake top before detecting local maxima.
    window = max(7, int(0.10 * len(z)))
    z_slow = moving_average(z, window)
    z_centered = z - z_slow

    peaks = []
    for i in range(1, len(z_centered)-1):
        if z_centered[i] > z_centered[i-1] and z_centered[i] > z_centered[i+1]:
            peaks.append(i)

    if len(peaks) < 2:
        return StripeResult(profile.name, np.nan, np.nan, len(peaks), 0)

    distances = np.diff(x[peaks])

    return StripeResult(
        profile=profile.name,
        mean_spacing_nm=float(np.nanmean(distances)),
        std_spacing_nm=float(np.nanstd(distances, ddof=1)) if len(distances) > 1 else 0.0,
        n_peaks=len(peaks),
        n_spacings=len(distances)
    )


def common_mean_profile(profiles):
    # Use the full union of the x-ranges, not only the common overlap.
    # Outside the range of a profile, the interpolation is set to NaN, so the mean
    # is computed from all profiles that actually contain data at that x-position.
    x_min = min(float(p.x_nm[0]) for p in profiles)
    x_max = max(float(p.x_nm[-1]) for p in profiles)

    n_grid = max(700, max(len(p.x_nm) for p in profiles))
    x_grid = np.linspace(x_min, x_max, n_grid)

    curves = []
    for p in profiles:
        curves.append(np.interp(x_grid, p.x_nm, p.z_nm, left=np.nan, right=np.nan))

    z_mean = np.nanmean(np.vstack(curves), axis=0)

    return x_grid, z_mean


def combine_height_results(forward_profiles, backward_profiles=None):
    forward_results = [analyze_height(p) for p in forward_profiles]
    backward_results = [analyze_height(p) for p in backward_profiles] if backward_profiles else []

    n_results = max(len(forward_results), len(backward_results))
    combined = []

    def mean_and_error(values):
        values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        if len(values) == 0:
            return np.nan, 0.0
        mean = float(np.nanmean(values))
        err = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        return mean, err

    for i in range(n_results):
        if backward_results:
            left_values = []
            right_values = []

            if i < len(forward_results):
                left_values.append(forward_results[i].height_left_nm)
                right_values.append(forward_results[i].height_right_nm)

            if i < len(backward_results):
                left_values.append(backward_results[i].height_left_nm)
                right_values.append(backward_results[i].height_right_nm)

            left_height, left_err = mean_and_error(left_values)
            right_height, right_err = mean_and_error(right_values)
        else:
            left_height = forward_results[i].height_left_nm if i < len(forward_results) else np.nan
            right_height = np.nan
            left_err = 0.0
            right_err = 0.0

        combined.append(HeightResult(
            profile=f"Profile {i+1}",
            height_left_nm=left_height,
            height_right_nm=right_height,
            height_left_err_nm=left_err,
            height_right_err_nm=right_err
        ))

    return combined


def combine_stripe_results(forward_profiles, backward_profiles=None):
    forward_results = [analyze_stripes(p) for p in forward_profiles]
    backward_results = [analyze_stripes(p) for p in backward_profiles] if backward_profiles else []

    n_results = max(len(forward_results), len(backward_results))
    combined = []

    for i in range(n_results):
        if i < len(forward_results):
            f = forward_results[i]
        else:
            f = StripeResult(f"Profile {i+1}", np.nan, np.nan, 0, 0)

        if i < len(backward_results):
            b = backward_results[i]
        else:
            b = StripeResult(f"Profile {i+1}", np.nan, np.nan, 0, 0)

        combined.append(PairedStripeResult(
            profile=f"Profile {i+1}",
            forward_mean_spacing_nm=f.mean_spacing_nm,
            forward_std_spacing_nm=f.std_spacing_nm,
            forward_n_peaks=f.n_peaks,
            forward_n_spacings=f.n_spacings,
            backward_mean_spacing_nm=b.mean_spacing_nm,
            backward_std_spacing_nm=b.std_spacing_nm,
            backward_n_peaks=b.n_peaks,
            backward_n_spacings=b.n_spacings,
        ))

    return combined


def height_stats(results):
    values = np.array(
        [v for r in results for v in [r.height_left_nm, r.height_right_nm] if np.isfinite(v)],
        dtype=float
    )
    mean_h = float(np.nanmean(values)) if len(values) else np.nan
    std_h = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    err_h = std_h
    return mean_h, err_h, std_h, values


def linear_fit_with_errors(x_values, y_values):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(y) < 2 or len(np.unique(x)) < 2:
        return np.nan, np.nan, np.nan, np.nan

    coeffs = np.polyfit(x, y, 1)
    slope, intercept = float(coeffs[0]), float(coeffs[1])

    if len(y) <= 2:
        return slope, intercept, 0.0, 0.0

    design = np.vstack([x, np.ones_like(x)]).T
    residuals = y - (slope * x + intercept)
    dof = len(y) - 2

    try:
        residual_variance = float(np.sum(residuals**2) / dof)
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        slope_err = float(np.sqrt(covariance[0, 0])) if covariance[0, 0] >= 0 else np.nan
        intercept_err = float(np.sqrt(covariance[1, 1])) if covariance[1, 1] >= 0 else np.nan
    except np.linalg.LinAlgError:
        slope_err = np.nan
        intercept_err = np.nan

    return slope, intercept, slope_err, intercept_err


def side_fit_values(results, side):
    fit_x_values = []
    fit_y_values = []

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'")

    for i, r in enumerate(results, start=1):
        value = r.height_left_nm if side == "left" else r.height_right_nm
        if np.isfinite(value):
            fit_x_values.append(float(i))
            fit_y_values.append(float(value))

    return np.asarray(fit_x_values, dtype=float), np.asarray(fit_y_values, dtype=float)


def side_fit_with_errors(results, side):
    fit_x_values, fit_y_values = side_fit_values(results, side=side)
    return linear_fit_with_errors(fit_x_values, fit_y_values)


def draw_height_panel(ax, results, show_profile_legend=False, title=None, outside_text=False, outside_profile_legend=False, compact=False):
    handles = []
    left_dx = -0.08
    right_dx = 0.08

    left_fit_x_values = []
    left_fit_y_values = []
    right_fit_x_values = []
    right_fit_y_values = []

    for i, r in enumerate(results, start=1):
        color = PROFILE_COLORS[(i-1) % len(PROFILE_COLORS)]

        if np.isfinite(r.height_left_nm):
            ax.errorbar(
                i + left_dx,
                r.height_left_nm,
                yerr=r.height_left_err_nm,
                fmt="o",
                color=color,
                capsize=3 if compact else 4,
                markersize=5 if compact else 7
            )
            left_fit_x_values.append(float(i))
            left_fit_y_values.append(float(r.height_left_nm))

        if np.isfinite(r.height_right_nm):
            ax.errorbar(
                i + right_dx,
                r.height_right_nm,
                yerr=r.height_right_err_nm,
                fmt="^",
                color=color,
                capsize=3 if compact else 4,
                markersize=5 if compact else 7
            )
            right_fit_x_values.append(float(i))
            right_fit_y_values.append(float(r.height_right_nm))

        if np.isfinite(r.height_left_nm) and np.isfinite(r.height_right_nm):
            ax.plot(
                [i + left_dx, i + right_dx],
                [r.height_left_nm, r.height_right_nm],
                color=color,
                alpha=0.7,
                lw=1.0 if compact else 1.2,
            )

        handles.append(Line2D([0], [0], color=color, lw=1.5, label=r.profile))

    left_slope, left_intercept, left_slope_err, left_intercept_err = linear_fit_with_errors(
        np.asarray(left_fit_x_values, dtype=float), np.asarray(left_fit_y_values, dtype=float)
    )
    right_slope, right_intercept, right_slope_err, right_intercept_err = linear_fit_with_errors(
        np.asarray(right_fit_x_values, dtype=float), np.asarray(right_fit_y_values, dtype=float)
    )

    fit_handles = []
    text_lines = []
    fit_x = np.linspace(1, len(results), 200)

    if np.isfinite(left_slope) and np.isfinite(left_intercept):
        ax.plot(fit_x, left_slope * fit_x + left_intercept, color="black", lw=1.5 if compact else 2, linestyle="-")
        fit_handles.append(Line2D([0], [0], color="black", lw=2, linestyle="-", label=r"Left fit"))
        text_lines.extend([
            rf"$h_\mathrm{{left}}(i)=a_L i+b_L$",
            rf"$a_L={left_slope:.4f} \pm {left_slope_err:.4f}\,\mathrm{{nm}}$",
            rf"$b_L={left_intercept:.4f} \pm {left_intercept_err:.4f}\,\mathrm{{nm}}$",
        ])

    if np.isfinite(right_slope) and np.isfinite(right_intercept):
        ax.plot(fit_x, right_slope * fit_x + right_intercept, color="black", lw=1.5 if compact else 2, linestyle="--")
        fit_handles.append(Line2D([0], [0], color="black", lw=2, linestyle="--", label=r"Right fit"))
        if text_lines:
            text_lines.append("")
        text_lines.extend([
            rf"$h_\mathrm{{right}}(i)=a_R i+b_R$",
            rf"$a_R={right_slope:.4f} \pm {right_slope_err:.4f}\,\mathrm{{nm}}$",
            rf"$b_R={right_intercept:.4f} \pm {right_intercept_err:.4f}\,\mathrm{{nm}}$",
        ])

    ax.set_xlabel(r"Profile index", fontsize=8 if compact else None)
    ax.set_ylabel(r"Apparent height $h$ (nm)", fontsize=8 if compact else None)
    ax.set_xticks(list(range(1, len(results)+1)))
    ax.set_xticklabels([str(i) for i in range(1, len(results)+1)], fontsize=7 if compact else None)
    ax.set_xlim(0.5, len(results) + 0.5)
    ax.tick_params(axis="y", labelsize=7 if compact else None)

    if title:
        ax.set_title(title)

    ax.grid(False)

    marker_notes = [
        Line2D([0], [0], color="none", marker="o", markerfacecolor="none", markeredgecolor=UNIBAS["anthrazit"], label=r"Left edge"),
        Line2D([0], [0], color="none", marker="^", markerfacecolor="none", markeredgecolor=UNIBAS["anthrazit"], label=r"Right edge"),
    ]

    fit_legend_handles = fit_handles if compact else fit_handles + marker_notes
    if fit_legend_handles:
        # In the standalone height plot, keep the marker meaning together with
        # the fit-line meaning in the lower-left legend.  In the inset, the
        # marker meaning is already shown in the main plot legend, so the inset
        # only keeps the compact left/right-fit legend.
        fit_legend = ax.legend(
            handles=fit_legend_handles,
            frameon=False,
            fontsize=7 if compact else 8,
            loc="lower left",
        )
        ax.add_artist(fit_legend)

    if show_profile_legend:
        legend_handles = handles
        if outside_profile_legend:
            ax.legend(
                handles=legend_handles,
                frameon=False,
                fontsize=8,
                ncol=2,
                loc="lower left",
                bbox_to_anchor=(1.02, 0.02),
                borderaxespad=0.0,
            )
        else:
            ax.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=2)

    if text_lines and not compact:
        if outside_text:
            ax.text(
                1.02,
                0.98,
                "\n".join(text_lines),
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
        else:
            ax.text(
                0.97,
                0.97,
                "\n".join(text_lines),
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.75),
            )

    return (
        left_slope,
        left_intercept,
        left_slope_err,
        left_intercept_err,
        right_slope,
        right_intercept,
        right_slope_err,
        right_intercept_err,
    )


def plot_profiles(forward_profiles, backward_profiles, results, display_left_h, display_left_err_h, display_right_h, display_right_err_h, out, transparent, show_legend, title, height_inset=False):
    fig, ax = plt.subplots(figsize=(8.8, 5.2 if height_inset else 4.5))

    def add_height_markers(profile, color):
        try:
            result = analyze_height(profile)
        except Exception:
            return

        if np.isfinite(result.left_marker_x_nm) and np.isfinite(result.left_marker_z_nm):
            ax.scatter(
                result.left_marker_x_nm,
                result.left_marker_z_nm,
                marker="o",
                s=22,
                color=color,
                edgecolors="none",
                zorder=5,
            )

        if np.isfinite(result.right_marker_x_nm) and np.isfinite(result.right_marker_z_nm):
            ax.scatter(
                result.right_marker_x_nm,
                result.right_marker_z_nm,
                marker="^",
                s=28,
                color=color,
                edgecolors="none",
                zorder=5,
            )

    for i, p in enumerate(forward_profiles):
        color = PROFILE_COLORS[i % len(PROFILE_COLORS)]
        ax.plot(
            p.x_nm,
            p.z_nm,
            color=color,
            lw=1.2,
            alpha=0.7,
            linestyle="-",
            label=p.name
        )
        add_height_markers(p, color)

    if backward_profiles:
        for i, p in enumerate(backward_profiles):
            color = PROFILE_COLORS[i % len(PROFILE_COLORS)]
            ax.plot(
                p.x_nm,
                p.z_nm,
                color=color,
                lw=1.2,
                alpha=0.7,
                linestyle="--",
                label="_nolegend_"
            )
            add_height_markers(p, color)

    profiles_for_mean = list(forward_profiles) + (list(backward_profiles) if backward_profiles else [])
    x_mean, z_mean = common_mean_profile(profiles_for_mean)

    ax.plot(
        x_mean,
        z_mean,
        color="black",
        lw=2.6,
        label="Mean profile"
    )

    ax.set_xlim(
        min(float(p.x_nm[0]) for p in profiles_for_mean),
        max(float(p.x_nm[-1]) for p in profiles_for_mean)
    )

    ax.set_xlabel(r"Distance $x$ (nm)")
    ax.set_ylabel(r"Height $z$ (nm)")

    if title:
        ax.set_title(title)

    text_lines = []
    if np.isfinite(display_left_h):
        text_lines.append(rf"$h_\mathrm{{left}} = {display_left_h:.3f} \pm {display_left_err_h:.3f}\,\mathrm{{nm}}$")
    if np.isfinite(display_right_h):
        text_lines.append(rf"$h_\mathrm{{right}} = {display_right_h:.3f} \pm {display_right_err_h:.3f}\,\mathrm{{nm}}$")
    if not text_lines:
        text_lines.append(r"$h_\mathrm{left},h_\mathrm{right} = \mathrm{nan}$")

    ax.text(
        0.97,
        0.95,
        "\n".join(text_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    ax.grid(False)

    height_fit_values = (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    if height_inset:
        inset_ax = ax.inset_axes([0.23, 0.08, 0.54, 0.35])
        inset_ax.set_facecolor((1, 1, 1, 0.88))
        height_fit_values = draw_height_panel(inset_ax, results, compact=True)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()

        marker_handles = [
            Line2D([0], [0], color="none", marker="o", markerfacecolor="none",
                   markeredgecolor=UNIBAS["anthrazit"], linestyle="None", label="Left edge marker"),
            Line2D([0], [0], color="none", marker="^", markerfacecolor="none",
                   markeredgecolor=UNIBAS["anthrazit"], linestyle="None", label="Right edge marker"),
        ]
        handles = handles + marker_handles
        labels = labels + [h.get_label() for h in marker_handles]

        if backward_profiles:
            style_handles = [
                Line2D([0], [0], color=UNIBAS["anthrazit"], lw=1.5, linestyle="-", label="Forward scan"),
                Line2D([0], [0], color=UNIBAS["anthrazit"], lw=1.5, linestyle="--", label="Backward scan"),
            ]
            handles = handles + style_handles
            labels = labels + [h.get_label() for h in style_handles]

        ax.legend(
            handles=handles,
            labels=labels,
            frameon=False,
            fontsize=8,
            loc="lower left",
            bbox_to_anchor=(1.02, 0.02),
            borderaxespad=0.0,
        )

    fig.tight_layout()
    fig.savefig(out, transparent=transparent, bbox_inches="tight")
    plt.close(fig)

    return height_fit_values


def plot_heights(results, out, transparent, show_legend, title):
    fig, ax = plt.subplots(figsize=(8.9, 4.5))

    fit_values = draw_height_panel(
        ax,
        results,
        show_profile_legend=show_legend,
        title=title,
        outside_text=True,
        outside_profile_legend=True,
        compact=False,
    )

    fig.tight_layout()
    fig.savefig(out, transparent=transparent, bbox_inches="tight")
    plt.close(fig)

    return fit_values


def plot_stripes(stripe_results, out, transparent, show_legend, title, has_backward=False):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    xvals = np.arange(1, len(stripe_results)+1)
    left_dx = -0.08 if has_backward else 0.0
    right_dx = 0.08

    fit_x_values = []
    fit_y_values = []

    handles = []

    for i, r in enumerate(stripe_results):
        color = PROFILE_COLORS[i % len(PROFILE_COLORS)]
        x_index = xvals[i]

        if np.isfinite(r.forward_mean_spacing_nm):
            ax.errorbar(
                x_index + left_dx,
                r.forward_mean_spacing_nm,
                yerr=r.forward_std_spacing_nm,
                fmt='s',
                color=color,
                capsize=4,
                markersize=7
            )
            fit_x_values.append(float(x_index))
            fit_y_values.append(float(r.forward_mean_spacing_nm))

        if has_backward and np.isfinite(r.backward_mean_spacing_nm):
            ax.errorbar(
                x_index + right_dx,
                r.backward_mean_spacing_nm,
                yerr=r.backward_std_spacing_nm,
                fmt='D',
                color=color,
                capsize=4,
                markersize=7
            )
            fit_x_values.append(float(x_index))
            fit_y_values.append(float(r.backward_mean_spacing_nm))

        if has_backward and np.isfinite(r.forward_mean_spacing_nm) and np.isfinite(r.backward_mean_spacing_nm):
            ax.plot(
                [x_index + left_dx, x_index + right_dx],
                [r.forward_mean_spacing_nm, r.backward_mean_spacing_nm],
                color=color,
                alpha=0.7
            )

        handles.append(Line2D([0], [0], color=color, lw=1.5, label=r.profile))

    fit_x_values = np.asarray(fit_x_values, dtype=float)
    fit_y_values = np.asarray(fit_y_values, dtype=float)

    slope, intercept, slope_err, intercept_err = linear_fit_with_errors(fit_x_values, fit_y_values)
    fit_handle = None

    if np.isfinite(slope) and np.isfinite(intercept):
        fit_x = np.linspace(1, len(stripe_results), 200)
        fit_y = slope * fit_x + intercept

        ax.plot(fit_x, fit_y, color="black", lw=2)
        fit_handle = Line2D([0], [0], color="black", lw=2, label=r"Linear fit")

        text = (
            rf"$d(i)=a\,i+b$" + "\n" +
            rf"$a={slope:.4f} \pm {slope_err:.4f}\,\mathrm{{nm}}$" + "\n" +
            rf"$b={intercept:.4f} \pm {intercept_err:.4f}\,\mathrm{{nm}}$"
        )

        ax.text(
            0.03,
            0.95,
            text,
            transform=ax.transAxes,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
        )

    ax.set_xlabel(r"Profile index $i$")
    ax.set_ylabel(r"Stripe spacing $d$ (nm)")
    ax.set_xticks(list(range(1, len(stripe_results) + 1)))
    ax.set_xlim(0.5, len(stripe_results) + 0.5)

    if title:
        ax.set_title(title)

    ax.grid(False)

    marker_note = Line2D([0], [0], color="none", marker="s", markerfacecolor="none", markeredgecolor=UNIBAS["anthrazit"], label=r"Forward spacing")
    marker_notes = [marker_note]

    if has_backward:
        marker_note_2 = Line2D([0], [0], color="none", marker="D", markerfacecolor="none", markeredgecolor=UNIBAS["anthrazit"], label=r"Backward spacing")
        marker_notes.append(marker_note_2)

    legend_handles = handles + marker_notes if show_legend else marker_notes
    if fit_handle is not None:
        legend_handles.append(fit_handle)

    ax.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, transparent=transparent, bbox_inches="tight")
    plt.close(fig)

    return slope, intercept, slope_err, intercept_err


def save_summary_csv(path, forward_data_path, backward_data_path, results, stripe_results, mean_h, err_h, std_h, height_left_slope, height_left_intercept, height_left_slope_err, height_left_intercept_err, height_right_slope, height_right_intercept, height_right_slope_err, height_right_intercept_err, stripe_slope, stripe_intercept, stripe_slope_err, stripe_intercept_err):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Input files"])
        writer.writerow(["forward_data_file", str(forward_data_path)])
        writer.writerow(["backward_data_file", str(backward_data_path) if backward_data_path is not None else ""])

        writer.writerow([])
        writer.writerow(["Height analysis"])
        writer.writerow([
            "profile",
            "height_left_nm",
            "height_right_nm",
            "height_left_error_nm",
            "height_right_error_nm"
        ])

        for r in results:
            writer.writerow([
                r.profile,
                f"{r.height_left_nm:.8g}" if np.isfinite(r.height_left_nm) else "nan",
                f"{r.height_right_nm:.8g}" if np.isfinite(r.height_right_nm) else "nan",
                f"{r.height_left_err_nm:.8g}" if np.isfinite(r.height_left_err_nm) else "nan",
                f"{r.height_right_err_nm:.8g}" if np.isfinite(r.height_right_err_nm) else "nan"
            ])

        writer.writerow([])
        writer.writerow(["height_left_source", "endpoint_scan_left"])
        writer.writerow(["height_right_source", "endpoint_scan_right"])
        writer.writerow(["global_mean_height_nm", f"{mean_h:.8g}" if np.isfinite(mean_h) else "nan"])
        writer.writerow([f"global_error_height_nm_{ERROR_BAR_MODE}", f"{err_h:.8g}" if np.isfinite(err_h) else "nan"])
        writer.writerow(["global_std_height_nm", f"{std_h:.8g}" if np.isfinite(std_h) else "nan"])
        writer.writerow(["height_left_linear_fit_model", "h_left(i)=a_L*i+b_L"])
        writer.writerow(["height_left_linear_fit_slope_a_L_nm", f"{height_left_slope:.8g}" if np.isfinite(height_left_slope) else "nan"])
        writer.writerow(["height_left_linear_fit_slope_a_L_error_nm", f"{height_left_slope_err:.8g}" if np.isfinite(height_left_slope_err) else "nan"])
        writer.writerow(["height_left_linear_fit_intercept_b_L_nm", f"{height_left_intercept:.8g}" if np.isfinite(height_left_intercept) else "nan"])
        writer.writerow(["height_left_linear_fit_intercept_b_L_error_nm", f"{height_left_intercept_err:.8g}" if np.isfinite(height_left_intercept_err) else "nan"])
        writer.writerow(["height_right_linear_fit_model", "h_right(i)=a_R*i+b_R"])
        writer.writerow(["height_right_linear_fit_slope_a_R_nm", f"{height_right_slope:.8g}" if np.isfinite(height_right_slope) else "nan"])
        writer.writerow(["height_right_linear_fit_slope_a_R_error_nm", f"{height_right_slope_err:.8g}" if np.isfinite(height_right_slope_err) else "nan"])
        writer.writerow(["height_right_linear_fit_intercept_b_R_nm", f"{height_right_intercept:.8g}" if np.isfinite(height_right_intercept) else "nan"])
        writer.writerow(["height_right_linear_fit_intercept_b_R_error_nm", f"{height_right_intercept_err:.8g}" if np.isfinite(height_right_intercept_err) else "nan"])

        writer.writerow([])
        writer.writerow(["Stripe spacing analysis"])
        writer.writerow([
            "profile",
            "forward_mean_spacing_nm",
            "forward_std_spacing_nm",
            "forward_n_peaks",
            "forward_n_spacings",
            "backward_mean_spacing_nm",
            "backward_std_spacing_nm",
            "backward_n_peaks",
            "backward_n_spacings"
        ])

        for r in stripe_results:
            writer.writerow([
                r.profile,
                f"{r.forward_mean_spacing_nm:.8g}" if np.isfinite(r.forward_mean_spacing_nm) else "nan",
                f"{r.forward_std_spacing_nm:.8g}" if np.isfinite(r.forward_std_spacing_nm) else "nan",
                r.forward_n_peaks,
                r.forward_n_spacings,
                f"{r.backward_mean_spacing_nm:.8g}" if np.isfinite(r.backward_mean_spacing_nm) else "nan",
                f"{r.backward_std_spacing_nm:.8g}" if np.isfinite(r.backward_std_spacing_nm) else "nan",
                r.backward_n_peaks,
                r.backward_n_spacings
            ])

        writer.writerow([])
        writer.writerow(["stripe_linear_fit_model", "d(i)=a*i+b"])
        writer.writerow(["stripe_linear_fit_slope_a_nm", f"{stripe_slope:.8g}" if np.isfinite(stripe_slope) else "nan"])
        writer.writerow(["stripe_linear_fit_slope_a_error_nm", f"{stripe_slope_err:.8g}" if np.isfinite(stripe_slope_err) else "nan"])
        writer.writerow(["stripe_linear_fit_intercept_nm", f"{stripe_intercept:.8g}" if np.isfinite(stripe_intercept) else "nan"])
        writer.writerow(["stripe_linear_fit_intercept_error_nm", f"{stripe_intercept_err:.8g}" if np.isfinite(stripe_intercept_err) else "nan"])


def main():
    args = parse_args()

    setup_matplotlib(args.usetex)

    forward_initial_path = args.forward_data if args.forward_data is not None else args.data
    forward_data_path = ask_data_path(forward_initial_path, label="forward", optional=False)
    backward_data_path = ask_data_path(args.backward_data, label="backward", optional=True)
    has_backward = backward_data_path is not None

    height_inset = ask_height_inset(args.height_inset)
    transparent = ask_transparent(args.transparent)

    add_titles = ask("Add titles to plots? [Y/n]: ")

    titles = ["", "", ""]

    if add_titles:
        titles[0] = input("Title for line profile plot: ")
        titles[1] = input("Title for height plot: ")
        titles[2] = input("Title for stripe-spacing plot: ")

    legends = [
        ask("Show legend for line profile plot? [Y/n]: "),
        ask("Show profile-color legend for height plot? [Y/n]: "),
        ask("Show profile-color legend for stripe-spacing plot? [Y/n]: ")
    ]

    forward_profiles = read_profiles(forward_data_path)
    backward_profiles = read_profiles(backward_data_path) if has_backward else None

    results = combine_height_results(forward_profiles, backward_profiles)
    stripe_results = combine_stripe_results(forward_profiles, backward_profiles)

    mean_h, err_h, std_h, all_height_values = height_stats(results)

    left_values_for_display = np.asarray([r.height_left_nm for r in results if np.isfinite(r.height_left_nm)], dtype=float)
    right_values_for_display = np.asarray([r.height_right_nm for r in results if np.isfinite(r.height_right_nm)], dtype=float)

    height_left_slope_pre, height_left_intercept_pre, height_left_slope_err_pre, height_left_intercept_err_pre = side_fit_with_errors(
        results, side="left"
    )
    height_right_slope_pre, height_right_intercept_pre, height_right_slope_err_pre, height_right_intercept_err_pre = side_fit_with_errors(
        results, side="right"
    )

    display_left_h = height_left_intercept_pre if np.isfinite(height_left_intercept_pre) else (float(np.nanmean(left_values_for_display)) if len(left_values_for_display) else np.nan)
    display_left_err_h = height_left_intercept_err_pre if np.isfinite(height_left_intercept_err_pre) else (float(np.nanstd(left_values_for_display, ddof=1)) if len(left_values_for_display) > 1 else 0.0)
    display_right_h = height_right_intercept_pre if np.isfinite(height_right_intercept_pre) else (float(np.nanmean(right_values_for_display)) if len(right_values_for_display) else np.nan)
    display_right_err_h = height_right_intercept_err_pre if np.isfinite(height_right_intercept_err_pre) else (float(np.nanstd(right_values_for_display, ddof=1)) if len(right_values_for_display) > 1 else 0.0)

    base = forward_data_path.stem if not has_backward else f"{forward_data_path.stem}_forward_backward"
    outdir = forward_data_path.parent

    profiles_plot = outdir / f"{base}_profiles_mean_height.png"
    heights_plot = outdir / f"{base}_left_right_heights.png"
    stripes_plot = outdir / f"{base}_stripe_spacing.png"
    summary_csv = outdir / f"{base}_height_summary.csv"

    height_fit_from_profile = plot_profiles(
        forward_profiles,
        backward_profiles,
        results,
        display_left_h,
        display_left_err_h,
        display_right_h,
        display_right_err_h,
        profiles_plot,
        transparent,
        legends[0],
        titles[0],
        height_inset=height_inset,
    )

    if height_inset:
        (
            height_left_slope,
            height_left_intercept,
            height_left_slope_err,
            height_left_intercept_err,
            height_right_slope,
            height_right_intercept,
            height_right_slope_err,
            height_right_intercept_err,
        ) = height_fit_from_profile
    else:
        (
            height_left_slope,
            height_left_intercept,
            height_left_slope_err,
            height_left_intercept_err,
            height_right_slope,
            height_right_intercept,
            height_right_slope_err,
            height_right_intercept_err,
        ) = plot_heights(
            results,
            heights_plot,
            transparent,
            legends[1],
            titles[1],
        )

    stripe_slope, stripe_intercept, stripe_slope_err, stripe_intercept_err = plot_stripes(
        stripe_results,
        stripes_plot,
        transparent,
        legends[2],
        titles[2],
        has_backward=has_backward
    )

    save_summary_csv(
        summary_csv,
        forward_data_path,
        backward_data_path,
        results,
        stripe_results,
        mean_h,
        err_h,
        std_h,
        height_left_slope,
        height_left_intercept,
        height_left_slope_err,
        height_left_intercept_err,
        height_right_slope,
        height_right_intercept,
        height_right_slope_err,
        height_right_intercept_err,
        stripe_slope,
        stripe_intercept,
        stripe_slope_err,
        stripe_intercept_err
    )

    print("\nAnalysis complete.")
    print(f"Forward data file: {forward_data_path}")
    if has_backward:
        print(f"Backward data file: {backward_data_path}")
    else:
        print("Backward data file: not provided; only forward data were used.")
    print(f"Number of forward profiles: {len(forward_profiles)}")
    if has_backward:
        print(f"Number of backward profiles: {len(backward_profiles)}")
    print(f"All selected height values (nm): {np.array2string(all_height_values, precision=4)}")
    print(f"Mean height: {mean_h:.4f} nm ± {err_h:.4f} nm")
    print(f"Error bar ({ERROR_BAR_MODE}): {err_h:.4f} nm")

    forward_stripe_means = np.array([r.forward_mean_spacing_nm for r in stripe_results], dtype=float)
    forward_stripe_stds = np.array([r.forward_std_spacing_nm for r in stripe_results], dtype=float)
    print(f"Forward mean stripe spacings (nm): {np.array2string(forward_stripe_means, precision=4)}")
    print(f"Forward stripe-spacing error bars (nm): {np.array2string(forward_stripe_stds, precision=4)}")

    if has_backward:
        backward_stripe_means = np.array([r.backward_mean_spacing_nm for r in stripe_results], dtype=float)
        backward_stripe_stds = np.array([r.backward_std_spacing_nm for r in stripe_results], dtype=float)
        print(f"Backward mean stripe spacings (nm): {np.array2string(backward_stripe_means, precision=4)}")
        print(f"Backward stripe-spacing error bars (nm): {np.array2string(backward_stripe_stds, precision=4)}")

    if np.isfinite(height_left_slope):
        print(f"Left-edge height linear fit: h_left(i) = a_L*i + b_L")
        print(f"  a_L = {height_left_slope:.4f} ± {height_left_slope_err:.4f} nm")
        print(f"  b_L = {height_left_intercept:.4f} ± {height_left_intercept_err:.4f} nm")
    else:
        print("Left-edge height linear fit: not enough valid points.")

    if np.isfinite(height_right_slope):
        print(f"Right-edge height linear fit: h_right(i) = a_R*i + b_R")
        print(f"  a_R = {height_right_slope:.4f} ± {height_right_slope_err:.4f} nm")
        print(f"  b_R = {height_right_intercept:.4f} ± {height_right_intercept_err:.4f} nm")
    else:
        print("Right-edge height linear fit: not enough valid points.")

    if np.isfinite(stripe_slope):
        print(f"Stripe-spacing linear fit: d(i) = a*i + b")
        print(f"  a = {stripe_slope:.4f} ± {stripe_slope_err:.4f} nm")
        print(f"  b = {stripe_intercept:.4f} ± {stripe_intercept_err:.4f} nm")
    else:
        print("Stripe-spacing linear fit: not enough valid points.")

    print("\nSaved files:")
    print(f"  {profiles_plot}")
    if height_inset:
        print("  Apparent-height plot: included as inset in the mean-profile plot")
    else:
        print(f"  {heights_plot}")
    print(f"  {stripes_plot}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
