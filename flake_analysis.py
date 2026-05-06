#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SMOOTH_WINDOW_FRACTION = 0.05
MIN_SMOOTH_WINDOW_POINTS = 5
THRESHOLD_FRACTION = 0.30
EDGE_MARGIN_NM = 0.30
AVERAGING_WINDOW_NM = 1.20
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


@dataclass
class StripeResult:
    profile: str
    mean_spacing_nm: float
    std_spacing_nm: float
    n_peaks: int
    n_spacings: int


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze AFM/Gwyddion line profiles across a flake.")
    parser.add_argument("--data", type=str, default=None, help="Path to Gwyddion profile data file.")
    parser.add_argument("--transparent", type=str, default=None, help="yes/no for transparent plot background.")
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


def ask_data_path(initial_path=None) -> Path:
    while True:
        if initial_path:
            raw = initial_path
        else:
            raw = input("Path to the Gwyddion profile data file: ").strip()

        path = Path(raw.strip('"').strip("'")).expanduser().resolve()
        print(f"\nSelected file:\n{path}")

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

            profile_name = names[col].strip() if col < len(names) and names[col].strip() else f"Profile {len(profiles)+1}"

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
    z = profile.z_nm
    n = len(z)

    window = max(MIN_SMOOTH_WINDOW_POINTS, int(SMOOTH_WINDOW_FRACTION * n))
    z_s = moving_average(z, window)

    low = np.percentile(z_s, 15)
    high = np.percentile(z_s, 85)

    threshold = low + THRESHOLD_FRACTION * (high - low)

    mask = z_s > threshold
    idx = np.where(mask)[0]

    if len(idx) < 2:
        raise RuntimeError(f"Could not detect the flake plateau for {profile.name}.")

    return idx[0], idx[-1]


def mean_window(x, z, a, b):
    m = (x >= a) & (x <= b)
    if np.sum(m) < 2:
        return np.nan
    return float(np.nanmean(z[m]))


def analyze_height(profile):
    left_idx, right_idx = detect_edges(profile)

    x = profile.x_nm
    z = profile.z_nm

    xl = x[left_idx]
    xr = x[right_idx]

    z_left_terrace = mean_window(x, z, max(x[0], xl - EDGE_MARGIN_NM - AVERAGING_WINDOW_NM), xl - EDGE_MARGIN_NM)
    z_left_flake = mean_window(x, z, xl + EDGE_MARGIN_NM, min(x[-1], xl + EDGE_MARGIN_NM + AVERAGING_WINDOW_NM))

    z_right_flake = mean_window(x, z, max(x[0], xr - EDGE_MARGIN_NM - AVERAGING_WINDOW_NM), xr - EDGE_MARGIN_NM)
    z_right_terrace = mean_window(x, z, xr + EDGE_MARGIN_NM, min(x[-1], xr + EDGE_MARGIN_NM + AVERAGING_WINDOW_NM))

    return HeightResult(
        profile.name,
        z_left_flake - z_left_terrace,
        z_right_flake - z_right_terrace
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


def height_stats(results):
    values = np.array(
        [v for r in results for v in [r.height_left_nm, r.height_right_nm] if np.isfinite(v)],
        dtype=float
    )
    mean_h = float(np.nanmean(values))
    std_h = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
    err_h = std_h
    return mean_h, err_h, std_h, values


def plot_profiles(profiles, mean_h, err_h, out, transparent, show_legend, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, p in enumerate(profiles):
        ax.plot(
            p.x_nm,
            p.z_nm,
            color=PROFILE_COLORS[i % len(PROFILE_COLORS)],
            lw=1.2,
            alpha=0.7,
            label=p.name
        )

    x_mean, z_mean = common_mean_profile(profiles)

    ax.plot(
        x_mean,
        z_mean,
        color="black",
        lw=2.6,
        label="Mean profile"
    )

    # Force the full x-axis to include the longest profile.
    ax.set_xlim(
        min(float(p.x_nm[0]) for p in profiles),
        max(float(p.x_nm[-1]) for p in profiles)
    )

    ax.set_xlabel(r"Distance $x$ (nm)")
    ax.set_ylabel(r"Height $z$ (nm)")

    if title:
        ax.set_title(title)

    ax.text(
        0.03,
        0.95,
        rf"$h = {mean_h:.3f} \pm {err_h:.3f}\,\mathrm{{nm}}$",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    ax.grid(False)

    if show_legend:
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out, transparent=transparent, bbox_inches="tight")
    plt.close(fig)


def plot_heights(results, mean_h, err_h, out, transparent, show_legend, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    handles = []

    for i, r in enumerate(results, start=1):
        color = PROFILE_COLORS[(i-1) % len(PROFILE_COLORS)]

        ax.scatter(i-0.08, r.height_left_nm, color=color, s=60)
        ax.scatter(i+0.08, r.height_right_nm, color=color, marker="^", s=60)

        ax.plot(
            [i-0.08, i+0.08],
            [r.height_left_nm, r.height_right_nm],
            color=color,
            alpha=0.7
        )

        handles.append(
            Line2D([0], [0], color=color, lw=1.5, label=r.profile)
        )

    mean_x = len(results) + 1

    ax.errorbar(
        mean_x,
        mean_h,
        yerr=err_h,
        color="black",
        marker="s",
        markersize=8,
        capsize=4,
        lw=1.6,
        label=rf"Mean $\pm 1\sigma$"
    )

    ax.set_xlabel(r"Profile index")
    ax.set_ylabel(r"Apparent height $h$ (nm)")

    ax.set_xticks(list(range(1, len(results)+1)) + [mean_x])
    ax.set_xticklabels([str(i) for i in range(1, len(results)+1)] + [r"Mean"])

    if title:
        ax.set_title(title)

    ax.grid(False)

    marker_note = Line2D([0], [0], color="none", marker="o", markerfacecolor="none", markeredgecolor=UNIBAS["anthrazit"], label=r"Left edge")
    marker_note_2 = Line2D([0], [0], color="none", marker="^", markerfacecolor="none", markeredgecolor=UNIBAS["anthrazit"], label=r"Right edge")

    # Always display the left/right marker legend. If show_legend is True,
    # also include the profile-color legend.
    legend_handles = handles + [marker_note, marker_note_2] if show_legend else [marker_note, marker_note_2]
    ax.legend(handles=legend_handles, frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, transparent=transparent, bbox_inches="tight")
    plt.close(fig)


def plot_stripes(stripe_results, out, transparent, show_legend, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    xvals = np.arange(1, len(stripe_results)+1)
    stripe_means = np.array([r.mean_spacing_nm for r in stripe_results], dtype=float)
    stripe_stds = np.array([r.std_spacing_nm for r in stripe_results], dtype=float)

    for i, r in enumerate(stripe_results):
        color = PROFILE_COLORS[i % len(PROFILE_COLORS)]

        ax.errorbar(
            xvals[i],
            r.mean_spacing_nm,
            yerr=r.std_spacing_nm,
            fmt='o',
            color=color,
            capsize=4,
            markersize=7,
            label=r.profile
        )

    valid = np.isfinite(stripe_means)

    slope = np.nan
    intercept = np.nan

    if np.sum(valid) >= 2:
        coeffs = np.polyfit(xvals[valid], stripe_means[valid], 1)
        slope, intercept = float(coeffs[0]), float(coeffs[1])

        fit_x = np.linspace(1, len(stripe_results), 200)
        fit_y = slope * fit_x + intercept

        ax.plot(fit_x, fit_y, color="black", lw=2, label=r"Linear fit")

        text = (
            rf"$d(i)=a\,i+b$" + "\n" +
            rf"$a={slope:.4f}\,\mathrm{{nm}}$" + "\n" +
            rf"$b={intercept:.4f}\,\mathrm{{nm}}$"
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

    if show_legend:
        ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out, transparent=transparent, bbox_inches="tight")
    plt.close(fig)

    return slope, intercept


def save_summary_csv(path, results, stripe_results, mean_h, err_h, std_h, stripe_slope, stripe_intercept):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Height analysis"])
        writer.writerow([
            "profile",
            "height_left_nm",
            "height_right_nm"
        ])

        for r in results:
            writer.writerow([
                r.profile,
                f"{r.height_left_nm:.8g}",
                f"{r.height_right_nm:.8g}"
            ])

        writer.writerow([])
        writer.writerow(["global_mean_height_nm", f"{mean_h:.8g}"])
        writer.writerow([f"global_error_height_nm_{ERROR_BAR_MODE}", f"{err_h:.8g}"])
        writer.writerow(["global_std_height_nm", f"{std_h:.8g}"])

        writer.writerow([])
        writer.writerow(["Stripe spacing analysis"])
        writer.writerow([
            "profile",
            "mean_spacing_nm",
            "std_spacing_nm",
            "n_peaks",
            "n_spacings"
        ])

        for r in stripe_results:
            writer.writerow([
                r.profile,
                f"{r.mean_spacing_nm:.8g}" if np.isfinite(r.mean_spacing_nm) else "nan",
                f"{r.std_spacing_nm:.8g}" if np.isfinite(r.std_spacing_nm) else "nan",
                r.n_peaks,
                r.n_spacings
            ])

        writer.writerow([])
        writer.writerow(["stripe_linear_fit_model", "d(i)=a*i+b"])
        writer.writerow(["stripe_linear_fit_slope_a_nm", f"{stripe_slope:.8g}" if np.isfinite(stripe_slope) else "nan"])
        writer.writerow(["stripe_linear_fit_intercept_nm", f"{stripe_intercept:.8g}" if np.isfinite(stripe_intercept) else "nan"])


def main():
    args = parse_args()

    setup_matplotlib(args.usetex)

    data_path = ask_data_path(args.data)
    transparent = ask_transparent(args.transparent)

    add_titles = ask("Add titles to plots? [Y/n]: ")

    titles = ["", "", ""]

    if add_titles:
        titles[0] = input("Title for line profile plot: ")
        titles[1] = input("Title for height plot: ")
        titles[2] = input("Title for stripe-spacing plot: ")

    legends = [
        ask("Show legend for line profile plot? [Y/n]: "),
        ask("Show legend for height plot? [Y/n]: "),
        ask("Show legend for stripe-spacing plot? [Y/n]: ")
    ]

    profiles = read_profiles(data_path)
    results = [analyze_height(p) for p in profiles]
    stripe_results = [analyze_stripes(p) for p in profiles]

    mean_h, err_h, std_h, all_height_values = height_stats(results)

    base = data_path.stem
    outdir = data_path.parent

    profiles_plot = outdir / f"{base}_profiles_mean_height.png"
    heights_plot = outdir / f"{base}_left_right_heights.png"
    stripes_plot = outdir / f"{base}_stripe_spacing.png"
    summary_csv = outdir / f"{base}_height_summary.csv"

    plot_profiles(
        profiles,
        mean_h,
        err_h,
        profiles_plot,
        transparent,
        legends[0],
        titles[0]
    )

    plot_heights(
        results,
        mean_h,
        err_h,
        heights_plot,
        transparent,
        legends[1],
        titles[1]
    )

    stripe_slope, stripe_intercept = plot_stripes(
        stripe_results,
        stripes_plot,
        transparent,
        legends[2],
        titles[2]
    )

    save_summary_csv(
        summary_csv,
        results,
        stripe_results,
        mean_h,
        err_h,
        std_h,
        stripe_slope,
        stripe_intercept
    )

    print("\nAnalysis complete.")
    print(f"Number of profiles: {len(profiles)}")
    print(f"All left/right height values (nm): {np.array2string(all_height_values, precision=4)}")
    print(f"Mean height: {mean_h:.4f} nm ± {err_h:.4f} nm")
    print(f"Error bar ({ERROR_BAR_MODE}): {err_h:.4f} nm")

    stripe_means = np.array([r.mean_spacing_nm for r in stripe_results], dtype=float)
    stripe_stds = np.array([r.std_spacing_nm for r in stripe_results], dtype=float)
    print(f"All mean stripe spacings (nm): {np.array2string(stripe_means, precision=4)}")
    print(f"All stripe-spacing error bars (nm): {np.array2string(stripe_stds, precision=4)}")
    if np.isfinite(stripe_slope):
        print(f"Stripe-spacing linear fit: d(i) = a*i + b")
        print(f"  a = {stripe_slope:.4f} nm")
        print(f"  b = {stripe_intercept:.4f} nm")
    else:
        print("Stripe-spacing linear fit: not enough valid points.")

    print("\nSaved files:")
    print(f"  {profiles_plot}")
    print(f"  {heights_plot}")
    print(f"  {stripes_plot}")
    print(f"  {summary_csv}")


if __name__ == "__main__":
    main()
