"""Plotting helpers — save PNGs to outputs/.

All functions accept an optional output_dir (default: config.OUTPUTS_DIR),
save the figure there, print the relative path, and return the Path.
No plt.show() is ever called — this module is pipeline-only.

Public API
----------
plot_decomposition(result, title, filename, output_dir)
plot_weekday_profile(profile, region, metric, output_dir)
plot_hour_profile(profile, region, output_dir)
plot_duration_distribution(all_dur, non_naive_dur, region, output_dir)
plot_structural_break(series, rolling_mean, region, output_dir)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported anywhere
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult

from airraid_tsa.config import OUTPUTS_DIR

_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slug(name: str) -> str:
    """Filename-safe slug: lowercase, spaces/slashes → underscores."""
    return name.lower().replace(" ", "_").replace("/", "_")


def _save(fig: plt.Figure, filename: str, output_dir: Path) -> Path:
    """Save figure to output_dir, close it, print the path. Returns Path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Plot: decomposition
# ---------------------------------------------------------------------------

def plot_decomposition(
    result: DecomposeResult,
    title: str,
    filename: str,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Four-panel additive decomposition figure: observed / trend / seasonal / residual.

    Parameters
    ----------
    result     : DecomposeResult   From statsmodels seasonal_decompose.
    title      : str               Figure suptitle.
    filename   : str               Output filename (e.g. 'kyiv_city_decomp.png').
    output_dir : Path

    Returns
    -------
    Path   Saved PNG path.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=13, y=1.01)

    panels = [
        (result.observed, "Observed"),
        (result.trend,    "Trend"),
        (result.seasonal, "Seasonal"),
        (result.resid,    "Residual"),
    ]
    for ax, (data, label) in zip(axes, panels):
        ax.plot(data.index, data.values, linewidth=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    return _save(fig, filename, output_dir)


# ---------------------------------------------------------------------------
# Plot: weekday profile
# ---------------------------------------------------------------------------

def plot_weekday_profile(
    profile: pd.Series,
    region: str,
    metric: str = "alert_minutes",
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Bar chart of mean metric value by day of week (Mon–Sun).

    Parameters
    ----------
    profile  : pd.Series   Index 0–6, values = mean of the metric.
    region   : str         Used in the title and filename.
    metric   : str         Column name for axis label (e.g. 'alert_minutes').
    output_dir : Path

    Returns
    -------
    Path
    """
    labels = [_DAY_LABELS[i] for i in profile.index]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, profile.values, color="steelblue")
    ax.set_title(f"{region} — mean {metric} by day of week")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_xlabel("Day of week")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, f"{_slug(region)}_weekday_profile.png", output_dir)


# ---------------------------------------------------------------------------
# Plot: hour-of-day profile
# ---------------------------------------------------------------------------

def plot_hour_profile(
    profile: pd.Series,
    region: str,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Bar chart of alert count by UTC start hour (0–23).

    Parameters
    ----------
    profile  : pd.Series   Index 0–23, values = alert start count.
    region   : str         Used in the title and filename.
    output_dir : Path

    Returns
    -------
    Path
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(profile.index, profile.values, color="tomato")
    ax.set_title(f"{region} — alert starts by hour of day (UTC)")
    ax.set_xlabel("Hour (UTC)")
    ax.set_ylabel("Alert count")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, f"{_slug(region)}_hour_profile.png", output_dir)


# ---------------------------------------------------------------------------
# Plot: duration distribution
# ---------------------------------------------------------------------------

def plot_duration_distribution(
    all_durations: pd.Series,
    non_naive_durations: pd.Series,
    region: str,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Overlapping histograms: all alerts vs non-naive alerts.

    Naive rows have a fixed 30-min estimated duration, so they create an
    artificial spike.  Showing both distributions makes that visible.

    Parameters
    ----------
    all_durations       : pd.Series   Durations in minutes (all rows).
    non_naive_durations : pd.Series   Durations in minutes (naive excluded).
    region              : str         Used in title and filename.
    output_dir          : Path

    Returns
    -------
    Path
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    bins = 30

    if not all_durations.empty:
        ax.hist(
            all_durations,
            bins=bins,
            alpha=0.5,
            color="steelblue",
            label=f"All (n={len(all_durations)})",
        )
    if not non_naive_durations.empty:
        ax.hist(
            non_naive_durations,
            bins=bins,
            alpha=0.55,
            color="darkorange",
            label=f"Non-naive (n={len(non_naive_durations)})",
        )

    ax.set_title(f"{region} — alert duration distribution")
    ax.set_xlabel("Duration (minutes)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    return _save(fig, f"{_slug(region)}_duration_hist.png", output_dir)


# ---------------------------------------------------------------------------
# Plot: structural break / rolling mean
# ---------------------------------------------------------------------------

def plot_structural_break(
    series: pd.Series,
    rolling_mean: pd.Series,
    region: str,
    output_dir: Path = OUTPUTS_DIR,
) -> Path:
    """Daily series + 30-day rolling mean, with 2025 break period shaded.

    The orange band marks Jan–Jul 2025, when some regions moved to district-level
    aggregation.  It is shaded only when the series extends into that window.

    Parameters
    ----------
    series       : pd.Series   Full-range daily series (e.g. alert_minutes).
    rolling_mean : pd.Series   30-day rolling mean (from structural_break_view).
    region       : str         Used in title and filename.
    output_dir   : Path

    Returns
    -------
    Path
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(
        series.index, series.values,
        linewidth=0.6, alpha=0.55, color="steelblue", label="Daily",
    )
    ax.plot(
        rolling_mean.index, rolling_mean.values,
        linewidth=2.0, color="firebrick", label="30-day rolling mean",
    )

    # Shade the 2025 methodology-change window only if data reaches it.
    break_start = pd.Timestamp("2025-01-01", tz="UTC")
    break_end   = pd.Timestamp("2025-07-01", tz="UTC")
    if series.index.max() >= break_start:
        shade_end = min(break_end, series.index.max())
        ax.axvspan(
            break_start, shade_end,
            alpha=0.15, color="orange", label="2025 methodology shift (shaded)",
        )

    ax.set_title(f"{region} — alert_minutes with 30-day rolling mean")
    ax.set_xlabel("Date")
    ax.set_ylabel("alert_minutes")
    ax.legend()
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    return _save(fig, f"{_slug(region)}_structural_break.png", output_dir)