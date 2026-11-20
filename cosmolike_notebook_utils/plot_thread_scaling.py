"""Display supplied thread-scaling measurements without performing any timing.

Each workload evaluates the same ten parameter points at one, four and
eight OpenMP threads, following a separate warmup. The legacy record field
``repetition`` identifies the parameter point; it does not mean repeated
evaluation of an identical fiducial model.
The baseline is that workload's median one-thread time. Dividing it by each
elapsed time gives the displayed speedup points; dividing it by a median
elapsed time gives the speedup summary. The vertical ranges show the
observed parameter bank. They include parameter dependence and machine
variation, so they measure neither repeatability nor confidence intervals.

This module imports no scientific prediction or compiled interface. Its
caller sets the Cocoa STIX plotting style and saves the returned figure.
"""

from numbers import Integral, Real

import numpy as np
from matplotlib import pyplot as plt


def _timing_arrays(records):
    """Validate complete measurements before creating a Matplotlib figure.

    Arguments:
        records = thirty dictionaries per workload containing workload, threads,
            repetition (parameter-point index) and seconds; extra metadata
            is retained by the caller.
    Returns:
        names = one, two or four workload labels in first appearance order.
        seconds = owned [workload, thread choice, parameter point] float64 array.

    A duplicate is an error even if its value equals the existing record.
    Neither a missing parameter point nor a failed measurement is imputed.
    """
    if not isinstance(records, (list, tuple)) or len(records) not in (30, 60, 120):
        raise ValueError(
            "provide 30, 60 or 120 records: one, two or four complete workloads"
        )
    workload_count = len(records) // 30
    names = []
    entries = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("every timing record must be a dictionary")
        required = ("workload", "threads", "repetition", "seconds")
        if any(name not in record for name in required):
            raise ValueError("timing records need workload, threads, repetition, seconds")
        workload = record["workload"]
        threads = record["threads"]
        repetition = record["repetition"]
        elapsed = record["seconds"]
        if not isinstance(workload, str) or not workload.strip():
            raise ValueError("workload must be a nonempty string")
        if (isinstance(threads, bool) or not isinstance(threads, Integral)
                or threads not in (1, 4, 8)):
            raise ValueError("threads must be one of the integers 1, 4, 8")
        if (isinstance(repetition, bool) or not isinstance(repetition, Integral)
                or repetition not in range(10)):
            raise ValueError("repetition is a parameter-point index from 0 through 9")
        if (isinstance(elapsed, bool) or not isinstance(elapsed, Real)
                or not np.isfinite(elapsed) or elapsed <= 0):
            raise ValueError("seconds must be positive finite measured numbers")
        key = (workload, int(threads), int(repetition))
        if key in entries:
            raise ValueError("duplicate workload/thread/repetition record")
        entries[key] = float(elapsed)
        if workload not in names:
            names.append(workload)
    if len(names) != workload_count:
        raise ValueError("each named workload needs exactly thirty measurements")
    seconds = np.empty((workload_count, 3, 10), dtype=np.float64)
    for workload_index, workload in enumerate(names):
        for thread_index, threads in enumerate((1, 4, 8)):
            for repetition in range(10):
                key = (workload, threads, repetition)
                if key not in entries:
                    raise ValueError(
                        "each workload needs all ten parameter points at 1, 4, 8 threads"
                    )
                seconds[workload_index, thread_index, repetition] = entries[key]
    return names, seconds


def plot_thread_scaling(records, *, title=None, figsize=(15, 6.5)):
    """Plot complete OpenMP timing series, including separate NLA/TATT workloads.

    Arguments:
        records = thirty dictionaries per workload, for one, two or four
            named workloads. Each has workload (string), threads
            (integer 1,4,8), repetition (parameter-point index0..9), seconds
            (positive finite). The ten parameter points must match across
            thread counts; records contain no warmup measurements. The caller
            retains the point coordinates and their sampling definition.
            Distinct labels identify LSST Y1 and 3pcf, each with NLA or TATT;
            one- and two-workload inputs also allow partial comparison figures.
        title = optional provenance label, such as "Controlled test arrays".
        figsize = width and height in inches; STIX style is set by the caller.
    Returns:
        figure, axes = Matplotlib figure and two axes, elapsed time then speedup.
    Raises:
        ValueError for incomplete, duplicated or invalid records, before drawing.
    Side effects:
        Creates one figure. Does not time, save, display or alter any input.

    For each workload, T1 is median(seconds at one thread). Raw speedup
    points are T1/seconds[n,repetition]; their summary is T1/median(Tn),
    which need not equal the median of the ratios. The extrema transform
    in reverse order: a longer elapsed time means a smaller speedup. Each
    workload uses its own baseline, with no normalization between models.

    Ten parameter points are retained at every thread count. Small deterministic
    horizontal offsets separate coincident observations; they do not
    represent fractional thread counts or change a measured ordinate.
    Min/max bars include both parameter dependence and machine variation.
    They do not estimate repeatability at fixed parameters or a confidence
    interval. The plot cannot infer the parameter coordinates from the times;
    the timing controller must preserve the matched bank and exclude warmup.
    The ideal line S=n describes perfect linear scaling, not a fitted model.
    """
    names, seconds = _timing_arrays(records=records)
    if title is not None and not isinstance(title, str):
        raise ValueError("title must be a string or None")
    ordered = np.sort(seconds, axis=2)
    # Exactly ten observations give the midpoint of entries4 and5. This
    # form avoids overflowing their sum while retaining very small times.
    medians = ordered[:, :, 4] + 0.5 * (ordered[:, :, 5] - ordered[:, :, 4])
    baseline = medians[:, 0]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        speedup = baseline[:, None, None] / seconds
        median_speedup = baseline[:, None] / medians
    if (not np.all(np.isfinite(speedup)) or np.any(speedup <= 0)
            or not np.all(np.isfinite(median_speedup))):
        raise ValueError("speedup cannot be represented; check the measured time units")
    largest_time = float(np.max(seconds))
    largest_speedup = max(8.0, float(np.max(speedup)))
    upper_bounds = np.asarray([largest_time, largest_speedup]) * 1.1
    if not np.all(np.isfinite(upper_bounds)):
        raise ValueError("elapsed-time or speedup plotting range overflows")

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize,
                                constrained_layout=True)
    thread_counts = np.asarray([1.0, 4.0, 8.0])
    jitter = np.linspace(-0.075, 0.075, 10)
    colors = ("#17629c", "#b34b00", "#08756c", "#813c8b")
    markers = ("o", "s", "^", "v")
    line_styles = ("solid", "dashdot", "dashed", "dotted")
    workload_offsets = np.linspace(-0.11 * (len(names) - 1),
                                    0.11 * (len(names) - 1), len(names))
    handles = []
    for workload in range(len(names)):
        centers = thread_counts + workload_offsets[workload]
        positions = centers[:, None] + jitter[None, :]
        for panel, (values, summary) in enumerate((
            (seconds[workload], medians[workload]),
            (speedup[workload], median_speedup[workload]),
        )):
            axis = axes[panel]
            axis.scatter(positions.ravel(), values.ravel(), s=24,
                         marker=markers[workload], color=colors[workload],
                         alpha=0.65, edgecolors="none", zorder=3)
            lower = np.min(values, axis=1)
            upper = np.max(values, axis=1)
            ranges = np.vstack((summary - lower, upper - summary))
            artist = axis.errorbar(
                centers, summary, yerr=ranges, marker="D", markersize=6,
                color=colors[workload], linestyle=line_styles[workload],
                linewidth=1.8, capsize=5, elinewidth=1.4, zorder=4,
                label=names[workload],
            )
            if panel == 0:
                handles.append(artist)
    ideal, = axes[1].plot(thread_counts, thread_counts, color="0.25",
                          linestyle="dashed", linewidth=1.5,
                          label="Ideal linear speedup")
    handles.append(ideal)
    labels = ["Elapsed time [s]", r"Speedup: median($T_1$) / $T_n$"]
    for index, axis in enumerate(axes):
        axis.set_xticks(thread_counts, labels=["1", "4", "8"])
        axis.set_xlim(0.55, 8.45)
        axis.set_ylim(0.0, upper_bounds[index])
        axis.set_ylabel(labels[index], fontsize=23)
        axis.tick_params(axis="both", labelsize=19)
    figure.supxlabel("OpenMP threads (BLAS threads = 1)", fontsize=23)
    explanation = (
        "Dots: 10 parameter points; diamonds: median summaries\n"
        "Bars: min–max across parameters + machine variation (not CI or repeatability)"
    )
    if title:
        explanation = title + "\n" + explanation
    figure.legend(handles=handles, loc="outside upper center", ncol=3,
                  frameon=False, fontsize=18, title=explanation,
                  title_fontsize=17)
    return figure, axes
