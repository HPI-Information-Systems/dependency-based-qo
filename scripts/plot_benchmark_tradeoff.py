#!/usr/bin/env python3.11

import argparse as ap
import json
import os
import re
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter
from palettable.cartocolors.qualitative import Safe_6, Safe_10


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("commit", type=str)
    parser.add_argument("--data", "-d", type=str, default="./hyrise/cmake-build-release/benchmark_plugin_results")
    parser.add_argument("--output", "-o", type=str, default="./figures")
    return parser.parse_args()


def format_number(n):
    return f"{int(n):,.0f}".replace(",", r"\thinspace") if n % 1 == 0 else str(n)


def get_old_new_latency(old_path, new_path):
    try:
        with open(old_path) as old_file:
            old_data = json.load(old_file)

        with open(new_path) as new_file:
            new_data = json.load(new_file)
    except FileNotFoundError:
        return (1, 1)

    if old_data["context"]["benchmark_mode"] != new_data["context"]["benchmark_mode"]:
        exit("Benchmark runs with different modes (ordered/shuffled) are not comparable")

    old_latencies = list()
    new_latencies = list()

    for old, new in zip(old_data["benchmarks"], new_data["benchmarks"]):
        # Create numpy arrays for old/new successful/unsuccessful runs from benchmark dictionary
        old_successful_durations = np.array([run["duration"] for run in old["successful_runs"]], dtype=np.float64)
        new_successful_durations = np.array([run["duration"] for run in new["successful_runs"]], dtype=np.float64)
        # np.mean() defaults to np.float64 for int input
        old_latencies.append(np.mean(old_successful_durations))
        new_latencies.append(np.mean(new_successful_durations))

    return sum(old_latencies), sum(new_latencies)


def get_validation_time(file_name):
    time_regex = re.compile(r"\d+(?=ns\))")

    candidate_time = 0
    with open(file_name) as f:
        for line in f:
            if ("Validated" in line or "Generated" in line) and "candidates" in line:
                candidate_time += int(time_regex.search(line).group())

    return candidate_time


def main(commit, data_dir, output_dir):
    benchmarks = {"TPCDS": "TPC-DS", "JoinOrder": "JOB"}
    benchmarks = {"TPCH": "TPC-H", "TPCDS": "TPC-DS", "StarSchema": "SSB", "JoinOrder": "JOB"}

    sns.set_theme(style="white")

    mpl.use("pgf")

    plt.rcParams.update(
        {
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "pgf.preamble": r"""\usepackage{iftex}
      \ifxetex
        \usepackage[libertine]{newtxmath}
        \usepackage[tt=false]{libertine}
        \setmonofont[StylisticSet=3]{inconsolata}
      \else
        \ifluatex
          \usepackage[libertine]{newtxmath}
          \usepackage[tt=false]{libertine}
          \setmonofont[StylisticSet=3]{inconsolata}
        \else
           \usepackage[tt=false, type1=true]{libertine}
           \usepackage[varqu]{zi4}
           \usepackage[libertine]{newtxmath}
        \fi
      \fi""",
        }
    )

    improvement = dict()
    improvement_rel = dict()
    discovery = dict()
    mode = "st"

    for benchmark, benchmark_title in benchmarks.items():
        common_path = os.path.join(data_dir, f"hyriseBenchmark{benchmark}_{commit}_{mode}")
        if benchmark != "JoinOrder":
            common_path += "_s10"

        base_file = common_path + "_schema.json"
        opt_file = common_path + "_plugin.json"
        log_file = common_path + "_schema_plugin.log"
        base_latency, opt_latency = get_old_new_latency(base_file, opt_file)
        improvement[benchmark_title] = (base_latency - opt_latency) / 10**6
        discovery[benchmark_title] = get_validation_time(log_file) / 10**6
        # print(get_validation_time(log_file) / 10**6)
        improvement_rel[benchmark_title] = 100 - (opt_latency / base_latency * 100)

    bens = list(benchmarks.values())

    bar_width = 0.35
    margin = 0.03

    group_centers = [x * 2 / 3 for x in np.arange(len(benchmarks))]
    group_centers = [x for x in np.arange(len(benchmarks))]
    print(group_centers)
    offsets = [-0.5, 0.5]

    ax = plt.gca()

    baseline_positions = [p + offsets[0] * (bar_width + margin) for p in group_centers]
    ax.bar(
        baseline_positions,
        [max(1, improvement[b]) for b in bens],
        bar_width,
        color=Safe_10.hex_colors[3],
        label="Workload improvement",
        edgecolor="none",
    )
    base_y_pos = list()
    for t, x_pos, b in zip([improvement[b] for b in bens], baseline_positions, bens):
        label = f"{format_number(round(t / 10**3, 1)) if t > 1 else 0}\\thinspace s"
        label = f"{format_number(round(t / 10**3, 1))}\\thinspace s"
        y_pos = t / 2
        # print(b, t, x_pos)
        y_pos = 10 ** ((np.log10(1) + np.log10(t)) / 2) if t > 1 else 1.2
        base_y_pos.append(y_pos)
        label += f"\n({round(improvement_rel[b])}\\thinspace\\%)"
        va = "center" if t > 1 else "bottom"
        color = "white" if t > 1 else "black"
        # print(label)
        ax.text(x_pos, y_pos, label, ha="center", va=va, size=6 * 2, color=color, rotation=0)
        # ax.text(x_pos, t + 50, label, ha="center", va="bottom", size=5 * 2, color="black", rotation=0)

    optimized_positions = [p + offsets[1] * (bar_width + margin) for p in group_centers]
    ax.bar(
        optimized_positions,
        [discovery[b] for b in bens],
        bar_width,
        color=Safe_10.hex_colors[1],
        label="One-shot discovery overhead",
        edgecolor="none",
    )
    for t, x_pos, b in zip([discovery[b] for b in bens], optimized_positions, bens):
        label = format_number(round(t)) if t >= 1 else "<1"
        y_pos = t / 2 if t > 20 else t + 5
        y_pos = 10 ** ((np.log10(1) + np.log10(t)) / 2) if t >= 5 else max(t, 1) * 1.2
        va = "center" if t >= 5 else "bottom"
        color = "white" if t >= 5 else "black"
        ax.text(x_pos, y_pos, label + r"\thinspace ms", ha="center", va=va, size=6 * 2, color=color, rotation=0)

    for ben, a, b, x_pos, y_pos in zip(
        bens, [improvement[b] for b in bens], [discovery[b] for b in bens], optimized_positions, base_y_pos
    ):
        factor = format_number(round(a / b))
        # ax.text(x_pos, y_pos, f"$\\times$\\thinspace{factor}", ha="center", va="center", size=6*2, color="black")
        # print(f"{ben}: {factor}")
        # print(ben, factor, x_pos, a, b, y_pos)
        y = 10 ** ((np.log10(max(2, b)) + np.log10(a)) / 2)
        print(y_pos, y)

        ax.annotate(
            f"$\\times$\\thinspace{factor}",
            xy=(x_pos, a),
            xycoords="data",
            xytext=(x_pos, y),
            textcoords="data",
            arrowprops=dict(arrowstyle="->", fc="0.6", ec="black", shrinkA=0),
            ha="center",
            va="center",
            size=6 * 2,
            color="black",
        )
        ax.annotate(
            "",
            xy=(x_pos, max(b, 2)),
            xycoords="data",
            xytext=(x_pos, y),
            textcoords="data",
            arrowprops=dict(arrowstyle="-", fc="0.6", ec="black", shrinkA=8),
            ha="center",
            va="center",
            size=6 * 2,
            color="black",
        )

    # ax.set_yscale("symlog", linthresh=100)
    ax.set_yscale("log")
    ax.set_ylim(1, ax.get_ylim()[1] * 5)

    max_lim = ax.get_ylim()[1]
    # max_lim = max_lim * 2.5  if scale != "linear" else max_lim * 1.05
    min_lim = 1  # if scale != "log" else 1
    ax.set_ylim(min_lim, max_lim)

    possible_minor_ticks = []
    if True:  # scale != "linear":
        factors = [1, 10, 100, 1000]
        # if True: #scale == "symlog":
        #     factors = [1 / 10] + factors
        for factor in factors:
            possible_minor_ticks += [n * factor for n in range(1, 10)]
    minor_ticks = list()
    for tick in possible_minor_ticks:
        if tick >= min_lim and tick <= max_lim:
            minor_ticks.append(tick)

    x_ticks = bens
    # plt.xticks(x_ticks, x_labels, rotation=15, ha="right", va="top")
    plt.xticks(group_centers, bens)  # , rotation=15, ha="right", va="top")
    y_label = r"Execution time [ms]"
    plt.ylabel(y_label, fontsize=8 * 2)
    plt.xlabel("Benchmark", fontsize=8 * 2)
    # plt.legend(loc="best", fontsize=7 * 2, ncol=2, fancybox=False, framealpha=1.0, edgecolor="black")
    # ncol = 4 if scale == "linear" else 3
    # handles, labels = ax.get_legend_handles_labels()
    # handles = list(reversed(handles))
    # labels = list(reversed(labels))
    plt.legend(
        loc="best", fontsize=6 * 2, fancybox=False, framealpha=1.0, edgecolor="black", ncol=2
    )  # handles=handles, labels=labels, columnspacing=1.0, labelspacing=0.25, handletextpad=0.4, handlelength=1.4)

    # for xpos, b in zip(bar_positions, bens):
    #     ax.text(xpos, -15, b, size=6*2, va="top", ha="center")

    plt.grid(axis="y", visible=True)
    fig = plt.gcf()

    # ax.tick_params(axis="x", which="major", labelsize=5 * 2, width=1, length=6, left=True, bottom=True)
    ax.tick_params(axis="both", which="major", labelsize=7 * 2, width=1, length=6, left=True, bottom=True)
    ax.tick_params(axis="y", which="minor", width=0.5, length=4, left=True)
    # y_ticks = [t for t  in ax.get_yticks() if t <= 100]
    # plt.yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x)))
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    column_width = 3.3374
    fig_width = column_width * 2
    fig_height = column_width * 0.475 * 2  # * 0.9
    fig.set_size_inches(fig_width, fig_height)
    plt.tight_layout(pad=0)

    plt.savefig(os.path.join(output_dir, f"tradeoff.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()

    return

    discovery_times_rel = defaultdict(dict)
    for b in bens:
        discovery_times_rel[b]["levels"] = discovery_times[b]["levels"]
        base_time = discovery_times[b]["times"][0]
        discovery_times_rel[b]["diffs"] = [d / base_time * 100 for d in discovery_times[b]["diffs"]]
    # print(discovery_times_rel)

    relevant_levels = list()
    for idx, level in enumerate(levels):
        if level == "Optimized":
            break
        for b in bens:
            if discovery_times_rel[b]["diffs"][idx] > 10:
                relevant_levels.append(level)
                break

    plot_data = defaultdict(dict)
    for b in bens:
        plot_data[b]["levels"] = [relevant_levels[0]]
        plot_data[b]["diffs"] = [discovery_times_rel[b]["diffs"][0]]
        for level in relevant_levels[1:]:
            plot_data[b]["levels"].append(level)
            diff = discovery_times_rel[b]["diffs"][levels.index(level)]
            if diff < 5:
                diff = 0
            plot_data[b]["diffs"].append(diff)
        plot_data[b]["levels"].append("Remaining")
        diff = 100 - sum(plot_data[b]["diffs"]) - discovery_times_rel[b]["diffs"][-1]
        plot_data[b]["diffs"].append(diff)
        # plot_data[b]["times"].append(discovery_times[b]["times"][-1])
        # plot_data[b]["diffs"].append(discovery_times[b]["times"][-1])
        # plot_data[b]["levels"].append("Optimized")

    relevant_levels += ["Remaining"]
    # relevant_levels += ["Remaining", "Optimized"]
    print(relevant_levels)

    bottom = [0 for _ in bens]

    legend = {
        "CandidateDependence": "Candidate dep.",
        "IndProbeDictionary": "IND probe dicts.",
        "OdSampling": "OD sampling",
        "UccDictionary": "UCC statistics invalid.",
        "UccBulkInsert": "UCC insert partition",
        "UccIndex": "UCC segment index",
        "Remaining": "Remaining opts.",
    }

    plot_colors = [Safe_10.hex_colors[0]] + Safe_10.hex_colors[2:3] + Safe_10.hex_colors[4:]
    # plot_colors = list(reversed(plot_colors[:len(relevant_levels) - 1]))
    bar_positions = [p + offsets[1] * (bar_width + margin) for p in group_centers]
    for idx, level in zip(reversed(range(len(relevant_levels))), reversed(relevant_levels)):
        times = [plot_data[ben]["diffs"][idx] for ben in bens]
        color = plot_colors[idx % len(plot_colors)]  # if level != "Optimized" else Safe_10.hex_colors[3]
        # label = legend[level] if level != "Optimized" else None
        label = legend[level]
        ax.bar(bar_positions, times, bar_width, color=color, label=label, edgecolor="none", bottom=bottom)
        bottom = [a + b for a, b in zip(bottom, times)]
        # if level == levels[0]:
        #     for t, x_pos in zip(times, bar_positions):
        #         ax.text(x_pos, t + 50, format_number(round(t)), ha="center", va="bottom", size=5 * 2, color="black", rotation=0)

        if level in ["Remaining", "Optimized"]:
            continue
        for diff, x_pos, b, y_offset in zip(times, bar_positions, bens, bottom):
            # diff = t - plot_data[b]["times"][idx + 1]
            y_val = round(diff)
            if diff < 1:
                continue
            label = format_number(y_val) + r"\thinspace \%"
            y_pos = y_offset - (diff / 2) if scale == "linear" else y_offset + diff
            va = "center" if scale == "linear" else "top"
            ax.text(x_pos, y_pos, label, ha="center", va=va, size=6 * 2, color="white", rotation=0)

    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylim(0, ax.get_ylim()[1] * 10)
    else:
        ax.set_yscale(scale)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        # ax.set_ylim(0, None)

    x_ticks = list()
    x_labels = [r"Na\"{i}ve", "Impact", "Opt."] * 4
    for n, a, o in zip(baseline_positions, bar_positions, optimized_positions):
        x_ticks += [n, a, o]

    opt_times = [discovery_times[b]["times"][-1] for b in bens]
    opt_times = [round(t) if t >= 10 else round(t, 1) for t in opt_times]
    # x_ticks = [f"{b} ({t}\\ ms)" for b, t in zip(bens, opt_times)]
    # x_ticks = bens
    # plt.xticks(x_ticks, x_labels, rotation=15, ha="right", va="top")
    plt.xticks(x_ticks, x_labels)  # , rotation=15, ha="right", va="top")
    y_label = r"Validation runtime [\%]"
    plt.ylabel(y_label, fontsize=8 * 2)
    plt.xlabel("Benchmark", fontsize=8 * 2, labelpad=13)
    # plt.legend(loc="best", fontsize=7 * 2, ncol=2, fancybox=False, framealpha=1.0, edgecolor="black")
    ncol = 4 if scale == "linear" else 3
    handles, labels = ax.get_legend_handles_labels()
    handles = list(reversed(handles))
    labels = list(reversed(labels))
    plt.legend(
        loc="best",
        fontsize=5 * 2,
        ncol=ncol,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        handles=handles,
        labels=labels,
        columnspacing=1.0,
        labelspacing=0.25,
        handletextpad=0.4,
        handlelength=1.4,
    )

    for xpos, b in zip(bar_positions, bens):
        ax.text(xpos, -15, b, size=6 * 2, va="top", ha="center")

    plt.grid(axis="y", visible=True)
    fig = plt.gcf()

    ax.tick_params(axis="x", which="major", labelsize=5 * 2, width=1, length=6, left=True, bottom=True)
    ax.tick_params(axis="y", which="major", labelsize=7 * 2, width=1, length=6, left=True, bottom=True)
    ax.tick_params(axis="y", which="minor", width=0.5, length=4, left=True)
    y_ticks = [t for t in ax.get_yticks() if t <= 100]
    plt.yticks(y_ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x)))
    # ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    column_width = 3.3374
    fig_width = column_width * 2
    fig_height = column_width * 0.475 * 2  # * 0.9
    fig.set_size_inches(fig_width, fig_height)
    plt.tight_layout(pad=0)

    plt.savefig(
        os.path.join(output_dir, f"validation_ablation_relative_{scale}.pdf"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close()

    return

    offsets = [-1, 1]

    for impl, disc_times, offset, color in zip(
        [r"na\"{i}ve", "optimized"], [discovery_times_old, discovery_times_new], offsets, Safe_6.hex_colors[:2]
    ):

        bar_positions = [p + offset * (0.5 * bar_width + margin) for p in group_centers]
        t_sum = [
            (sum(disc_times[b]["valid"]) + sum(disc_times[b]["invalid"]) + sum(disc_times[b]["skipped"])) / 10**6
            for b in bens
        ]

        print(impl.upper())
        res = [bens.copy(), [str(round(x, 2)) for x in t_sum]]
        for i in range(len(bens)):
            max_len = max([len(r[i]) for r in res])
            for r in res:
                r[i] = r[i].rjust(max_len)
        for r in res:
            print("  ".join(r))
        print()

        ax = plt.gca()
        ax.bar(bar_positions, t_sum, bar_width, color=color, label=f"{impl[0].upper()}{impl[1:]}", edgecolor="none")

        for y, x in zip(t_sum, bar_positions):
            label = str(round(y, 1)) if y < 1 else format_number(round(y))
            # y = y * 0.2 if scale != "linear" else y - 100
            print_above = (scale == "linear" and y < 1500) or (scale != "linear" and y < 1)
            y_pos = y + 100 if print_above else y - 100
            if scale != "linear":
                y_pos = y * 1.2 if print_above else y * 0.8
            va = "bottom" if print_above else "top"
            color = "black" if print_above else "white"
            ax.text(x, y_pos, label, ha="center", va=va, size=7 * 2, color=color, rotation=90)

    print("SPEEDUP")
    for benchmark in bens:
        discovery_time_old = (
            sum(discovery_times_old[benchmark]["valid"])
            + sum(discovery_times_old[benchmark]["invalid"])
            + sum(discovery_times_old[benchmark]["skipped"])
        )
        discovery_time_new = (
            sum(discovery_times_new[benchmark]["valid"])
            + sum(discovery_times_new[benchmark]["invalid"])
            + sum(discovery_times_new[benchmark]["skipped"])
        )
        print(f"{benchmark.rjust(max([len(b) for b in bens]))}: {discovery_time_old / discovery_time_new}")

    for benchmark in bens:
        print(f"\nCOMPARISON {benchmark}")
        for candidate in sorted(candidate_times_old[benchmark].keys()):
            status_old, time_old, time_readable_old = candidate_times_old[benchmark][candidate]
            status_new, time_new, time_readable_new = candidate_times_new[benchmark][candidate]
            print(
                candidate,
                status_old,
                status_new,
                f"{round(time_new * 100 / time_old, 2)}%",
                time_readable_old,
                time_readable_new,
            )

    if scale == "symlog":
        ax.set_yscale("symlog", linthresh=1)
    else:
        ax.set_yscale(scale)
    max_lim = ax.get_ylim()[1]
    max_lim = max_lim * 2.5 if scale != "linear" else max_lim * 1.05
    min_lim = 0 if scale != "log" else 1
    ax.set_ylim(0, max_lim)

    possible_minor_ticks = []
    if scale != "linear":
        factors = [1, 10, 100, 1000]
        if scale == "symlog":
            factors = [1 / 10] + factors
        for factor in factors:
            possible_minor_ticks += [n * factor for n in range(1, 10)]
    minor_ticks = list()
    for tick in possible_minor_ticks:
        if tick >= min_lim and tick <= max_lim:
            minor_ticks.append(tick)

    plt.xticks(group_centers, bens, rotation=0)
    y_label = "Validation runtime [ms]"
    plt.ylabel(y_label, fontsize=8 * 2)
    plt.xlabel("Benchmark", fontsize=8 * 2)
    plt.legend(
        loc="best",
        fontsize=6 * 2,
        ncol=4,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        columnspacing=1.0,
        labelspacing=0.25,
        handlelength=1.0,
        handletextpad=0.4,
    )
    plt.grid(axis="y", visible=True)
    fig = plt.gcf()

    ax.tick_params(axis="both", which="major", labelsize=7 * 2, width=1, length=6, left=True, bottom=True)
    ax.tick_params(axis="y", which="minor", width=0.5, length=4, left=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x)))
    ax.yaxis.set_minor_locator(FixedLocator(minor_ticks))

    column_width = 3.3374
    fig_width = column_width * 2
    fig_height = column_width * 0.475 * 2  # * 0.9
    fig.set_size_inches(fig_width, fig_height)
    plt.tight_layout(pad=0)

    plt.savefig(
        os.path.join(output_dir, f"validation_improvement_{scale}.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.01
    )
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.commit, args.data, args.output)
