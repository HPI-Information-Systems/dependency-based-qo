#!/usr/bin/env python3.11

import argparse as ap
import json
import os
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter
from palettable.cartocolors.qualitative import Safe_10


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
        y_pos = 10 ** ((np.log10(1) + np.log10(t)) / 2) if t > 1 else 1.2
        base_y_pos.append(y_pos)
        label += f"\n({round(improvement_rel[b])}\\thinspace\\%)"
        va = "center" if t > 1 else "bottom"
        color = "white" if t > 1 else "black"
        ax.text(x_pos, y_pos, label, ha="center", va=va, size=6 * 2, color=color, rotation=0)

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

        line_offset = bar_width * 0.45
        ax.plot([x_pos - line_offset, x_pos + line_offset], [a, a], color="black", lw=1)

    ax.set_yscale("log")
    ax.set_ylim(1, ax.get_ylim()[1] * 5)

    max_lim = ax.get_ylim()[1]
    min_lim = 1
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

    plt.xticks(group_centers, bens)
    y_label = r"Execution time [ms]"
    plt.ylabel(y_label, fontsize=8 * 2)
    plt.xlabel("Benchmark", fontsize=8 * 2)
    plt.legend(loc="best", fontsize=6 * 2, fancybox=False, framealpha=1.0, edgecolor="black", ncol=2)

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

    plt.savefig(os.path.join(output_dir, "tradeoff.pdf"), dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.commit, args.data, args.output)
