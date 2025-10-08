#!/usr/bin/env python3.11

import argparse as ap
import json
import os
import re
from collections import defaultdict
from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter
from palettable.cartocolors.qualitative import Safe_6


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("commit", type=str)
    parser.add_argument("--data", "-d", type=str, default="./hyrise/cmake-build-release/benchmark_plugin_results")
    parser.add_argument("--output", "-o", type=str, default="./figures")
    parser.add_argument("--scale", "-s", type=str, default="linear", choices=["linear", "log", "symlog"])
    return parser.parse_args()


def format_number(n, penalty_factor):
    # if n < 0:
    #     n = n / abs(penalty_factor)

    if n < 1:
        return str(round(n, 1))
    return f"{int(n):,.0f}".replace(",", r"\thinspace") if n % 1 == 0 else str(n)


def to_s(v):
    def val_to_s(x):
        return x / 10**9

    if not isinstance(v, list):
        return val_to_s(v)
    return [val_to_s(i) for i in v]


def get_latencies(old_path, new_path):
    try:
        with open(old_path) as old_file:
            old_data = json.load(old_file)

        with open(new_path) as new_file:
            new_data = json.load(new_file)
    except FileNotFoundError:
        return (2000, 1000)

    if old_data["context"]["benchmark_mode"] != new_data["context"]["benchmark_mode"]:
        exit("Benchmark runs with different modes (ordered/shuffled) are not comparable")

    old_latencies = list()
    new_latencies = list()

    for old, new in zip(old_data["benchmarks"], new_data["benchmarks"]):
        # Create numpy arrays for old/new successful/unsuccessful runs from benchmark dictionary
        old_successful_durations = np.array([run["duration"] for run in old["successful_runs"]], dtype=np.float64)
        new_successful_durations = np.array([run["duration"] for run in new["successful_runs"]], dtype=np.float64)
        old_latencies.append(np.mean(old_successful_durations))
        new_latencies.append(np.mean(new_successful_durations))

    return sum(old_latencies), sum(new_latencies)


def get_discovery_time(common_path):
    time_regexes = [
        re.compile(r"\d+(?=\ss\s)"),
        re.compile(r"\d+(?=\sms\s)"),
        re.compile(r"\d+(?=\sµs\s)"),
        re.compile(r"\d+(?=\sns\s)"),
    ]
    time_divs = list(reversed([1, 10**3, 10**6, 10**9]))
    generation_time_indicator = "Generated "
    validation_time_indicator = "Validated "
    discovery_time = 0

    with open(common_path) as f:
        for line in f:
            if not (line.startswith(generation_time_indicator) or line.startswith(validation_time_indicator)):
                continue

            for regex, div in zip(time_regexes, time_divs):
                r = regex.search(line)
                if not r:
                    continue
                discovery_time += int(r.group(0)) * div

    return discovery_time


def main(commit, data_dir, output_dir, scale):
    benchmarks = {"TPCH": "TPC-H", "TPCDS": "TPC-DS", "StarSchema": "SSB"}
    all_scale_factors = range(1, 101)
    discovery_visualization_factor = 1

    base_palette = Safe_6.hex_colors

    latency_improvements = defaultdict(list)
    discovery_times = defaultdict(list)
    scale_factors = list()
    latency_improvements_relative = defaultdict(list)
    discovery_times_relative = defaultdict(list)

    for scale_factor in all_scale_factors:
        sf_indicator = f"_s{scale_factor}"
        if not os.path.isfile(os.path.join(data_dir, f"hyriseBenchmarkTPCH_{commit}_st{sf_indicator}_plugin.log")):
            continue

        scale_factors.append(scale_factor)

        for benchmark, benchmark_title in benchmarks.items():
            common_path = os.path.join(data_dir, f"hyriseBenchmark{benchmark}_{commit}_st{sf_indicator}")
            old_path = common_path + "_all_off.json"
            new_path = common_path + "_plugin.json"

            old_latency, new_latency = get_latencies(old_path, new_path)
            discovery_time = get_discovery_time(f"{common_path}_plugin.log")

            latency_improvements[benchmark_title].append(to_s(old_latency - new_latency))
            discovery_times[benchmark_title].append(to_s(discovery_time))

            latency_improvements_relative[benchmark_title].append((old_latency - new_latency) * 100 / old_latency)
            discovery_times_relative[benchmark_title].append(discovery_time * 100 / old_latency)

    result_table = list()
    result_table.append(["", ""] + [str(sf) for sf in scale_factors])
    result_table.append(["-", "-"] + ["-" for _ in scale_factors])

    for benchmark_title in benchmarks.values():
        result_table.append(
            [benchmark_title, "Latency"] + [str(round(lat, 2)) + " s" for lat in latency_improvements[benchmark_title]]
        )
        result_table.append(
            [benchmark_title, "Latency [%]"]
            + [str(round(lat, 2)) + " %" for lat in latency_improvements_relative[benchmark_title]]
        )
        result_table.append(
            [benchmark_title, "Validation"]
            + [str(round(lat * 1000, 1)) + " ms" for lat in discovery_times[benchmark_title]]
        )
        result_table.append(
            [benchmark_title, "Val [%]"]
            + [str(round(lat, 2)) + " %" for lat in discovery_times_relative[benchmark_title]]
        )

    for i in range(len(result_table[0])):
        max_len = max([len(res[i]) for res in result_table])
        for j in range(len(result_table)):
            info = result_table[j][i]
            if j == 0:
                info = info.center(max_len)
            elif j == 1:
                info = info * max_len
            elif i == 1:
                info = info.ljust(max_len)
            else:
                info = info.rjust(max_len)
            result_table[j][i] = info

    for i in range(len(result_table)):
        column_sep = " | " if i != 1 else "-+-"
        merge_sep = " " if i != 1 else "-"
        prompt = column_sep.join([merge_sep.join(result_table[i][:2])] + result_table[i][2:])
        print(prompt)

    for measurement_type, lat_improvements, disc_times in zip(
        ["abs", "rel"],
        [latency_improvements, latency_improvements_relative],
        [discovery_times, discovery_times_relative],
    ):
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

        x_axis_sf = list()
        y_axis_time = list()
        indicator_measurement = list()
        indicator_benchmark = list()

        for benchmark, latency_improvement in lat_improvements.items():
            discovery_time = [t * discovery_visualization_factor for t in disc_times[benchmark]]
            assert len(discovery_time) == len(latency_improvement) and len(discovery_time) == len(scale_factors)

            x_axis_sf += scale_factors * 2

            y_axis_time += latency_improvement
            y_axis_time += discovery_time

            indicator_measurement += ["Latency improvement"] * len(latency_improvement)
            indicator_measurement += ["Discovery overhead"] * len(discovery_time)

            indicator_benchmark += [benchmark] * len(scale_factors) * 2

        assert (
            len(x_axis_sf) == len(y_axis_time)
            and len(x_axis_sf) == len(indicator_measurement)
            and len(x_axis_sf) == len(indicator_benchmark)
        )

        values = pd.DataFrame(
            data={
                "x": x_axis_sf,
                "y": y_axis_time,
                "Measurement": indicator_measurement,
                "Benchmark": indicator_benchmark,
            }
        )

        dashes = {"Discovery overhead": (3, 3), "Latency improvement": ""}
        markers = ["^", "X", "s", "D", ".", "o"]

        sns.lineplot(
            data=values,
            x="x",
            y="y",
            style="Measurement",
            markers=markers[:2],
            markersize=8,
            hue="Benchmark",
            dashes=dashes,
            palette=base_palette[: len(benchmarks)],
        )

        ax = plt.gca()

        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=1)
        else:
            ax.set_yscale(scale)
        min_lim, max_lim = plt.ylim()
        if measurement_type == "rel":
            plt.ylim((min_lim, max_lim * 2))
        elif scale == "linear":
            plt.ylim((min_lim, max_lim * 1.1))
        minimal_tick = [0.1 * discovery_visualization_factor] if abs(discovery_visualization_factor) != 1 else []
        ax.yaxis.set_major_locator(FixedLocator(minimal_tick + list(range(0, ceil(plt.ylim()[1]), 25))))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x, discovery_visualization_factor)))

        y_label = "Execution benefit [s]" if measurement_type == "abs" else "Share of execution benefit [%]"
        plt.ylabel(y_label, fontsize=8 * 2)
        plt.xlabel("Scale factor", fontsize=8 * 2)
        plt.legend(fontsize=6 * 2, fancybox=False, framealpha=1.0, ncols=2, edgecolor="black")
        ax.tick_params(axis="both", which="major", labelsize=7 * 2, width=1, length=6)

        fig = plt.gcf()
        column_width = 3.3374
        fig_width = column_width * 2
        fig_height = column_width * 0.475 * 2
        fig.set_size_inches(fig_width, fig_height)
        plt.tight_layout(pad=0)

        plt.savefig(
            os.path.join(output_dir, f"benchmarks_combined_sf_{measurement_type}_{scale}.pdf"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.01,
        )
        plt.close()

        for measurement in values.Measurement.unique():
            pl_data = values[values.Measurement == measurement].copy()
            if measurement == "Discovery overhead" and measurement_type == "abs":
                pl_data.y = pl_data.y * 1000 / discovery_visualization_factor
            sns.lineplot(
                data=pl_data,
                x="x",
                y="y",
                style="Benchmark",
                markers=markers[:3],
                markersize=8,
                hue="Benchmark",
                dashes=False,
                palette=base_palette[: len(benchmarks)],
            )

            ax = plt.gca()
            if scale == "symlog":
                thresh = 1 if measurement == "Latency improvement" else 0.01
                ax.set_yscale("symlog", linthresh=thresh)
            else:
                ax.set_yscale(scale)
            max_lim = plt.ylim()[1]
            if scale != "log":
                if measurement_type == "abs" and measurement == "Latency improvement" and max(pl_data.y) < 123:
                    max_lim = min(max_lim, 124.99)
                    ax.yaxis.set_major_locator(FixedLocator(list(range(0, int(max_lim), 20))))
                plt.ylim((0, max_lim))
            if measurement_type == "rel" and measurement == "Latency improvement":
                plt.ylim((0, 1.5 * max_lim))

            ax.xaxis.set_major_locator(FixedLocator([1] + list(range(20, 101, 20))))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x, 1)))

            y_label = measurement
            time_unit = " [s]" if measurement == "Latency improvement" else " [ms]"
            y_label += time_unit if measurement_type == "abs" else " [%]"
            plt.ylabel(y_label, fontsize=8 * 2)
            plt.xlabel("Scale factor", fontsize=8 * 2)
            plt.legend(fontsize=7 * 2, fancybox=False, framealpha=1.0, edgecolor="black")
            plt.grid(axis="y", visible=True)
            plt.grid(axis="x", visible=True)
            ax.tick_params(axis="both", which="major", labelsize=7 * 2, width=1, length=6, left=True, bottom=True)

            fig = plt.gcf()
            column_width = 3.3374
            fig_width = column_width * 0.475 * 2
            fig_height = column_width * 0.475 * 2
            fig.set_size_inches(fig_width, fig_height)

            plt.tight_layout(pad=0)
            # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: format_number(x, discovery_visualization_factor)))

            measurement_name = measurement.replace(" ", "_").lower()
            plt.savefig(
                os.path.join(output_dir, f"benchmarks_combined_sf_{measurement_type}_{measurement_name}_{scale}.pdf"),
                dpi=300,
                bbox_inches="tight",
                pad_inches=0.01,
            )
            plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.commit, args.data, args.output, args.scale)
