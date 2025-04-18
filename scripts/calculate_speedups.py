#!/usr/bin/env python3.11

import argparse as ap
import json
import os
import sys
from statistics import geometric_mean

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter
from palettable.cartocolors.qualitative import Safe_6

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + f"{os.sep}..")
from python.queries import static_job_queries, static_ssb_queries, static_tpcds_queries, static_tpch_queries


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("commit1", type=str)
    parser.add_argument("commit2", type=str)
    parser.add_argument("--data", "-d", type=str, default="./hyrise/cmake-build-release/benchmark_plugin_results")
    return parser.parse_args()


def format_number(n):
    return str(int(n)) if n % 1 == 0 else str(n)


def to_s(lst):
    return [x / 10**9 for x in lst]


def get_old_new_latencies(old_path, new_path, verbose=False):  #  items=None):
    with open(old_path) as old_file:
        old_data = json.load(old_file)

    with open(new_path) as new_file:
        new_data = json.load(new_file)

    if old_data["context"]["benchmark_mode"] != new_data["context"]["benchmark_mode"]:
        exit("Benchmark runs with different modes (ordered/shuffled) are not comparable")

    old_latencies = list()
    new_latencies = list()

    for old, new in zip(old_data["benchmarks"], new_data["benchmarks"]):
        assert old["name"] == new["name"]
        name = old["name"] if "TPCH" not in old_path else old["name"][len("TPC-H ") :]
        # if items and name not in items:
        #     # print(old["name"])
        #     continue

        # Create numpy arrays for old/new successful/unsuccessful runs from benchmark dictionary
        old_successful_durations = np.array([run["duration"] for run in old["successful_runs"]], dtype=np.float64)
        new_successful_durations = np.array([run["duration"] for run in new["successful_runs"]], dtype=np.float64)
        # np.mean() defaults to np.float64 for int input
        old_latencies.append(np.mean(old_successful_durations))
        new_latencies.append(np.mean(new_successful_durations))

        if verbose and (np.mean(new_successful_durations) / np.mean(old_successful_durations)) <= 0.95:
            print(name)

    return old_latencies, new_latencies


def get_trend(old, new):
    diff = new / old
    if diff <= 0.95:
        return "better"
    elif diff >= 1.05:
        return "worse"
    return "same"


def main(commit1, commit2, data_dir):
    benchmarks = ["TPCH", "TPCDS", "JoinOrder", "StarSchema"]
    base_palette = Safe_6.hex_colors

    queries_o3 = {
        "TPCH": set(static_tpch_queries.queries_o3.keys()),
        "TPCDS": set(static_tpcds_queries.queries_o3.keys()),
        "JoinOrder": set(static_job_queries.queries_o3.keys()),
        "StarSchema": set(static_ssb_queries.queries_o3.keys()),
    }

    for benchmark in benchmarks:
        common_path = f"hyriseBenchmark{benchmark}_{commit1}_st"
        if benchmark != "JoinOrder":
            common_path = common_path + "_s10"
        old_path = os.path.join(data_dir, common_path + "_schema.json")
        base_path = os.path.join(data_dir, common_path + "_all_off.json")
        new_path = os.path.join(data_dir, common_path + "_plugin.json")

        print(benchmark)  # , queries_o3[benchmark])

        old_latencies, new_latencies = get_old_new_latencies(old_path, new_path)  # , queries_o3[benchmark])
        base_latencies, _ = get_old_new_latencies(base_path, base_path)
        speedups = list()
        latencies = list()
        latencies_base = list()
        speedups_base = list()
        for o, n, b in zip(old_latencies, new_latencies, base_latencies):
            diff = n / o
            # if diff > 0.95 and diff < 1.05:
            # if diff > 0.95 and diff < 1.05:
            #     continue
            latencies.append(1 - diff)
            latencies_base.append(1 - n / b)
            if diff <= 0.95:
                speedup = o / n
                speedups.append(speedup)
            if n / b <= 0.95:
                speedups_base.append(b / n)
        speedups.sort(reverse=True)
        latencies.sort(reverse=True)
        latencies_base.sort(reverse=True)
        speedups_base.sort(reverse=True)
        # print(speedups)
        print(f"\t# {len(speedups)} / {len(speedups_base)}")
        print(f"""\tTop 5 speedups {[float(round(s, 1)) for s in speedups[:min(len(speedups), 5)]]}""")
        print(f"""\tTop 5 latencies {[float(round(l * 100, 1)) for l in latencies[:min(len(latencies), 5)]]}""")
        print(f"""\tTop 5 speedups (base) {[float(round(s, 1)) for s in speedups_base[:min(len(speedups_base), 5)]]}""")
        print(
            f"""\tTop 5 latencies (base) {[float(round(l * 100, 1)) for l in latencies_base[:min(len(latencies_base), 5)]]}"""
        )
        # print(speedups)
        # print(f"\t{geometric_mean([s for s in speedups]) * 100 - 100}")
        print(f"\tGeomean {geometric_mean([s - 1 for s in speedups]) * 100}")
        # print(f"\t{geometric_mean([s * 100 for s in speedups]) - 100}")
        print(f"\tGeomean {geometric_mean([s * 100 - 100 for s in speedups])}")
        print(f"\tGeomean (base) {geometric_mean([s * 100 - 100 for s in speedups_base])}")

        print("")
        # print(f"\t{np.mean([s for s in speedups]) * 100 - 100}")
        print(f"\tMean {np.mean([s - 1 for s in speedups]) * 100}")
        # print(f"\t{np.mean([s * 100 for s in speedups]) - 100}")
        print(f"\tMean {np.mean([s * 100 - 100 for s in speedups])}")

        print("")
        # print(f"\t{np.median([s for s in speedups]) * 100 - 100}")
        print(f"\tMedian {np.median([s - 1 for s in speedups]) * 100}")
        # print(f"\t{np.median([s * 100 for s in speedups]) - 100}")
        print(f"\tMedian {np.median([s * 100 - 100 for s in speedups])}")

        print([n / o for n, o in zip(new_latencies, old_latencies) if n / o >= 1.05])

        print()

    print("PRUNING")
    old_path = os.path.join(data_dir, f"hyriseBenchmarkJoinOrder_{commit2}_no_pruning_st_plugin.json")
    new_path = os.path.join(data_dir, f"hyriseBenchmarkJoinOrder_{commit1}_st_plugin.json")
    old_latencies, new_latencies = get_old_new_latencies(old_path, new_path)  # , True) #, queries_o3[benchmark])
    speedups = list()
    latencies = list()
    for o, n in zip(old_latencies, new_latencies):
        diff = n / o
        # if diff > 0.95 and diff < 1.05:
        # if diff > 0.95 and diff < 1.05:
        #     continue
        if diff > 0.95:
            continue
        speedup = o / n
        # if speedup <= 1:
        #     continue
        speedups.append(speedup)
        latencies.append(1 - n / o)
    speedups.sort(reverse=True)
    latencies.sort(reverse=True)
    # print(speedups)
    print(f"\t# {len(speedups)}")
    print(f"""\tTop 5 speedups {[float(round(s, 1)) for s in speedups[:min(len(speedups), 100)]]}""")
    print(f"""\tTop 5 latencies {[float(round(l * 100, 1)) for l in latencies[:min(len(latencies), 5)]]}""")

    # print(speedups)
    # print(f"\t{geometric_mean([s for s in speedups]) * 100 - 100}")
    print(f"\tGeomean {geometric_mean([s - 1 for s in speedups]) * 100}")
    # print(f"\t{geometric_mean([s * 100 for s in speedups]) - 100}")
    print(f"\tGeomean {geometric_mean([s * 100 - 100 for s in speedups])}")


if __name__ == "__main__":
    args = parse_args()
    main(args.commit1, args.commit2, args.data)
