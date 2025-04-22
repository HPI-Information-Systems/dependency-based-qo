#!/usr/bin/env python3.11

import argparse as ap
import json
import os
from statistics import geometric_mean

import numpy as np


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


def get_old_new_latencies(old_path, new_path):
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
        # name = old["name"] if "TPCH" not in old_path else old["name"][len("TPC-H ") :]

        # Create numpy arrays for old/new successful/unsuccessful runs from benchmark dictionary
        old_successful_durations = np.array([run["duration"] for run in old["successful_runs"]], dtype=np.float64)
        new_successful_durations = np.array([run["duration"] for run in new["successful_runs"]], dtype=np.float64)
        # np.mean() defaults to np.float64 for int input
        old_latencies.append(np.mean(old_successful_durations))
        new_latencies.append(np.mean(new_successful_durations))

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

    for benchmark in benchmarks:
        common_path = f"hyriseBenchmark{benchmark}_{commit1}_st"
        if benchmark != "JoinOrder":
            common_path = common_path + "_s10"
        old_path = os.path.join(data_dir, common_path + "_schema.json")
        base_path = os.path.join(data_dir, common_path + "_all_off.json")
        new_path = os.path.join(data_dir, common_path + "_plugin.json")

        print(benchmark)

        old_latencies, new_latencies = get_old_new_latencies(old_path, new_path)
        base_latencies, _ = get_old_new_latencies(base_path, base_path)
        speedups = list()
        latencies = list()
        latencies_base = list()
        speedups_base = list()
        for o, n, b in zip(old_latencies, new_latencies, base_latencies):
            diff = n / o
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
        print(f"\t# {len(speedups)} / {len(speedups_base)}")
        print(f"""\tTop 5 speedups {[float(round(s, 1)) for s in speedups[:min(len(speedups), 5)]]}""")
        print(f"""\tTop 5 latencies {[float(round(lat * 100, 1)) for lat in latencies[:min(len(latencies), 5)]]}""")
        print(f"""\tTop 5 speedups (base) {[float(round(s, 1)) for s in speedups_base[:min(len(speedups_base), 5)]]}""")
        print(
            "\tTop 5 latencies (base)",
            [float(round(lat * 100, 1)) for lat in latencies_base[: min(len(latencies_base), 5)]],
        )
        print(f"\tGeomean {geometric_mean([s - 1 for s in speedups]) * 100}")
        print(f"\tGeomean (base) {geometric_mean([s * 100 - 100 for s in speedups_base])}")

        print("")
        print(f"\tMean {np.mean([s - 1 for s in speedups]) * 100}")

        print("")
        print(f"\tMedian {np.median([s - 1 for s in speedups]) * 100}")

        print()

    print("PRUNING")
    old_path = os.path.join(data_dir, f"hyriseBenchmarkJoinOrder_{commit2}_no_pruning_st_plugin.json")
    new_path = os.path.join(data_dir, f"hyriseBenchmarkJoinOrder_{commit1}_st_plugin.json")
    old_latencies, new_latencies = get_old_new_latencies(old_path, new_path)
    speedups = list()
    latencies = list()
    for o, n in zip(old_latencies, new_latencies):
        diff = n / o
        if diff > 0.95:
            continue
        speedup = o / n
        speedups.append(speedup)
        latencies.append(1 - n / o)
    speedups.sort(reverse=True)
    latencies.sort(reverse=True)
    print(f"\t# {len(speedups)}")
    print(f"""\tTop 5 speedups {[float(round(s, 1)) for s in speedups[:min(len(speedups), 100)]]}""")
    print(f"""\tTop 5 latencies {[float(round(lat * 100, 1)) for lat in latencies[:min(len(latencies), 5)]]}""")

    print(f"\tGeomean {geometric_mean([s - 1 for s in speedups]) * 100}")


if __name__ == "__main__":
    args = parse_args()
    main(args.commit1, args.commit2, args.data)
