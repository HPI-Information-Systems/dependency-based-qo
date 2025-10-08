#!/usr/bin/env python3

import argparse as ap
import os
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from palettable.cartocolors.qualitative import Safe_10


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="./db_comparison_results")
    parser.add_argument("--output", "-o", type=str, default="./figures")
    parser.add_argument("--metric", "-m", type=str, default="runtime", choices=["throughput", "runtime"])
    return parser.parse_args()


def grep_throughput_change(old_result_file, new_result_file, clients, runtime):
    if not (os.path.isfile(old_result_file) and os.path.isfile(new_result_file)):
        return 0
    df_old = pd.read_csv(old_result_file)
    df_new = pd.read_csv(new_result_file)

    df_old = df_old[df_old.CLIENTS == clients]
    df_new = df_new[df_new.CLIENTS == clients]

    if "hana" in old_result_file:
        df_old = df_old[df_old.RUNTIME_MS > 1000]
        df_new = df_new[df_new.RUNTIME_MS > 1000]

    old_throughput = runtime / (df_old["RUNTIME_MS"].median() / 1000)
    new_throughput = runtime / (df_new["RUNTIME_MS"].median() / 1000)

    return new_throughput / old_throughput * 100 - 100


def get_offsets(key, changes, factor):
    num_configs = max(len(c.keys()) for c in changes.values())
    step_size = 2 / (num_configs - 1)
    num_entries = len(changes[key].keys())
    diff = num_configs - num_entries

    base_offsets = np.arange(-1, 1.001, step_size)
    if diff % 2 != 0:
        base_offsets = np.arange(-1 + (step_size / 2), 1.001 - (step_size / 2), step_size)
        num_configs = len(base_offsets)
        diff = num_configs - num_entries
    return [o * factor for o in base_offsets[(diff // 2) : num_configs - diff // 2]]


def grep_runtime_change(old_result_file, new_result_file, clients, runtime):
    if not (os.path.isfile(old_result_file) and os.path.isfile(new_result_file)):
        return 0
    df_old = pd.read_csv(old_result_file)
    df_new = pd.read_csv(new_result_file)

    df_old = df_old[df_old.CLIENTS == clients]
    df_new = df_new[df_new.CLIENTS == clients]

    if "hana" in old_result_file:
        df_old = df_old[df_old.RUNTIME_MS > 1000]
        df_new = df_new[df_new.RUNTIME_MS > 1000]

    old_runtime = df_old["RUNTIME_MS"].median()
    new_runtime = df_new["RUNTIME_MS"].median()

    return 100 - (new_runtime / old_runtime) * 100


def grep_runtime(result_file, clients):
    if not os.path.isfile(result_file):
        return 0
    df = pd.read_csv(result_file)

    df = df[df.CLIENTS == clients]

    if "hana" in result_file:
        df = df[df.RUNTIME_MS > 1000]
    return df["RUNTIME_MS"].median()


def main(data_dir, output_dir, metric):
    clients = 32
    runtime = 7200
    order = list(reversed(["hyrise-int", "hyrise", "hana-int", "hana", "umbra", "monetdb", "duckdb", "greenplum"]))[1:]
    changes = defaultdict(dict)
    HANA_NAME = "System X"
    HANA_NAME = "SAP HANA"
    min_lim = -10

    for benchmark in ["all"]:  # , "TPCH", "TPCDS", "SSB", "JOB"]:
        print(f"\n\n{benchmark}")

        for dbms in order:
            common_path = f"database_comparison__{benchmark}__{dbms}"
            base_path = os.path.join(data_dir, common_path + ".csv")
            rewrites_path = os.path.join(data_dir, common_path + "__rewrites.csv")
            keys_path = os.path.join(data_dir, common_path + "__keys.csv")
            rewrites_keys_path = os.path.join(data_dir, common_path + "__rewrites__keys.csv")
            method = grep_throughput_change if metric == "throughput" else grep_runtime_change
            if dbms == ("hyrise-int"):
                optimizer_path = base_path
                base_path = os.path.join(data_dir, common_path[: -len("-int")] + ".csv")
                changes[dbms[: -len("-int")]]["optimizer"] = method(base_path, optimizer_path, clients, runtime)
                # changes[dbms[:-len("-int")]]["opt_rewrites"] = method(base_path, rewrites_path, clients, runtime)
                continue
            print(dbms, grep_runtime(base_path, clients) / 1000 / 60)

            changes[dbms]["rewrites"] = method(base_path, rewrites_path, clients, runtime)
            changes[dbms]["keys"] = method(base_path, keys_path, clients, runtime)
            changes[dbms]["rewrites_keys"] = method(base_path, rewrites_keys_path, clients, runtime)
        order = [d for d in order if not d.endswith("-int")]

        print(metric.upper())
        max_len = max([len(s) for s in changes["hyrise"].keys()])
        for dbms in order:
            print(dbms.title())
            for c, v in changes[dbms].items():
                print(f"    {c.rjust(max_len)}: {round(v, 2)}%")

        names = {
            "hyrise": "Hyrise",
            "monetdb": "MonetDB",
            "umbra": "Umbra",
            "hana": HANA_NAME,
            "greenplum": "Greenplum",
            "greenplum-rows": "Greenplum\n(row)",
            "duckdb": "DuckDB",
        }

        changes["hyrise"].pop("optimizer")

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

        base_palette = Safe_10.hex_colors
        configs = ["keys", "rewrites", "rewrites_keys", "optimizer"]  # , "opt_rewrites"]
        colors = {c: base_palette[i] for i, c in enumerate(configs)}

        group_centers = np.arange(len(order))
        offsets = {d: get_offsets(d, changes, 1.5) for d in changes.keys()}
        bar_width = 0.225
        margin = 0
        margin = 0.05

        labels = {
            "keys": r"PKs \& FKs",
            "rewrites": "SQL rewrites",
            "rewrites_keys": r"PKs \& FKs, SQL rewrites",
            "optimizer": r"Dependency optimizer",
            "opt_rewrites": "Optimizer, SQL rewrites",
        }

        ax = plt.gca()
        max_val = max([max(d.values()) for d in changes.values()])
        min_val = min([min(d.values()) for d in changes.values()])
        for offset_id, config in enumerate(configs[:3]):
            bar_positions = [
                p + offsets[d][offset_id] * (0.5 * bar_width + margin) for d, p in zip(order, group_centers)
            ]
            data = [changes[d][config] for d in order]
            ax.bar(bar_positions, data, bar_width, color=colors[config], label=labels[config], edgecolor="none")
            for pos, val in zip(bar_positions, data):
                offset = max_val / 100
                y_pos = val - offset if val > 0 else max(min_lim, val) + offset
                va = "top" if val > 0 else "bottom"
                color = "white"
                if abs(val) < 3.5:
                    va = "top" if val < 0 else "bottom"
                    color = "black"
                    y_pos = val + offset if val > 0 else max(min_lim, val) - offset
                label = str(round(val, 1)) if abs(val) > 0.2 else str(round(val))
                ax.text(pos, y_pos, label, ha="center", va=va, size=7 * 2, rotation=90, color=color)

        ax.set_ylim(None if min_val > min_lim else min_lim, max_val * 1.3)

        plt.xticks(group_centers, [names[d] for d in order], rotation=0)
        ax = plt.gca()
        metric_title = metric if metric != "runtime" else "execution time".capitalize()
        plt.ylabel(f"{metric_title}\nimprovement [\\%]", fontsize=8 * 2)
        ax.tick_params(axis="both", which="major", labelsize=7 * 2, width=1, length=6, left=True, bottom=True)

        plt.grid(axis="y", visible=True)
        plt.legend(
            loc="best",
            fontsize=6 * 2,
            ncol=3,
            fancybox=False,
            framealpha=1.0,
            columnspacing=1.0,
            labelspacing=0.25,
            handlelength=1.5,
            handletextpad=0.4,
            edgecolor="black",
        )

        fig = plt.gcf()
        column_width = 3.3374
        fig_width = column_width * 2
        fig_height = column_width * 0.475 * 2 * 0.9
        fig.set_size_inches(fig_width, fig_height)
        plt.tight_layout(pad=0)

        plt.savefig(
            os.path.join(output_dir, f"systems_comparison_{benchmark.lower()}_{metric}.pdf"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.01,
        )
        plt.close()


if __name__ == "__main__":
    args = parse_args()
    main(args.data, args.output, args.metric)
