#!/usr/bin/env python3.11

import argparse as ap
import re
from collections import defaultdict


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument("file_name", type=str)
    parser.add_argument("level1", type=str)
    parser.add_argument("level2", type=str)
    return parser.parse_args()


def main(file_name, level1, level2):
    time_regex = re.compile(r"(?<=\()\d+(?=ns\))")
    candidate_regex = re.compile(r"(?<=Checking\s).+(?=\[(skipped|confirmed|rejected))")

    diffs = defaultdict(list)

    current_level = None

    levels = list()

    with open(file_name) as f:
        for line in f:
            if "Perform validation with ablation level" in line:
                if level1 in line:
                    current_level = level1
                    levels.append(level1)
                elif level2 in line:
                    current_level = level2
                    levels.append(level2)
                else:
                    current_level = None
                continue

            if current_level is None:
                continue

            candidate = candidate_regex.search(line)
            if candidate:
                # print(current_level, line)
                candidate = candidate.group()
                time = time_regex.search(line)
                assert time
                time = int(time.group()) / 1000**2
                diffs[candidate].append(time)

    # print(diffs)
    diffs_sorted = list()
    c_len = 0
    for candidate, times in diffs.items():
        d = times[1] - times[0]
        diffs_sorted.append([candidate, times[0], times[1], d])
        if d > 0.01:
            c_len = max(c_len, len(candidate))

    diffs_sorted.sort(key=lambda x: x[3], reverse=True)

    print(levels)
    for c, t1, t2, d in diffs_sorted:
        if round(d, 2) > 0:
            print(c.rjust(c_len), round(t1, 2), round(t2, 2), round(d, 2), sep="\t")


if __name__ == "__main__":
    args = parse_args()
    main(args.file_name, args.level1, args.level2)
