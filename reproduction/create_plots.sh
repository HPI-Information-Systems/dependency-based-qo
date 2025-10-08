#!/bin/bash

set -e

mkdir -p figures

python3 ./scripts/plot_validation_time.py "$(cd hyrise && git rev-parse HEAD)"
python3 ./scripts/plot_performance_impact.py "$(cd hyrise && git rev-parse HEAD)"
python3 ./scripts/plot_tradeoff_sf.py "$(cd hyrise && git rev-parse HEAD)"
python3 ./scripts/plot_comparison.py
python3 ./scripts/plot_benchmark_tradeoff.py "$(cd hyrise && git rev-parse HEAD)"
python3 ./scripts/plot_ablation_relative.py
python3 ./scripts/plot_benchmark_tradeoff.py "$(cd hyrise && git rev-parse HEAD)"
