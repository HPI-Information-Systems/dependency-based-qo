# Unleashing the Power of Data Dependencies for Query Optimization With Efficient Discovery and System Integration

This repository contains the artifacts for the paper _
Unleashing the Power of Data Dependencies for Query Optimization With Efficient Discovery and System Integration_.

## Reproduction Guide

We listed all steps required to compile the DBMS code and execute all experiments in [`reproduction.sh`](reproduction.sh). See this file for details, or execute it as is.

Reproducing all results will require multiple days. The script is expected to run on a recent Ubuntu version (tested on 24.04). See the usage in the following:
```
./reproduction.sh [NUMA_NODE] [CLIENTS]
```
- `NUMA_NODE` is the NUMA node ID to bind the experiments to. Defaults to 0.
- `CLIENTS` is the number of clients to use for the high-load experiments. Defaults to the number of cores avaible on NUMA node `NUMA_NODE` * 0.6.

The script calls all reproduction scripts in `reproduction`:
- [`install.sh`](reproduction/install.sh) loads the subdirectories, compiles the DBMSs, and installs them.
- [`experiments_hyrise.sh`](reproduction/experiments_hyrise.sh) executes the experiments for dependency-based optimizations in Hyrise.
- [`experiments_systems.sh`](reproduction/experiments_systems.sh) executes the throughput experiments for different DBMSs.
- [`experiments_naive_validation.sh`](reproduction/experiments_naive_validation.sh) runs the naive dependency validation as a baseline for metadata-aware techniques.
- [`create_plots.sh`](reproduction/create_plots.sh) creates all plots.

## Repository Structure

The `hyrise` submodule imports the adapted version of Hyrise for dependency-based query optimization.
- The presented query rewrites are implemented as optimizer rules, found in `hyrise/src/lib/optimizer/strategy`. The relevant implementations are:
  - O-1: `dependent_group_by_reduction_rule.[c|h]pp`
  - O-2: `join_to_semi_join_rule.[c|h]pp`
  - O-3: `join_to_predicate_rewrite_rule.[c|h]pp`
- `hyrise/src/plugins/dependency_discovery_plugin.[c|h]pp` contains the dependency discovery plug-in. The implementation is further split.
  - The `candidate_strategy` subdirectory contains the candidate generation rules.
  - The `validation_strategy` subdirectory contains the metadata-aware dependency validation algorithms.
- The `hyrise/scripts` folder contains various scripts, e.g., for benchmarking Hyrise.
  - `benchmark_single_optimizations.sh` orchestrates all expriments for the impact of dependency-based optimizations in Hyrise, including dependency discovery times.
  - `benchmark_compare_plugin_sf.sh` runs the experiments for the tradeoff between latency improvements achieved by dependency-based optimizations and the discovery overhead for different scale factors.

The code to run the experiments for dependency-based optimizations on different systems is mostly located in the `python` folder.

- `python/db_comparison_runner.py` executes the experiment that measures the throughput improvement for different DBMSs.


The `resources` directory contains the benchmark schema/create table statements and log files.
Python scripts for visualization and some helpers are located in `scripts`.
