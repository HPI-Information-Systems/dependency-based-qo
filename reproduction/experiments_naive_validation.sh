#!/bin/bash

set -e

if [ $# -ne 1 ]; then
  echo 'Usage: ./experiments_naive_validation.sh NUMA_NODE'
  echo '  NUMA_NODE is the NUMA node ID to bind the experiments to.'
  exit 1
fi

node_id=$1

# Validation times for naive dependency discovery.
numactl -N "${node_id}" -m "${node_id}" PERFORM_ABLATION=1 SCHEMA_CONSTRAINTS=0 VALIDATION_LOOPS=100 ./cmake-build-release/hyriseBenchmarkTPCH \
    -r 0 -p ./cmake-build-release/lib/libhyriseDependencyDiscoveryPlugin.so \
    > cmake-build-release/benchmark_plugin_results/ablation_tpch.log
numactl -N "${node_id}" -m "${node_id}" PERFORM_ABLATION=1 SCHEMA_CONSTRAINTS=0 VALIDATION_LOOPS=100 ./cmake-build-release/hyriseBenchmarkTPCDS \
    -r 0 -p ./cmake-build-release/lib/libhyriseDependencyDiscoveryPlugin.so \
    > cmake-build-release/benchmark_plugin_results/ablation_tpcds.log
numactl -N "${node_id}" -m "${node_id}" PERFORM_ABLATION=1 SCHEMA_CONSTRAINTS=0 VALIDATION_LOOPS=100 ./cmake-build-release/hyriseBenchmarkStarSchema \
    -r 0 -p ./cmake-build-release/lib/libhyriseDependencyDiscoveryPlugin.so \
    > cmake-build-release/benchmark_plugin_results/ablation_ssb.log
numactl -N "${node_id}" -m "${node_id}" PERFORM_ABLATION=1 SCHEMA_CONSTRAINTS=0 VALIDATION_LOOPS=100 ./cmake-build-release/hyriseBenchmarkJoinOrder \
    -r 0 -p ./cmake-build-release/lib/libhyriseDependencyDiscoveryPlugin.so \
    > cmake-build-release/benchmark_plugin_results/ablation_job.log
