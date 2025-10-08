#!/bin/bash

set -e

for param in "$@"
do
  if [ "$param" == "--help" ] || [ "$param" == "-h" ]; then
    echo 'Usage: ./reproduction.sh [NUMA_NODE] [CLIENTS]'
    echo '  NUMA_NODE is the NUMA node ID to bind the experiments to. Defaults to 0.'
    echo '  CLIENTS is the number of clients to use for the high-load experiments. Defaults to the number of cores'
    echo '          avaible on NUMA node NUMA_NODE * 0.6.'
    exit 1
  fi
done

# Load submodules, install dependencies, build and install DBMS binaries.
if [[ -z $SKIP_INSTALL ]]; then
  echo "Install dependencies."
  ./reproduction/install.sh
else
  echo "Skipping installation."
fi

if [ $# -gt 0 ]; then
  node_id=$1
else
  node_id="0"
fi

if ! which numactl > /dev/null || ! numactl -s > /dev/null; then
  num_cpu=$(nproc)
  node_id="-1"
else
  num_cpu=$(numactl --hardware | grep "node[[:space:]]${node_id}[[:space:]]cpus" | tail -c +"$(echo "node ${node_id} cpus: " | wc -c)" | wc -w)
fi


num_clients=$((num_cpu * 3 / 5))
if [ $# -eq 2 ]; then
  num_clients=$2
fi

# Execute experiments, limited to chosen NUMA node.
./reproduction/experiments_hyrise.sh "$node_id"
./reproduction/experiments_systems.sh "$node_id" "$num_cpu" "$num_clients"
./reproduction/experiments_naive_validation.sh "$node_id"

# Create plots.
./reproduction/create_plots.sh
