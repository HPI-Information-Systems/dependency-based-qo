#!/bin/bash

set -e

if [ $# -ne 3 ]; then
  echo 'Usage: ./experiments_systems.sh NUMA_NODE NUM_CPU NUM_CLIENTS'
  echo '  NUMA_NODE is the NUMA node ID to bind the experiments to.'
  echo '  NUM_CPU is the number of logical cores the DBMSs can use.'
  echo '  NUM_CLIENTS is the number of clients that query the DBMSs.'
  exit 1
fi

node_id=$1
num_cpu=$2
num_clients=$3

numactl_command="numactl -N ${node_id}"
no_numa=""

if [ "$node_id" -eq "-1" ]; then
  numactl_command=""
  no_numa="--no_numactl"
  docker_cpuset=""
else
  cpus=$(numactl -H --cpu-compress | grep "node[[:space:]]${node_id}[[:space:]]cpus" | tail -c +"$(echo "node ${node_id} cpus: " | wc -c)" | tr -d ' ')
  num_len=$(echo "${cpus}" | grep -o -e "([0-9]\+)" | wc -c)
  cpus=$(echo "${cpus}" | head -c -"${num_len}")
  docker_cpuset="--cpuset-cpus ${cpus} --cpuset-mems ${node_id}"
fi

# Throughput for different systems using no optimizations, external rewrites, and the internal optimizer. Expected to
# take ca. 8h. We expect this to be bound to one NUMA node and the scripts should be limited using numactl. In our
# experiments, we used 32 clients and 56 logical cores on one NUMA node.
$numactl_command ./python/db_comparison_runner.py hyrise --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}"
$numactl_command ./python/db_comparison_runner.py hyrise --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}" --rewrites
$numactl_command ./python/db_comparison_runner.py hyrise --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}" --schema_keys
$numactl_command ./python/db_comparison_runner.py hyrise --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}" --rewrites --schema_keys
$numactl_command ./python/db_comparison_runner.py hyrise-int --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}"

rm -rf db_comparison_data/monetdb/data
mkdir -r db_comparison_data/monetdb/data
$numactl_command ./python/db_comparison_runner.py monetdb --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}"
$numactl_command ./python/db_comparison_runner.py monetdb --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" --skip_data_loading "${no_numa}" --rewrites
$numactl_command ./python/db_comparison_runner.py monetdb --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" --skip_data_loading "${no_numa}" --schema_keys
$numactl_command ./python/db_comparison_runner.py monetdb --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" --skip_data_loading "${no_numa}" --rewrites --schema_keys
rm -rf db_comparison_data/monetdb/data

rm -rf db.duckdb
$numactl_command ./python/db_comparison_runner.py duckdb --clients "${num_clients}" --cores "${num_cpu}"
$numactl_command ./python/db_comparison_runner.py duckdb --clients "${num_clients}" --cores "${num_cpu}" --skip_data_loading --rewrites;
$numactl_command ./python/db_comparison_runner.py duckdb --clients "${num_clients}" --cores "${num_cpu}" --schema_keys
$numactl_command ./python/db_comparison_runner.py duckdb --clients "${num_clients}" --cores "${num_cpu}" --skip_data_loading --schema_keys --rewrites
rm -rf db.duckdb

rm -rf db_comparison_data/umbra/*
mkdir -r db_comparison_data/umbra
sudo systemctl start docker docker.socket
sudo docker run -v "$(pwd)"/db_comparison_data/umbra:/var/db -v "$(pwd)":"$(pwd)" -p 5432:5432 "${docker_cpuset}" --name umbra-bench -d umbradb/umbra:25.01
$numactl_command ./python/db_comparison_runner.py umbra --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}"
$numactl_command ./python/db_comparison_runner.py umbra --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}" --rewrites
docker stop umbra-bench

# Delete all data and make sure to get a new DB because we cannot add/drop constraints with Umbra
rm -rf db_comparison_data/umbra/*
mkdir -r db_comparison_data/umbra
docker system prune -fa

sudo docker run -v "$(pwd)"/db_comparison_data/umbra:/var/db -v "$(pwd)":"$(pwd)" -p 5432:5432 "${docker_cpuset}" --name umbra-bench -d umbradb/umbra:25.01
$numactl_command ./python/db_comparison_runner.py umbra --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}" --schema_keys
$numactl_command ./python/db_comparison_runner.py umbra --clients "${num_clients}" --cores "${num_cpu}" -m "${node_id}" "${no_numa}" --rewrites --schema_keys

docker stop umbra-bench
rm -rf db_comparison_data/umbra/*
mkdir -r db_comparison_data/umbra
docker system prune -fa
sudo systemctl stop docker docker.socket
