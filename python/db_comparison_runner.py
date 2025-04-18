#!/usr/bin/python3
# Thanks to Markus Dreseler, who initially built this script, and Martin Boissier, who extended it.

import argparse
import atexit
import csv
import json
import os
import random
import re
import socket
import statistics
import struct
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
from helpers import schema_keys
from queries import static_job_queries, static_ssb_queries, static_tpcds_queries, static_tpch_queries

# For a fair comparison, we use the same queries as the Umbra demo does.
tpch_queries = static_tpch_queries.queries
ssb_queries = static_ssb_queries.queries


def query_blacklist(filename):
    blacklist = set()
    with open(filename) as f:
        for line in f:
            if not line.startswith("#"):
                blacklist.add(line.strip())
    return blacklist


def load_queries(dir, blacklist={}):
    queries = {}
    for filename in Path(dir).glob("*.sql"):
        if not filename.is_file() or filename.name in blacklist:
            continue

        with open(str(filename), "r") as sql_file:
            sql_query = sql_file.read()
            queries[filename.stem] = sql_query.strip()
    return queries


job_queries = load_queries("hyrise/third_party/join-order-benchmark", {"fkindexes.sql", "schema.sql"})
tpcds_path = "hyrise/resources/benchmark/tpcds"
tpcds_blacklist = query_blacklist(os.path.join(tpcds_path, "query_blacklist.cfg"))
tpcds_queries = load_queries(os.path.join(tpcds_path, "tpcds-result-reproduction/query_qualification"), tpcds_blacklist)


# gather size information
tables = {
    "JOB": [
        "aka_name",
        "aka_title",
        "cast_info",
        "char_name",
        "comp_cast_type",
        "company_name",
        "company_type",
        "complete_cast",
        "info_type",
        "keyword",
        "kind_type",
        "link_type",
        "movie_companies",
        "movie_info",
        "movie_info_idx",
        "movie_keyword",
        "movie_link",
        "name",
        "person_info",
        "role_type",
        "title",
    ],
    "TPC-H": ["nation", "region", "part", "supplier", "partsupp", "customer", "orders", "lineitem"],
    "TPC-DS": [
        "call_center",
        "catalog_page",
        "catalog_returns",
        "catalog_sales",
        "customer",
        "customer_address",
        "customer_demographics",
        "date_dim",
        "household_demographics",
        "income_band",
        "inventory",
        "item",
        "promotion",
        "reason",
        "ship_mode",
        "store",
        "store_returns",
        "store_sales",
        "time_dim",
        "warehouse",
        "web_page",
        "web_returns",
        "web_sales",
        "web_site",
    ],
    "SSB": ["customer", "date", "lineorder", "part", "supplier"],
}

parser = argparse.ArgumentParser()
parser.add_argument(
    "dbms", type=str, choices=["monetdb", "hyrise", "greenplum", "umbra", "hana", "hana-int", "hyrise-int", "duckdb"]
)
parser.add_argument("--time", "-t", type=int, default=7200)
parser.add_argument("--port", "-p", type=int, default=5432)
parser.add_argument("--clients", type=int, default=1)
parser.add_argument("--cores", type=int, default=1)
parser.add_argument("--memory_node", "-m", type=int, default=2)
parser.add_argument("--benchmark", "-b", type=str, default="all", choices=["TPCH", "TPCDS", "JOB", "SSB", "all"])
parser.add_argument("--hyrise_server_path", type=str, default="hyrise/cmake-build-release")
parser.add_argument("--skip_warmup", action="store_true")
parser.add_argument("--skip_data_loading", action="store_true")
parser.add_argument("--rewrites", action="store_true")
parser.add_argument("--O1", action="store_true")
parser.add_argument("--O3", action="store_true")
parser.add_argument("--rows", action="store_true")
parser.add_argument("--no_numactl", action="store_true")
parser.add_argument("--schema_keys", action="store_true")
args = parser.parse_args()
assert not (args.rewrites and (args.O1 or args.O3)), "--rewrites is shorthand for --O1 --O3"
# assert not (
#     any([args.rewrites, args.O1, args.O3]) and "-int" in args.dbms
# ), "Internal optimization works on original queries"

if args.dbms in ["hyrise", "hyrise-int"]:
    hyrise_server_path = Path(args.hyrise_server_path).expanduser().resolve()
    assert (hyrise_server_path / "hyriseServer").exists(), "Please pass valid --hyrise_server_path"

assert (
    args.clients == 1 or args.time >= 300
), "When multiple clients are set, a shuffled run is initiated, which should last at least 300s."

if args.dbms in ["hyrise", "hyrise-int"]:
    args.skip_data_loading = False


def update_hana_optimized_queries(original_queries, items):
    updated_queries = original_queries.copy()
    hints = [
        "HEX_TABLE_SCAN_SEMI_JOIN",
    ]
    for item in items:
        query = original_queries[item].strip()
        if query.endswith(";"):
            query = query[:-1]
        query += f""" WITH HINT({", ".join(hints)});"""
        updated_queries[item] = query
    return updated_queries


if args.dbms in ["hana", "hana-int"]:
    tpch_queries.update(static_tpch_queries.hana_queries)
    job_queries.update(static_job_queries.hana_queries)
    ssb_queries.update(static_ssb_queries.umbra_queries)

elif args.dbms == "umbra":
    ssb_queries.update(static_ssb_queries.umbra_queries)

if args.rewrites or args.O1:
    tpch_queries.update(static_tpch_queries.queries_o1)
    tpcds_queries.update(static_tpcds_queries.queries_o1)

    if args.dbms in ["hana", "hana-int"]:
        tpch_queries.update(static_tpch_queries.hana_queries_o1)

if args.rewrites or args.O3:
    tpch_queries.update(static_tpch_queries.queries_o3)
    ssb_queries.update(static_ssb_queries.queries_o3)
    job_queries.update(static_job_queries.queries_o3)
    tpcds_queries.update(static_tpcds_queries.queries_o3)

    if args.dbms in ["hana", "hana-int"]:
        tpch_queries.update(static_tpch_queries.hana_queries_o3)
        job_queries.update(static_job_queries.hana_queries_o3)
        ssb_queries.update(static_ssb_queries.umbra_queries_o3)
    elif args.dbms == "umbra":
        ssb_queries.update(static_ssb_queries.umbra_queries_o3)

if args.dbms == "hana-int":
    tpch_queries = update_hana_optimized_queries(tpch_queries, list(static_tpch_queries.queries_o3.keys()))
    tpcds_queries = update_hana_optimized_queries(tpcds_queries, list(static_tpcds_queries.queries_o3.keys()))
    ssb_queries = update_hana_optimized_queries(ssb_queries, list(static_ssb_queries.queries_o3.keys()))
    job_queries = update_hana_optimized_queries(job_queries, list(static_job_queries.queries_o3.keys()))

tpch_queries = [tpch_queries[q] for q in sorted(tpch_queries.keys())]
tpcds_queries = [tpcds_queries[q] for q in sorted(tpcds_queries.keys())]
ssb_queries = [ssb_queries[q] for q in sorted(ssb_queries.keys())]
job_queries = [job_queries[q] for q in sorted(job_queries.keys())]

assert len(tpch_queries) == 22
assert len(tpcds_queries) == 48
assert len(ssb_queries) == 13
assert len(job_queries) == 113


def get_cursor():
    if args.dbms == "monetdb":
        connection = None
        attempts = 0
        while connection is None:
            try:
                attempts += 1
                connection = pymonetdb.connect("", connect_timeout=600, autocommit=True)
            except Exception as e:
                print(e)
                time.sleep(1)
                if attempts > 5:
                    raise e
        connection.settimeout(600)
    elif args.dbms in ["hyrise", "hyrise-int"]:
        connection = psycopg2.connect("host=localhost port={}".format(args.port))
    elif args.dbms == "umbra":
        # connection = psycopg2.connect(host="/tmp", user="postgres")
        connection = psycopg2.connect(host="127.0.0.1", user="postgres", password="postgres")
    elif args.dbms == "greenplum":
        host = socket.gethostname()
        connection = psycopg2.connect(host=host, port=args.port, dbname="dbbench", user="bench", password="password")
    elif args.dbms in ["hana", "hana-int"]:
        with open("resources/database_connection.json", "r") as file:
            connection_data = json.load(file)
        connection = dbapi.connect(
            address=connection_data["host"],
            port=connection_data["port"],
            user=connection_data["db_user"],
            password=connection_data["db_user_password"],
            # encrypt=True,
            sslValidateCertificate=False,
            autocommit=connection_data["autocommit"],
        )
    elif args.dbms == "duckdb":
        connection = duckdb.connect(database="db.duckdb", read_only=False, config=dict(threads=args.cores))

    cursor = connection.cursor()
    return (connection, cursor)


def add_constraints(skip):
    if skip:
        return
    connection, cursor = get_cursor()

    start = time.perf_counter()

    add_pk_command = """ALTER TABLE {} ADD CONSTRAINT comp_pk_{} PRIMARY KEY ({});"""
    constraint_id = 1
    for table_name, column_names in schema_keys.primary_keys:
        print(f"\r- Add PRIMARY KEY constraints ({constraint_id}/{len(schema_keys.primary_keys)})", end="")
        table = f'"{table_name}"' if table_name == "date" else table_name
        cursor.execute(add_pk_command.format(table, constraint_id, ", ".join(column_names)))
        constraint_id += 1
    end = time.perf_counter()
    print(f"\r- Added {len(schema_keys.primary_keys)} PRIMARY KEY constraints ({round(end - start, 1)} s)")
    start = end

    add_fk_command = """ALTER TABLE {} ADD CONSTRAINT comp_fk_{} FOREIGN KEY ({}) REFERENCES {} ({});"""
    constraint_id = 1
    for table_name, column_names, referenced_table, referenced_column_names in schema_keys.foreign_keys:
        print(f"\r- Add FOREIGN KEY constraints ({constraint_id}/{len(schema_keys.foreign_keys)})", end="")
        table = f'"{table_name}"' if table_name == "date" else table_name
        referenced_table_name = f'"{referenced_table}"' if referenced_table == "date" else referenced_table
        try:
            cursor.execute(
                add_fk_command.format(
                    table,
                    constraint_id,
                    ", ".join(column_names),
                    referenced_table_name,
                    ", ".join(referenced_column_names),
                )
            )
        except Exception as e:
            print(f"""\n - Error adding FK {table} ({", ".join(column_names)})""", end=" ")
            print(f"""REFERENCES {referenced_table_name} ({", ".join(referenced_column_names)}):""")
            print(f"    {str(e)}")
            pass

        constraint_id += 1
    end = time.perf_counter()
    print(f"\r- Added {len(schema_keys.foreign_keys)} FOREIGN KEY constraints ({round(end - start, 1)} s)")

    cursor.close()
    if args.dbms in ["umbra", "greenplum"]:
        connection.commit()
    connection.close()


def drop_constraints(skip):
    if skip:
        return
    connection, cursor = get_cursor()

    print("- Drop FOREIGN KEY constraints ...")
    drop_fk_command = """ALTER TABLE {} DROP CONSTRAINT comp_fk_{};"""
    constraint_id = 1

    for table_name, _, _, _ in schema_keys.foreign_keys:
        table = f'"{table_name}"' if table_name == "date" else table_name
        try:
            cursor.execute(drop_fk_command.format(table, constraint_id))
        except Exception:
            pass
        constraint_id += 1

    print("- Drop PRIMARY KEY constraints ...")
    drop_pk_command = """ALTER TABLE {} DROP CONSTRAINT comp_pk_{};"""
    constraint_id = 1

    for table_name, _ in schema_keys.primary_keys:
        table = f'"{table_name}"' if table_name == "date" else table_name
        try:
            cursor.execute(drop_pk_command.format(table, constraint_id))
        except Exception:
            pass
        constraint_id += 1

    cursor.close()
    if args.dbms in ["umbra", "greenplum"]:
        connection.commit()
    connection.close()


dbms_process = None


def cleanup():
    if dbms_process:
        print("Shutting {} down...".format(args.dbms))
        if args.dbms == "hana-int" or args.schema_keys:
            drop_constraints(args.dbms in ["duckdb", "umbra", "hyrise", "hyrise-int"])
        dbms_process.kill()
        time.sleep(10)


atexit.register(cleanup)

print("Starting {}...".format(args.dbms))
numactl_command = ["numactl", "-C", f"+0-+{args.cores - 1}"]
if not args.no_numactl:
    numactl_command += ["-m", str(args.memory_node)]

if args.dbms == "monetdb":
    import pymonetdb

    monetdb_home = os.path.join(os.getcwd(), "db_comparison_data", "monetdb")

    subprocess.Popen(["pkill", "-9", "mserver5"])
    time.sleep(5)
    cmd = numactl_command + [
        "{}/bin/mserver5".format(monetdb_home),
        "--dbpath={}/data".format(monetdb_home),
        "--set",
        "gdk_nr_threads={}".format(args.cores),
    ]

    if args.clients > 62:
        cmd.extend(["--set", f"max_clients={args.clients + 2}"])
    if args.clients < 33:
        # We have seen strange errors when using the inmemory option and 64 clients (bat file not existing)
        cmd.append("--dbextra=inmemory")
    dbms_process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    while True:
        line = dbms_process.stdout.readline()
        if b"MonetDB/SQL module loaded" in line:
            break
elif args.dbms in ["hyrise", "hyrise-int"]:
    import psycopg2

    allow_schema_env = {"JOIN_TO_PREDICATE": "0"} if args.schema_keys and args.dbms != "hyrise-int" else {}
    dbms_process = subprocess.Popen(
        numactl_command
        + [
            "{}/hyriseServer".format(hyrise_server_path),
            "-p",
            str(args.port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=allow_schema_env,
    )
    time.sleep(5)
    while True:
        line = dbms_process.stdout.readline()
        print(line.decode(), end="")
        if b"Server started at" in line:
            break
elif args.dbms == "umbra":
    import psycopg2

    print("Make sure to start Umbra before by starting the Docker container")
    time.sleep(1)
elif args.dbms == "greenplum":
    import psycopg2

    print("Make sure to start Greenplum before by running ./scripts/greenplum_init.sh")
    time.sleep(1)
elif args.dbms in ["hana", "hana-int"]:
    from hdbcli import dbapi
elif args.dbms == "duckdb":
    import duckdb
else:
    raise AttributeError(f"Unknown DBMS: '{args.dbms}'")


def parse_data_type(type_string):
    if type_string == "int":
        return "Int32"
    elif type_string == "long":
        return "Int64"
    elif type_string == "float":
        return "Float32"
    elif type_string == "double":
        return "Float64"
    elif type_string == "string":
        return "string"
    raise AttributeError(f"Unknown data type: '{type_string}'")


def parse_csv_meta(meta):
    column_names = list()
    column_data_types = dict()
    nullable = dict()
    for column_meta in meta["columns"]:
        column_name = column_meta["name"]
        column_names.append(column_name)
        column_data_types[column_name] = parse_data_type(column_meta["type"])
        nullable[column_name] = column_meta["nullable"]
    return column_names, column_data_types, nullable


def import_data():
    data_path = os.path.join(os.getcwd(), "resources/experiment_data")

    if args.dbms == "monetdb":
        load_command = """COPY INTO "{}" FROM '{}' USING DELIMITERS ',', '\n', '"' NULL AS '';"""
    elif args.dbms in ["hyrise", "hyrise-int"]:
        load_command = """COPY "{}" FROM '{}';"""
    elif args.dbms == "umbra":
        load_command = """COPY "{}" FROM '{}' WITH (FORMAT CSV, DELIMITER ',', NULL '', QUOTE '"');"""
    elif args.dbms == "greenplum":
        load_command = """COPY "{}" FROM '{}' WITH (FORMAT CSV, DELIMITER ',', NULL '', QUOTE '"');"""
    elif args.dbms in ["hana", "hana-int"]:
        load_command = (
            """IMPORT FROM CSV FILE '{}' INTO {} WITH FIELD DELIMITED BY ',' ESCAPE '"' FAIL ON INVALID DATA;"""
        )
    elif args.dbms == "duckdb":
        load_command = """COPY "{}" FROM '{}' WITH (FORMAT CSV, DELIMITER ',', NULL '', QUOTE '"');"""

    connection, cursor = get_cursor()
    table_name_regex = re.compile(r'(?<=CREATE\sTABLE\s)"?\w+"?(?=\s*\()', flags=re.IGNORECASE)
    table_order = []
    create_table_statements = []
    print("- Loading data ...")

    for benchmark in ["tpch", "tpcds", "ssb", "job"]:
        if args.benchmark != "all" and args.benchmark.lower() != benchmark:
            continue
        with open(f"resources/schema_{benchmark}.sql") as f:
            for line in f:
                stripped_line = line.strip()
                if not stripped_line:
                    continue
                table_name = table_name_regex.search(stripped_line).group().replace('"', "")
                table_order.append(table_name)
                create_table_statements.append(stripped_line)

    for table_name in reversed(table_order):
        if not args.dbms.startswith("hana"):
            cursor.execute(f'DROP TABLE IF EXISTS "{table_name}";')
        else:
            try:
                table = f'"{table_name}"' if table_name == "date" else table_name
                cursor.execute(f"DROP TABLE {table};")
            except Exception as e:
                print("-  Could not drop table {} ({}) - continue".format(table, e))
                pass

    primary_keys = {}
    for table_name, column_names in schema_keys.primary_keys:
        primary_keys[table_name] = column_names
    foreign_keys = defaultdict(dict)
    for table_name, column_names, referenced_table, referenced_column_names in schema_keys.foreign_keys:
        foreign_keys[table_name][referenced_table] = (column_names, referenced_column_names)

    if args.dbms not in ["hyrise", "hyrise-int"]:
        for table_name, create_statement in zip(table_order, create_table_statements):
            if args.dbms == "greenplum":
                create_statement = create_statement[:-1] if create_statement.endswith(";") else create_statement
                if not args.rows:
                    create_statement += " WITH (appendoptimized=true, orientation=column)"
                # Greenplum allows PKs only on the columns used for distributing.
                if table_name in primary_keys:
                    create_statement += f""" DISTRIBUTED BY ({", ".join(primary_keys[table_name])})"""
                create_statement += ";"

            if args.dbms in ["hana", "hana-int"]:
                # change movie_info.info, person_info.info to nclob because their values exceed the nvarchar length
                # limit of 5000 bytes.
                if table_name in ["movie_info", "person_info"]:
                    create_statement = create_statement.replace("info text", "info nclob")  # , count=1)

                create_statement = create_statement.replace("text", "nvarchar(1024)")

            # Umbra does not allow to add constraints later, so we have to do it now.
            if args.dbms in ["umbra", "duckdb"] and args.schema_keys:
                create_statement = create_statement[:-1].strip() if create_statement.endswith(";") else create_statement
                create_statement = create_statement[:-1]
                if table_name in primary_keys:
                    create_statement += f""", PRIMARY KEY ({", ".join(primary_keys[table_name])})"""
                if table_name in foreign_keys:
                    for referenced_table, columns in foreign_keys[table_name].items():
                        column_names, referenced_column_names = columns
                        create_statement += f""", FOREIGN KEY ({", ".join(column_names)}) REFERENCES """
                        create_statement += f""""{referenced_table}" ({", ".join(referenced_column_names)})"""
                create_statement += ");"

            cursor.execute(create_statement)

    if args.dbms in ["hana", "hana-int"]:
        cursor.execute(
            "alter system alter configuration ('indexserver.ini','SYSTEM') set "
            "('import_export','enable_csv_import_path_filter') = 'false' with reconfigure;"
        )

        # We could not manage to load these tuples from CSV.
        cursor.execute(
            "INSERT INTO title VALUES (   9795, 'Null', NULL, 7, NULL, NULL, 'N4', 9785,    1,    11, NULL, '22370d39fa0b1593019c23d5e4ccfca9');"  # noqa: E501
        )
        cursor.execute(
            "INSERT INTO title VALUES (2162886, 'Null', NULL, 1, 2009, NULL, 'N4', NULL, NULL, NULL , NULL, '59cf04844319a809042d47e26ac4074b');"  # noqa: E501
        )
        cursor.execute(
            "INSERT INTO char_name VALUES (590883, 'Null', NULL, NULL, 'N4' , NULL, 'bbb93ef26e3c101ff11cdd21cab08a94');"  # noqa: E501
        )

    # DuckDB v1.2+ cannot deal with double-escaped quotes.
    duckdb_escape_tables = {"company_name", "keyword", "name", "title", "aka_title", "movie_info"}

    for t_id, table_name in enumerate(table_order):
        table_file_path = f"{data_path}/{table_name}.csv"
        binary_file_path = f"{data_path}/{table_name}.bin"
        has_binary = os.path.isfile(binary_file_path)
        if has_binary and args.dbms in ["hyrise", "hyrise-int"]:
            table_file_path = binary_file_path
        print(f" - ({t_id + 1}/{len(table_order)}) Import {table_name} from {table_file_path} ...", end=" ", flush=True)
        start = time.perf_counter()

        if args.dbms == "monetdb" and table_name in tables["JOB"]:
            # MonetDB does not like some IMDB CSV files, so we encode them in their binary format.
            binary_data_types = {"Int32": "<i", "Int64": "<q"}
            with open(table_file_path + ".json") as f:
                meta = json.load(f)
            column_names, column_types, nullable = parse_csv_meta(meta)
            data = pd.read_csv(
                table_file_path, header=None, names=column_names, dtype=column_types, keep_default_na=False
            )
            all_column_files = list()
            for column_name in column_names:
                t_id = 0
                binary_file_path = f"{data_path}/{table_name}.{column_name}.bin"
                all_column_files.append(f"'{binary_file_path}'")

                if not os.path.isfile(binary_file_path):
                    with open(binary_file_path, "wb") as f:
                        if column_types[column_name] == "string":
                            for value in data[column_name].values:
                                t_id = t_id + 1
                                if pd.isna(value):
                                    assert nullable[column_name], f"{column_name} NULL in L{t_id}"
                                    f.write(b"\x80")
                                else:
                                    f.write(value.encode())
                                f.write(b"\x00")
                        else:
                            binary_type = binary_data_types[column_types[column_name]]
                            s = struct.Struct(binary_type)
                            for value in data[column_name].values:
                                t_id = t_id + 1
                                if pd.isna(value):
                                    assert nullable[column_name], f"{column_name} NULL in L{t_id}"
                                    assert column_types[column_name] == "Int32"
                                    f.write(b"\x00\x00\x00\x80")
                                else:
                                    f.write(s.pack(value))

            cursor.execute(
                """COPY LITTLE ENDIAN BINARY INTO "{}" FROM {} ON SERVER;""".format(
                    table_name, ", ".join(all_column_files)
                )
            )

        # elif args.dbms == "duckdb" and table_name in duckdb_escape_tables:
        #     with open(table_file_path + ".json") as f:
        #         meta = json.load(f)
        #     column_names, column_types, nullable = parse_csv_meta(meta)
        #     data = pd.read_csv(
        #         table_file_path, header=None, names=column_names, dtype=column_types, keep_default_na=False
        #     )
        #     new_file_path = f"{data_path}/{table_name}.duckdb.csv"
        #     data.to_csv(new_file_path, sep=",", header=False, index=False, quotechar="|")
        #     # print("""COPY "{}" FROM '{}' WITH (FORMAT CSV, DELIMITER ',', NULL '', QUOTE '|');""".format(table_name, new_file_path))
        #     cursor.execute("""COPY "{}" FROM '{}' WITH (FORMAT CSV, DELIMITER ',', NULL '', QUOTE '|');""".format(table_name, new_file_path))

        elif args.dbms not in ["hana", "hana-int"]:
            cursor.execute(load_command.format(table_name, table_file_path))

        else:
            table = f'"{table_name}"' if table_name == "date" else table_name
            if table_name not in tables["JOB"]:
                try:
                    cursor.execute(load_command.format(table_file_path, table))
                except Exception as e:
                    print("\nFailed to import table {}... with exception {}".format(table_name, e))
                    pass
            else:
                # HANA seems to have issues with some JOB tables, so we rewrite the CSVs with a delimiter that does not
                # occur in any file using pandas. The HANA documentation suggests to use '\u0007'. To be on the safe
                # side, we also use '\u0010' as quoting char because the files contain a lot of double quotes.
                new_file_path = f"{data_path}/{table_name}.hana.csv"
                if (not os.path.isfile(new_file_path)) or True:
                    with open(table_file_path + ".json") as f:
                        meta = json.load(f)
                    column_names, column_types, nullable = parse_csv_meta(meta)
                    data = pd.read_csv(
                        table_file_path, header=None, names=column_names, dtype=column_types, keep_default_na=False
                    )
                    # remove failing tuples that we already inserted via SQL
                    if table_name == "title":
                        data = data[(data.id != 9795) & (data.id != 2162886)]
                    if table_name == "char_name":
                        data = data[data.id != 590883]
                    data.to_csv(
                        new_file_path,
                        sep="\u0007",
                        header=False,
                        index=False,
                        escapechar="+",
                        quoting=csv.QUOTE_NONE,
                        quotechar='"',
                    )

                    data = None
                    with open(new_file_path) as f:
                        data = f.read()
                    data = data.replace('"', '+"').replace("\u0007Null\u0007", '\u0007"Null"\u0007')
                    with open(new_file_path, "w") as f:
                        f.write(data)
                import_statement = "IMPORT FROM CSV FILE '{}' INTO {} WITH FIELD DELIMITED BY '\\u0007' ESCAPE '+' FAIL ON INVALID DATA;"  # noqa: E501
                try:
                    cursor.execute(import_statement.format(new_file_path, table))
                except Exception as e:
                    print("\nFailed to import table {}... with exception {}".format(table_name, e))
                    pass

            try:
                cursor.execute(f"MERGE DELTA OF {table};")
            except Exception as e:
                print("\nCould not merge delta of table {}... with exception {}".format(table_name, e))
                pass

        if not has_binary and args.dbms in ["hyrise", "hyrise-int"]:
            print("and cache as binary ...", end=" ", flush=True)
            cursor.execute(f"""COPY "{table_name}" TO '{binary_file_path}';""")

        end = time.perf_counter()
        print(f"({round(end - start, 1)} s)")

    cursor.close()
    if args.dbms in ["umbra", "greenplum"]:
        connection.commit()
    connection.close()


def split_query(query):
    return [statement for statement in query.split(";") if statement.strip()]


def loop(thread_id, queries, query_id, start_time, successful_runs, timeout, is_warmup=False):
    connection, cursor = get_cursor()

    if is_warmup:
        if args.skip_warmup:
            return

        for q_id, query in enumerate(queries):
            try:
                cursor.execute(query)
                print("({})".format(q_id + 1), end="", flush=True)
            except Exception as e:
                print(e)
                print(query)
                raise e

        cursor.close()
        connection.close()
        return

    while True:
        if query_id == "shuffled":
            items = queries.copy()
            random.shuffle(items)
        else:
            items = [queries[query_id - 1]]
        if args.dbms in ["hana", "hana-int"]:
            split_items = []
            for item in items:
                split_items += split_query(item)
            items = split_items
        item_start_time = time.perf_counter()
        for query in items:
            cursor.execute(query)
            cursor.fetchall()
        item_end_time = time.perf_counter()

        if (item_end_time - start_time < timeout) or len(successful_runs) == 0:
            successful_runs.append((item_end_time - item_start_time) * 1000)
        else:
            break

    cursor.close()
    connection.close()


if args.benchmark == "TPCH":
    selected_benchmark_queries = tpch_queries
elif args.benchmark == "TPCDS":
    selected_benchmark_queries = tpcds_queries
elif args.benchmark == "SSB":
    selected_benchmark_queries = ssb_queries
elif args.benchmark == "JOB":
    selected_benchmark_queries = job_queries
elif args.benchmark == "all":
    selected_benchmark_queries = tpch_queries + tpcds_queries + ssb_queries + job_queries


if args.dbms == "monetdb":
    selected_benchmark_queries = [
        q.replace("!=", "<>")
        .replace("SELECT MIN(chn.name) AS character,", 'SELECT MIN(chn.name) AS "character",')
        .replace("ss_list_price BETWEEN 122 AND 122+10", "ss_list_price BETWEEN 122 AND 122+10.0")
        for q in selected_benchmark_queries
    ]
elif args.dbms == "duckdb":
    # "at" is a reserved keyword in DuckDB
    selected_benchmark_queries = [
        q.replace("aka_title AS at", "aka_title AS akat").replace(" at.", " akat.").replace("(at.", "(akat.")
        for q in selected_benchmark_queries
    ]


drop_constraints(args.dbms in ["duckdb", "umbra", "hyrise", "hyrise-int"])

if not args.skip_data_loading:
    import_data()

if args.schema_keys or args.dbms == "hana-int":
    add_constraints(args.dbms in ["duckdb", "umbra", "hyrise", "hyrise-int"])

if args.dbms in ["monetdb", "umbra", "greenplum", "hyrise-int", "duckdb"] or (
    args.dbms == "hyrise" and args.schema_keys
):
    print("Warming up database (complete single-threaded run) due to initial persistence on disk: ", end="")
    sys.stdout.flush()
    loop(0, selected_benchmark_queries, "warmup", time.perf_counter(), [], 3600, True)
    print(" done.")
    sys.stdout.flush()

if args.dbms == "hyrise-int" or (args.dbms == "hyrise" and args.schema_keys):
    print("Performing dependency discovery", end="")
    sys.stdout.flush()
    connection, cursor = get_cursor()
    cursor.execute(
        """INSERT INTO meta_plugins values ('{}/lib/libhyriseDependencyDiscoveryPlugin.so');""".format(
            hyrise_server_path
        )
    )
    cursor.execute("INSERT INTO meta_exec values ('hyriseDependencyDiscoveryPlugin', 'DiscoverDependencies');")
    print(" done.")
    cursor.close()
    connection.close()

os.makedirs("db_comparison_results", exist_ok=True)

runtimes = {}
benchmark_queries = list(range(1, len(selected_benchmark_queries) + 1))

if args.clients > 1:
    benchmark_queries = ["shuffled"]
for query_id in benchmark_queries:
    query_name = "{} {:02}".format(args.benchmark, query_id) if query_id != "shuffled" else "shuffled"
    print("Benchmarking {}...".format(query_name), end="", flush=True)

    successful_runs = []
    start_time = time.perf_counter()

    timeout = args.time

    threads = []
    for thread_id in range(0, args.clients):
        threads.append(
            threading.Thread(
                target=loop,
                args=(thread_id, selected_benchmark_queries, query_id, start_time, successful_runs, timeout),
            )
        )
        threads[-1].start()

    while True:
        time_left = start_time + timeout - time.perf_counter()
        if time_left < 0:
            break
        print("\rBenchmarking {}... {:.0f} seconds left".format(query_name, time_left), end="")
        time.sleep(1)

    while True:
        joined_threads = 0
        for thread_id in range(0, args.clients):
            if not threads[thread_id].is_alive():
                # print(f't{thread_id} finished')
                joined_threads += 1

        if joined_threads == args.clients:
            break
        else:
            print(
                "\rBenchmarking {}... waiting for {} more clients to finish".format(
                    query_name, args.clients - joined_threads
                ),
                end="",
            )
            time.sleep(1)

    print("\r" + " " * 80, end="")
    print(
        "\r{}\t>>\t avg.: {:10.4f} ms\tmed.: {:10.4f} ms\tmin.: {:10.4f} ms\tmax.: {:10.4f} ms".format(
            query_name,
            sum(successful_runs) / len(successful_runs) if len(successful_runs) > 0 else 0,
            statistics.median(successful_runs) if len(successful_runs) > 0 else 0,
            min(successful_runs) if len(successful_runs) > 0 else 0,
            max(successful_runs) if len(successful_runs) > 0 else 0,
        )
    )

    runtimes[query_name] = successful_runs

row_suffix = "-rows" if args.rows else ""
rewrite_suffix = ""
if args.O1:
    rewrite_suffix += "__O1"
if args.O3:
    rewrite_suffix += "__O3"
if args.rewrites:
    rewrite_suffix += "__rewrites"
if args.schema_keys:
    rewrite_suffix += "__keys"
result_csv_filename = "db_comparison_results/database_comparison__{}__{}{}{}.csv".format(
    args.benchmark, args.dbms, row_suffix, rewrite_suffix
)
result_csv_exists = Path(result_csv_filename).exists()
with open(result_csv_filename, "a" if result_csv_exists else "w") as result_csv:
    if not result_csv_exists:
        result_csv.write("BENCHMARK,DATABASE_SYSTEM,CORES,CLIENTS,ITEM_NAME,RUNTIME_MS\n")
    for item_name, runs in runtimes.items():
        for run in runs:
            result_csv.write(
                "{},{},{},{},{},{}\n".format(args.benchmark, args.dbms, args.cores, args.clients, item_name, run)
            )
