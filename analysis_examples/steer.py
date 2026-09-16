#!/usr/bin/env python3
"""Run Data_Loader.process_run on a path built from a run number.

Usage: steer.py <run_number>
"""
import argparse
import os
import sys
import json
from Analyze_T5 import fit_and_plot_t5

try:
    from Data_Loader import process_run
except Exception as e:
    sys.exit(f"Failed to import process_run from Data_Loader: {e}")


def build_path(run_number: int) -> str:
    # Build a simple path in the current directory named run_<run_number>
    return os.path.abspath(f"/eos/experiment/wcte/data/2025_commissioning/processed_offline_data/production_v1_0/{run_number}/WCTE_merged_production_R{run_number}.root")


def main():
    parser = argparse.ArgumentParser(description="Run processing for a given run number")
    parser.add_argument("-r", "--run", help="Run number to process", type=int)
    args = parser.parse_args()
    particles = ["pions", "muons", "electrons", "protons"]

    for particle in particles:
        output_dir = os.path.join(os.getcwd(), "data", particle)
        os.makedirs(output_dir, exist_ok=True)

    # If a single run was provided on the command line, use it. Otherwise load config and
    # select runs with beam momentum > 700 MeV/c
    runs_to_process = []
    if args.run is not None:
        runs_to_process = [args.run]
    else:
        # Build path to config relative to this script
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "projects", "configs", "all_runs_processed.json"))
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
        except Exception as e:
            sys.exit(f"Failed to load config {config_path}: {e}")

        runs_to_process = []
        if isinstance(cfg, list):
            # If list of runs -> info, use values but keep keys for run number when needed
            for entry in cfg:
                try:
                    run_number = int(entry["run_number"])
                    beam_momentum = float(entry["beam_momentum"])
                except (KeyError, ValueError) as e:
                    print(f"Invalid entry in config {config_path}")
                    continue
                if beam_momentum > 700:
                    runs_to_process.append(run_number)

        else:
            sys.exit(f"Unexpected config format in {config_path}")

    if not runs_to_process:
        sys.exit("No runs to process")

    for run_number in runs_to_process:
        path = build_path(run_number)
        try:
            process_run(run_number, path)
            for particle in particles:
                pq_path = os.path.join(os.getcwd(), "data", particle, f"run{run_number}_T5_{particle}.parquet")
                fit_and_plot_t5(particle, pq_path, run_number)
                print(f"Processed run {run_number} for particle {particle}. Output saved in {pq_path}")

        except Exception as e:
            print(f"process_run failed for run {run_number}: {e}")
            continue

if __name__ == "__main__":
    main()
