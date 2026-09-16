import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import Analyze_T5

import matplotlib.pyplot as plt
import pandas as pd

PARTICLES = ["electrons", "protons", "pions", "muons"]
RUN_FILE_PATTERN = re.compile(
    r"run(?P<run>\d+).*_(?P<particle>electrons|protons|pions|muons)\.parquet$"
)


def find_metadata_path(script_dir: str) -> str:
    candidates = [
        os.path.join(
            script_dir, "..", "..", "projects", "configs", "all_runs_processed.json"
        ),
        os.path.join(
            script_dir,
            "..",
            "..",
            "projects",
            "processed_analysis",
            "configs",
            "all_runs_processed.json",
        ),
    ]
    for candidate in candidates:
        abs_path = os.path.abspath(candidate)
        if os.path.isfile(abs_path):
            return abs_path
    raise FileNotFoundError(
        "Could not find all_runs_processed.json. Checked: "
        + ", ".join(map(os.path.abspath, candidates))
    )


def load_run_metadata(metadata_path: str) -> pd.DataFrame:
    with open(metadata_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Expected metadata file to contain a JSON list of runs")

    df = pd.DataFrame(data)
    if "run_number" not in df.columns:
        raise ValueError("Metadata file does not contain run_number field")

    df["run_number"] = df["run_number"].astype(str).str.strip()
    return df


def collect_parquet_counts(data_root: str) -> pd.DataFrame:
    rows = []
    data_root = os.path.abspath(data_root)
    for particle in PARTICLES:
        particle_dir = os.path.join(data_root, particle)
        if not os.path.isdir(particle_dir):
            continue

        for path in sorted(glob.glob(os.path.join(particle_dir, "*.parquet"))):
            filename = os.path.basename(path)
            match = RUN_FILE_PATTERN.search(filename)
            if not match:
                continue

            run_number = match.group("run")
            particle_name = match.group("particle")

            try:
                df = pd.read_parquet(path)
                n_events = len(df)
            except Exception as exc:
                raise RuntimeError(f"Failed to read {path}: {exc}") from exc

            rows.append(
                {
                    "run_number": run_number,
                    "particle": particle_name,
                    "n_events": n_events,
                    "path": path,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No parquet files found in {data_root}")

    return pd.DataFrame(rows)


def build_run_summary(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    pivot = counts.pivot_table(
        index="run_number",
        columns="particle",
        values="n_events",
        aggfunc="sum",
        fill_value=0,
    )

    for particle in PARTICLES:
        if particle not in pivot.columns:
            pivot[particle] = 0

    pivot = pivot.reset_index()
    pivot.columns.name = None

    if "run_number" in metadata.columns:
        metadata = metadata.drop_duplicates(subset=["run_number"])
        merged = pivot.merge(metadata, on="run_number", how="left")
    else:
        merged = pivot

    for act in [f"act{i}" for i in range(6)]:
        if act not in merged.columns:
            merged[act] = ""

    merged["act_indices"] = (
        merged[[f"act{i}" for i in range(6)]]
        .fillna("")
        .apply(
            lambda row: ", ".join([v for v in row.values if str(v).strip()]),
            axis=1,
        )
    )
    merged["beam_momentum"] = (
        merged.get("beam_momentum", "").astype(str).replace("nan", "")
    )
    merged["total_events"] = merged[PARTICLES].sum(axis=1)
    merged = merged.sort_values(
        by="run_number", key=lambda col: col.astype(int, errors="ignore")
    )
    return merged


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def plot_particle_totals(counts: pd.DataFrame, output_path: str) -> None:
    totals = counts.groupby("particle")["n_events"].sum().reindex(PARTICLES).fillna(0)

    fig, ax = plt.subplots(figsize=(9, 6))
    totals.plot(kind="bar", ax=ax, color=["#4C72B0", "#55A868", "#C44E52", "#1AEADF"])
    ax.set_title("Total Number of Data Points by Particle")
    ax.set_xlabel("Particle")
    ax.set_ylabel("Number of data points")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_xticklabels(PARTICLES, rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_run_comparison(run_summary: pd.DataFrame, output_path: str, x_col: str) -> None:
    if x_col not in run_summary.columns:
        raise ValueError(f"Column {x_col} not found in run summary")

    df = run_summary.copy()
    df = df.sort_values(by="run_number", key=lambda col: col.astype(int, errors="ignore"))
    x_values = df[x_col].astype(str).tolist()
    x_numeric = safe_numeric(df[x_col])

    if x_col == "run_number" or x_numeric.isna().all():
        x_plot = list(range(len(df)))
    else:
        x_plot = x_numeric.tolist()

    fig, ax = plt.subplots(figsize=(14, 7))
    for particle in PARTICLES:
        if particle not in df.columns:
            continue
        ax.plot(x_plot, df[particle].fillna(0), marker="o", label=particle)

    ax.set_title(f"Particle Counts by {x_col.replace('_', ' ').title()}")
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.set_ylabel("Number of data points")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    if x_col == "run_number" or x_numeric.isna().all():
        tick_locations = x_plot[:: max(1, len(x_plot) // 20)]
        tick_labels = x_values[:: max(1, len(x_values) // 20)]
        ax.set_xticks(tick_locations)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    else:
        ax.set_xticks(sorted(set(x_plot))[:20])
        ax.set_xticklabels([str(x) for x in sorted(set(x_plot))[:20]], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_particle_counts_vs_momentum(run_summary: pd.DataFrame, output_path: str) -> None:
    df = run_summary.copy()
    df["beam_momentum_numeric"] = safe_numeric(df.get("beam_momentum"))
    df = df.dropna(subset=["beam_momentum_numeric"])
    if df.empty:
        raise ValueError("No numeric beam_momentum values available for plotting")

    df = df.sort_values(by=["beam_momentum_numeric", "run_number"])
    colors = ["#4C72B0", "#55A868", "#C44E52", "#1AEADF"]

    fig, ax = plt.subplots(figsize=(15, 5))

    for particle, color in zip(PARTICLES, colors):
        if particle not in df.columns:
            continue
        ax.plot(
            df["beam_momentum_numeric"],
            df[particle].fillna(0),
            marker="o",
            linestyle="None",
            color=color,
            label=particle,
        )

    ax.set_yscale("log")
    ax.set_title("Particle Counts by Beam Momentum")
    ax.set_xlabel("Beam Momentum")
    ax.set_ylabel("Number of data points")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def collect_gaussian_sigmas(data_root: str) -> pd.DataFrame:
    rows = []
    data_root = os.path.abspath(data_root)
    for particle in PARTICLES:
        particle_dir = os.path.join(data_root, particle)
        if not os.path.isdir(particle_dir):
            continue

        for path in sorted(glob.glob(os.path.join(particle_dir, "*.parquet"))):
            filename = os.path.basename(path)
            match = RUN_FILE_PATTERN.search(filename)
            if not match:
                continue

            run_number = match.group("run")
            result = Analyze_T5.fit_t5(path)

            if result is None or result["sigma_y"] > 100 or result["sigma_x"] > 100:
                continue

            rows.append(
                {
                    "run_number": int(run_number),
                    "particle": match.group("particle"),
                    "sigma_x": result["sigma_x"],
                    "sigma_y": result["sigma_y"],
                    "path": path,
                }
            )

    if not rows:
        raise FileNotFoundError(f"No successful Gaussian fits found in {data_root}")

    return pd.DataFrame(rows)


def plot_sigma_by_run(sigmas: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#1AEADF"]

    for particle, color in zip(PARTICLES, colors):
        subset = sigmas[sigmas["particle"] == particle].sort_values(
            by="run_number", key=lambda col: col.astype(int, errors="ignore")
        )
        if subset.empty:
            continue
        x = subset["run_number"].astype(int)
        sigma_x = subset["sigma_x"]
        sigma_y = subset["sigma_y"]
        
        y_lower, y_upper = sigma_y.quantile([0.05, 0.95])
        y_margin = 0.05 * (y_upper - y_lower)

        axes[0].plot(x, sigma_x, marker="o", linestyle="None", color=color, label=particle)
        axes[1].plot(x, sigma_y, marker="o", linestyle="None", color=color, label=particle)

        axes[1].set_ylim(y_lower - y_margin, y_upper + y_margin)    


    axes[0].set_title("Gaussian Fit Sigma X by Run")
    axes[0].set_ylabel("Sigma X (mm)")
    axes[0].set_ylim([20, 40])
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[0].set_box_aspect(1 / 3)

    axes[1].set_title("Gaussian Fit Sigma Y by Run")
    axes[1].set_xlabel("Run Number")
    axes[1].set_ylabel("Sigma Y (mm)")
    axes[1].set_ylim([10, 30])
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()
    axes[1].set_box_aspect(1 / 3)

    all_runs = sorted(sigmas["run_number"].astype(int).unique())
    if len(all_runs) > 0:
        step = max(1, len(all_runs) // 20)
        axes[1].set_xticks(all_runs[::step])
        axes[1].set_xticklabels([str(x) for x in all_runs[::step]], rotation=45, ha="right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sigma_by_momentum(sigmas: pd.DataFrame, run_summary: pd.DataFrame, output_path: str) -> None:
    momentum_map = run_summary[["run_number", "beam_momentum"]].copy()
    momentum_map["run_number"] = pd.to_numeric(momentum_map["run_number"], errors="coerce")
    momentum_map["beam_momentum_numeric"] = safe_numeric(momentum_map["beam_momentum"])
    momentum_map = momentum_map.dropna(subset=["run_number", "beam_momentum_numeric"])
    momentum_map = momentum_map.drop_duplicates(subset=["run_number"])

    df = sigmas.merge(momentum_map[["run_number", "beam_momentum_numeric"]], on="run_number", how="left")
    df = df.dropna(subset=["beam_momentum_numeric"])
    if df.empty:
        raise ValueError("No overlapping numeric beam momentum values found for sigma plotting")

    colors = ["#4C72B0", "#55A868", "#C44E52", "#1AEADF"]
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    for particle, color in zip(PARTICLES, colors):
        subset = df[df["particle"] == particle].sort_values(
            by=["beam_momentum_numeric", "run_number"]
        )
        if subset.empty:
            continue
        x = subset["beam_momentum_numeric"]
        axes[0].plot(
            x,
            subset["sigma_x"],
            marker="o",
            markersize=4,
            linestyle="None",
            color=color,
            label=particle,
        )
        axes[1].plot(
            x,
            subset["sigma_y"],
            marker="o",
            markersize=4,
            linestyle="None",
            color=color,
            label=particle,
        )

    axes[0].set_title("Gaussian Fit Sigma X by Beam Momentum")
    axes[0].set_ylabel("Sigma X (mm)")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    axes[0].legend()
    axes[0].set_box_aspect(1 / 3)

    axes[1].set_title("Gaussian Fit Sigma Y by Beam Momentum")
    axes[1].set_xlabel("Beam Momentum")
    axes[1].set_ylabel("Sigma Y (mm)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1].legend()
    axes[1].set_box_aspect(1 / 3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_sigma_zoom_by_run(sigmas: pd.DataFrame, output_dir: str, ranges: list[tuple[int, int]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#1AEADF"]

    for start, end in ranges:
        subset = sigmas[
            (sigmas["run_number"] >= start) & (sigmas["run_number"] <= end)
        ].copy()
        if subset.empty:
            continue

        subset = subset.sort_values(by="run_number")
        runs = subset["run_number"].astype(int)

        fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        for particle, color in zip(PARTICLES, colors):
            particle_df = subset[subset["particle"] == particle]
            if particle_df.empty:
                continue
            axes[0].plot(
                particle_df["run_number"].astype(int),
                particle_df["sigma_x"],
                marker="o",
                linestyle="-",
                color=color,
                label=particle,
            )
            axes[1].plot(
                particle_df["run_number"].astype(int),
                particle_df["sigma_y"],
                marker="o",
                linestyle="-",
                color=color,
                label=particle,
            )

        axes[0].set_title(f"Gaussian Fit Sigma X by Run ({start}-{end})")
        axes[0].set_ylabel("Sigma X (mm)")
        axes[0].grid(axis="y", linestyle="--", alpha=0.4)
        axes[0].legend()
        axes[0].set_box_aspect(1 / 3)

        axes[1].set_title(f"Gaussian Fit Sigma Y by Run ({start}-{end})")
        axes[1].set_xlabel("Run Number")
        axes[1].set_ylabel("Sigma Y (mm)")
        axes[1].grid(axis="y", linestyle="--", alpha=0.4)
        axes[1].legend()
        axes[1].set_box_aspect(1 / 3)

        if len(runs) > 0:
            step = max(1, len(runs) // 15)
            axes[1].set_xticks(runs[::step])
            axes[1].set_xticklabels([str(x) for x in runs[::step]], rotation=45, ha="right")

        fig.tight_layout()
        output_path = os.path.join(output_dir, f"sigma_by_run_{start}_{end}.png")
        fig.savefig(output_path, dpi=200)
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize parquet files in the data folder and join run metadata."
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(os.path.dirname(__file__), "data"),
        help="Root directory containing particle subfolders with parquet files.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to all_runs_processed.json metadata file.",
    )
    parser.add_argument(
        "--output-csv",
        default=os.path.join(os.path.dirname(__file__), "run_summary.csv"),
        help="Output CSV file for run-level summary.",
    )
    parser.add_argument(
        "--output-plot",
        default=os.path.join(os.path.dirname(__file__), "plots", "particle_counts.png"),
        help="Output plot file for total particle counts.",
    )
    parser.add_argument(
        "--output-plot-run",
        default=os.path.join(
            os.path.dirname(__file__), "plots", "particle_counts_by_run.png"
        ),
        help="Output plot file for per-run particle counts.",
    )
    parser.add_argument(
        "--output-plot-momentum",
        default=os.path.join(
            os.path.dirname(__file__),
            "plots",
            "particle_counts_by_beam_momentum.png",
        ),
        help="Output plot file for particle counts versus beam momentum.",
    )
    parser.add_argument(
        "--output-sigma-csv",
        default=os.path.join(os.path.dirname(__file__), "run_gaussian_summary.csv"),
        help="Output CSV file for Gaussian sigma fit results.",
    )
    parser.add_argument(
        "--output-plot-sigma",
        default=os.path.join(os.path.dirname(__file__), "plots", "sigma_by_run.png"),
        help="Output plot file for sigma_x and sigma_y comparison by run.",
    )
    parser.add_argument(
        "--output-plot-sigma-momentum",
        default=os.path.join(os.path.dirname(__file__), "plots", "sigma_by_beam_momentum.png"),
        help="Output plot file for sigma_x and sigma_y versus beam momentum.",
    )
    parser.add_argument(
        "--output-plot-sigma-zoom-dir",
        default=os.path.join(os.path.dirname(__file__), "plots", "sigma_zoom"),
        help="Output directory for zoomed sigma plots by run range.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    metadata_path = args.metadata or find_metadata_path(script_dir)
    counts = collect_parquet_counts(args.data_root)
    metadata = load_run_metadata(metadata_path)
    run_summary = build_run_summary(counts, metadata)

    output_csv = os.path.abspath(args.output_csv)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    run_summary.to_csv(output_csv, index=False)

    output_plot = os.path.abspath(args.output_plot)
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plot_particle_totals(counts, output_plot)

    output_plot_run = os.path.abspath(args.output_plot_run)
    os.makedirs(os.path.dirname(output_plot_run), exist_ok=True)
    plot_run_comparison(run_summary, output_plot_run, "run_number")

    output_plot_momentum = os.path.abspath(args.output_plot_momentum)
    os.makedirs(os.path.dirname(output_plot_momentum), exist_ok=True)
    plot_particle_counts_vs_momentum(run_summary, output_plot_momentum)

    sigma_csv = os.path.abspath(args.output_sigma_csv)
    os.makedirs(os.path.dirname(sigma_csv), exist_ok=True)
    fit_sigmas = collect_gaussian_sigmas(args.data_root)
    fit_sigmas.to_csv(sigma_csv, index=False)

    output_plot_sigma = os.path.abspath(args.output_plot_sigma)
    os.makedirs(os.path.dirname(output_plot_sigma), exist_ok=True)
    plot_sigma_by_run(fit_sigmas, output_plot_sigma)

    output_plot_sigma_momentum = os.path.abspath(args.output_plot_sigma_momentum)
    os.makedirs(os.path.dirname(output_plot_sigma_momentum), exist_ok=True)
    plot_sigma_by_momentum(fit_sigmas, run_summary, output_plot_sigma_momentum)

    output_plot_sigma_zoom_dir = os.path.abspath(args.output_plot_sigma_zoom_dir)
    plot_sigma_zoom_by_run(
        fit_sigmas,
        output_plot_sigma_zoom_dir,
        [(1550, 1650), (1850, 1900), (2100, 2200), (2200, 2400), (2400, 2500)],
    )

    print(f"Wrote run summary table to: {output_csv}")
    print(f"Wrote particle totals plot to: {output_plot}")
    print(f"Wrote per-run particle comparison plot to: {output_plot_run}")
    print(f"Wrote beam momentum comparison plot to: {output_plot_momentum}")
    print(f"Wrote gaussian fit summary to: {sigma_csv}")
    print(f"Wrote sigma comparison plot to: {output_plot_sigma}")
    print(f"Wrote sigma versus momentum plot to: {output_plot_sigma_momentum}")
    print(f"Wrote zoomed sigma plots to: {output_plot_sigma_zoom_dir}")
    print("\nRun summary table (first 20 rows):")
    print(run_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
