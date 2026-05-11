"""
plot_compression_profile.py
---------------------------
Combines compression_profile_results.csv files from a context-only compression
profiling sweep, writes an aggregate summary table, and generates
compression-side plots.

These plots are for diffusion.context_fn(images): latent encoding,
hyperprior/BPP computation, quantization, and context feature generation. They
do not describe diffusion reconstruction timing.

Usage:
  cd /projects/bfod/$USER/sc26-cdc-deltaai
  python xparam/plot_compression_profile.py \
    --profile_dir outputs/compression_profile/<job_id> \
    --out_dir outputs/compression_profile/<job_id>

Outputs:
  compression_profile_summary.csv
  plots/plot_context_time_vs_checkpoint.png
  plots/plot_bpp_vs_checkpoint.png
  plots/plot_speed_vs_compression_ratio.png
  plots/plot_memory_vs_checkpoint.png
  plots/plot_context_time_vs_bpp.png
  plots/plot_compression_ratio_vs_checkpoint.png
"""

import argparse
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "checkpoint_label",
    "avg_context_sec",
    "std_context_sec",
    "avg_total_sec",
    "avg_images_per_hour",
    "avg_peak_gpu_mem_mb",
    "avg_bpp",
    "avg_raw_rgb_to_est_compressed_ratio",
    "avg_compression_ratio",
    "avg_orig_file_to_est_compressed_ratio",
    "avg_estimated_compressed_bytes",
    "total_orig_size_bytes",
    "avg_orig_size_bytes",
    "n_rows",
    "n_unique_images",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CDC context-only compression profile results")
    parser.add_argument("--profile_dir", type=str, required=True, help="Root directory containing profile result CSVs")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for summary CSV and plots")
    return parser.parse_args()


def checkpoint_sort_key(label):
    match = re.search(r"b([0-9]+(?:\.[0-9]+)?)", str(label))
    if match:
        return float(match.group(1))
    return float("inf")


def read_profile_csvs(profile_dir: pathlib.Path) -> pd.DataFrame:
    csv_paths = sorted(profile_dir.rglob("compression_profile_results.csv"))
    if not csv_paths:
        print(f"No compression_profile_results.csv files found under: {profile_dir}")
        return pd.DataFrame()

    frames = []
    for csv_path in csv_paths:
        try:
            if csv_path.stat().st_size == 0:
                print(f"Skipping empty result file: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Skipping unreadable result file: {csv_path} ({exc})")
            continue

        if df.empty:
            print(f"Skipping result file with no rows: {csv_path}")
            continue

        df["source_csv"] = str(csv_path)
        frames.append(df)

    if not frames:
        print("No non-empty profile result files were readable.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    if "status" in df.columns:
        df = df[df["status"].fillna("success") == "success"].copy()

    if "checkpoint_label" not in df.columns:
        if "checkpoint" in df.columns:
            df["checkpoint_label"] = df["checkpoint"]
        else:
            df["checkpoint_label"] = "unknown"

    if "context_sec" not in df.columns and "compress_sec" in df.columns:
        print(
            "WARNING: Found legacy compress_sec without context_sec; treating "
            "compress_sec as context_sec. This may be legacy end-to-end timing."
        )
        df["context_sec"] = df["compress_sec"]

    if "raw_rgb_to_est_compressed_ratio" not in df.columns and "compression_ratio" in df.columns:
        df["raw_rgb_to_est_compressed_ratio"] = df["compression_ratio"]
    if "compression_ratio" not in df.columns and "raw_rgb_to_est_compressed_ratio" in df.columns:
        df["compression_ratio"] = df["raw_rgb_to_est_compressed_ratio"]

    numeric_columns = [
        "context_sec",
        "total_sec",
        "images_per_hour",
        "peak_gpu_mem_mb",
        "bpp",
        "raw_rgb_to_est_compressed_ratio",
        "compression_ratio",
        "orig_file_to_est_compressed_ratio",
        "estimated_compressed_bytes",
        "orig_size_bytes",
    ]
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "image" not in df.columns:
        df["image"] = ""

    required = ["context_sec", "total_sec", "bpp", "raw_rgb_to_est_compressed_ratio"]
    df = df.dropna(subset=required, how="any").copy()
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    summary = (
        df.groupby("checkpoint_label")
        .agg(
            avg_context_sec=("context_sec", "mean"),
            std_context_sec=("context_sec", "std"),
            avg_total_sec=("total_sec", "mean"),
            avg_images_per_hour=("images_per_hour", "mean"),
            avg_peak_gpu_mem_mb=("peak_gpu_mem_mb", "mean"),
            avg_bpp=("bpp", "mean"),
            avg_raw_rgb_to_est_compressed_ratio=("raw_rgb_to_est_compressed_ratio", "mean"),
            avg_compression_ratio=("compression_ratio", "mean"),
            avg_orig_file_to_est_compressed_ratio=("orig_file_to_est_compressed_ratio", "mean"),
            avg_estimated_compressed_bytes=("estimated_compressed_bytes", "mean"),
            total_orig_size_bytes=("orig_size_bytes", "sum"),
            avg_orig_size_bytes=("orig_size_bytes", "mean"),
            n_rows=("image", "count"),
            n_unique_images=("image", "nunique"),
        )
        .reset_index()
    )
    summary["std_context_sec"] = summary["std_context_sec"].fillna(0.0)
    summary["sort_key"] = summary["checkpoint_label"].map(checkpoint_sort_key)
    summary = summary.sort_values(["sort_key", "checkpoint_label"]).drop(columns=["sort_key"])
    return summary[SUMMARY_COLUMNS]


def print_summary(summary: pd.DataFrame):
    if summary.empty:
        print("No successful compression profile rows found.")
        return

    print("\n" + "=" * 92)
    print("  COMPRESSION PROFILE SUMMARY")
    print("=" * 92)
    print(
        f"  {'Checkpoint':>10}  {'Context(s)':>10}  {'Total(s)':>8}  {'Img/hr':>8}  "
        f"{'Mem(MB)':>8}  {'BPP':>7}  {'RGB/Est':>8}  {'Rows':>5}"
    )
    print("-" * 92)
    for _, row in summary.iterrows():
        print(
            f"  {row.checkpoint_label:>10}  "
            f"{row.avg_context_sec:>10.3f}  "
            f"{row.avg_total_sec:>8.3f}  "
            f"{row.avg_images_per_hour:>8.2f}  "
            f"{row.avg_peak_gpu_mem_mb:>8.1f}  "
            f"{row.avg_bpp:>7.4f}  "
            f"{row.avg_raw_rgb_to_est_compressed_ratio:>7.2f}x  "
            f"{int(row.n_rows):>5}"
        )
    print("=" * 92 + "\n")


def save_time_plot(summary: pd.DataFrame, plots_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["avg_context_sec"],
        yerr=summary["std_context_sec"],
        capsize=4,
        color=plt.get_cmap("tab10")(0),
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["checkpoint_label"], rotation=0)
    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Avg Context Time per Image (s)", fontsize=12)
    ax.set_title("Compression Context Time vs Checkpoint", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = plots_dir / "plot_context_time_vs_checkpoint.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_bpp_plot(summary: pd.DataFrame, plots_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    ax.plot(
        x,
        summary["avg_bpp"],
        marker="o",
        linewidth=2,
        color=plt.get_cmap("tab10")(1),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["checkpoint_label"], rotation=0)
    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Average BPP", fontsize=12)
    ax.set_title("BPP vs Checkpoint", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = plots_dir / "plot_bpp_vs_checkpoint.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_speed_ratio_plot(summary: pd.DataFrame, plots_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, row in summary.reset_index(drop=True).iterrows():
        ax.scatter(
            row.avg_raw_rgb_to_est_compressed_ratio,
            row.avg_images_per_hour,
            s=90,
            color=cmap(idx % 10),
            label=row.checkpoint_label,
            zorder=3,
        )
        ax.annotate(
            row.checkpoint_label,
            xy=(row.avg_raw_rgb_to_est_compressed_ratio, row.avg_images_per_hour),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Average Raw RGB / Estimated CDC Size", fontsize=12)
    ax.set_ylabel("Average Images per Hour", fontsize=12)
    ax.set_title("Speed vs Compression Ratio", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    if len(summary) > 1:
        ax.legend(title="Checkpoint")
    plt.tight_layout()
    out_path = plots_dir / "plot_speed_vs_compression_ratio.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_memory_plot(summary: pd.DataFrame, plots_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["avg_peak_gpu_mem_mb"],
        color=plt.get_cmap("tab10")(2),
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["checkpoint_label"], rotation=0)
    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=12)
    ax.set_title("GPU Memory Usage vs Checkpoint", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = plots_dir / "plot_memory_vs_checkpoint.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_context_bpp_plot(summary: pd.DataFrame, plots_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for idx, row in summary.reset_index(drop=True).iterrows():
        ax.scatter(
            row.avg_bpp,
            row.avg_context_sec,
            s=90,
            color=cmap(idx % 10),
            label=row.checkpoint_label,
            zorder=3,
        )
        ax.annotate(
            row.checkpoint_label,
            xy=(row.avg_bpp, row.avg_context_sec),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.set_xlabel("Average BPP", fontsize=12)
    ax.set_ylabel("Avg Context Time per Image (s)", fontsize=12)
    ax.set_title("Context Time vs BPP", fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    if len(summary) > 1:
        ax.legend(title="Checkpoint")
    plt.tight_layout()
    out_path = plots_dir / "plot_context_time_vs_bpp.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def save_compression_ratio_plot(summary: pd.DataFrame, plots_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["avg_raw_rgb_to_est_compressed_ratio"],
        color=plt.get_cmap("tab10")(3),
        alpha=0.85,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["checkpoint_label"], rotation=0)
    ax.set_xlabel("Checkpoint", fontsize=12)
    ax.set_ylabel("Average Raw RGB / Estimated CDC Size", fontsize=12)
    ax.set_title("Raw RGB / Estimated CDC Size vs Checkpoint", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = plots_dir / "plot_compression_ratio_vs_checkpoint.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    args = parse_args()
    profile_dir = pathlib.Path(args.profile_dir)
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else profile_dir
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_dataframe(read_profile_csvs(profile_dir))
    summary = build_summary(df)

    summary_path = out_dir / "compression_profile_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    print_summary(summary)

    if summary.empty:
        print("No plots generated because the summary is empty.")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    save_time_plot(summary, plots_dir)
    save_bpp_plot(summary, plots_dir)
    save_speed_ratio_plot(summary, plots_dir)
    save_memory_plot(summary, plots_dir)
    save_context_bpp_plot(summary, plots_dir)
    save_compression_ratio_plot(summary, plots_dir)
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
