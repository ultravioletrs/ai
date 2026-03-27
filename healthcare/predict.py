"""
Healthcare - Multi-Hospital Patient Readmission Prediction (Inference / Analysis)

Loads the trained consortium readmission model and produces evaluation
metrics, visualizations, and a summary report demonstrating the value of
multi-hospital collaborative analytics over single-hospital models.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def load_results(results_dir):
    """Load and display all result artifacts."""
    print("=" * 60)
    print("HEALTHCARE - CONSORTIUM RESULTS ANALYSIS")
    print("=" * 60)

    # ── Benchmark Report ──────────────────────────────────────────────────
    benchmark_path = os.path.join(results_dir, "benchmark_report.csv")
    if os.path.exists(benchmark_path):
        benchmark = pd.read_csv(benchmark_path)
        print("\n📊 BENCHMARK: Consortium vs. Individual Hospital Models")
        print("-" * 60)
        print(benchmark.to_string(index=False))

        consortium_auc = benchmark.loc[
            benchmark["Dataset"].str.contains("Consortium.*all", na=False), "AUC"
        ].values
        solo_auc = benchmark.loc[
            benchmark["Dataset"].str.contains("solo", na=False), "AUC"
        ].values

        if len(consortium_auc) > 0 and len(solo_auc) > 0:
            avg_solo = solo_auc.mean()
            improvement = ((consortium_auc[0] - avg_solo) / max(abs(avg_solo), 0.01)) * 100
            print(f"\n  Consortium AUC : {consortium_auc[0]:.4f}")
            print(f"  Avg Solo AUC   : {avg_solo:.4f}")
            print(f"  Improvement    : {improvement:+.1f}%")

        plot_benchmark(benchmark, results_dir)
    else:
        print(f"  ⚠ Benchmark report not found at {benchmark_path}")

    # ── Feature Importance ────────────────────────────────────────────────
    importance_path = os.path.join(results_dir, "feature_importance.csv")
    if os.path.exists(importance_path):
        importance = pd.read_csv(importance_path)
        print("\nTOP PREDICTIVE FEATURES FOR READMISSION")
        print("-" * 60)
        print(importance.head(15).to_string(index=False))
        plot_feature_importance(importance, results_dir)
    else:
        print(f"   Feature importance not found at {importance_path}")

    # ── Risk Distribution ─────────────────────────────────────────────────
    risk_path = os.path.join(results_dir, "risk_distribution.csv")
    if os.path.exists(risk_path):
        risk = pd.read_csv(risk_path)
        print("\n PATIENT READMISSION RISK DISTRIBUTION")
        print("-" * 60)
        print(risk.to_string(index=False))
        plot_risk_distribution(risk, results_dir)
    else:
        print(f"  Risk distribution not found at {risk_path}")

    # ── Model Info ────────────────────────────────────────────────────────
    model_path = os.path.join(results_dir, "readmission_model.ubj")
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"\n Model loaded successfully: {model_path}")
        print(f"   Model attributes: {model.attributes()}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


def plot_benchmark(benchmark, results_dir):
    """Plot benchmark comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(
        "Consortium Model vs. Individual Hospital Models",
        fontsize=14, fontweight="bold",
    )

    for ax, metric in zip(axes, ["AUC", "F1", "Accuracy"]):
        data = benchmark[["Dataset", metric]].copy()
        colors = [
            "#7c3aed" if "Consortium" in d and "all" in d
            else "#14b8a6" if "solo" in d
            else "#94a3b8"
            for d in data["Dataset"]
        ]
        short_labels = [
            d.replace("Consortium (all hospitals)", "Consortium")
            .replace("Consortium on ", "C→")
            .replace(" (solo model)", " Solo")
            for d in data["Dataset"]
        ]
        ax.barh(short_labels, data[metric], color=colors)
        ax.set_xlabel(metric)
        ax.set_title(metric)

    plt.tight_layout()
    chart_path = os.path.join(results_dir, "benchmark_comparison.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved chart: {chart_path}")


def plot_feature_importance(importance, results_dir):
    """Plot feature importance bar chart."""
    top = importance.head(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color="#7c3aed")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Top 15 Predictive Features — Readmission Risk Model")
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved chart: {chart_path}")


def plot_risk_distribution(risk, results_dir):
    """Plot risk bucket distribution."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    x = range(len(risk))
    bars = ax1.bar(x, risk["Count"], color="#7c3aed", alpha=0.7, label="Patient Count")
    ax1.set_xlabel("Risk Bucket")
    ax1.set_ylabel("Patient Count", color="#7c3aed")
    ax1.set_xticks(x)
    ax1.set_xticklabels(risk["RiskBucket"], rotation=15)

    ax2 = ax1.twinx()
    ax2.plot(x, risk["ActualReadmitRate"] * 100, "o-", color="#ef4444",
             linewidth=2, markersize=8, label="Actual Readmit %")
    ax2.set_ylabel("Actual Readmission Rate (%)", color="#ef4444")

    fig.suptitle("Patient Risk Distribution vs. Actual Readmission Rate",
                 fontsize=13, fontweight="bold")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "risk_distribution.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved chart: {chart_path}")


def main():
    results_dir = "results"
    if not os.path.isdir(results_dir):
        # Fallback: check datasets dir
        if os.path.isdir("datasets"):
            for f in os.listdir("datasets"):
                if f.endswith(".ubj"):
                    results_dir = "datasets"
                    break
        if not os.path.isdir(results_dir):
            print(f"Results directory {results_dir} not found")
            return

    load_results(results_dir)


if __name__ == "__main__":
    main()

