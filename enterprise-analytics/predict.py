"""
Enterprise Supply Chain Demand Forecasting - Inference / Result Analysis

Loads the trained consortium demand model and produces evaluation metrics,
visualizations, and a summary report demonstrating the value of multi-party
collaborative analytics over single-company models.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb


def load_results(results_dir: str):
    """Load and display all result artifacts."""
    print("=" * 60)
    print("ENTERPRISE ANALYTICS - CONSORTIUM RESULTS ANALYSIS")
    print("=" * 60)

    # ── Benchmark Report ─────────────────────────────────────────────────
    benchmark_path = os.path.join(results_dir, "benchmark_report.csv")
    if os.path.exists(benchmark_path):
        benchmark = pd.read_csv(benchmark_path)
        print("\n📊 BENCHMARK: Consortium vs. Individual Company Models")
        print("-" * 60)
        print(benchmark.to_string(index=False))

        # Calculate improvement
        consortium_r2 = benchmark.loc[
            benchmark["Dataset"].str.contains("Consortium.*all", na=False), "R2"
        ].values
        solo_r2 = benchmark.loc[
            benchmark["Dataset"].str.contains("solo", na=False), "R2"
        ].values

        if len(consortium_r2) > 0 and len(solo_r2) > 0:
            avg_solo = solo_r2.mean()
            improvement = ((consortium_r2[0] - avg_solo) / max(abs(avg_solo), 0.01)) * 100
            print(f"\n  ✅ Consortium R² : {consortium_r2[0]:.4f}")
            print(f"  📉 Avg Solo R²   : {avg_solo:.4f}")
            print(f"  📈 Improvement   : {improvement:+.1f}%")

        # Plot benchmark comparison
        plot_benchmark(benchmark, results_dir)
    else:
        print(f"  ⚠ Benchmark report not found at {benchmark_path}")

    # ── Feature Importance ───────────────────────────────────────────────
    importance_path = os.path.join(results_dir, "feature_importance.csv")
    if os.path.exists(importance_path):
        importance = pd.read_csv(importance_path)
        print("\n TOP PREDICTIVE FEATURES")
        print("-" * 60)
        print(importance.head(10).to_string(index=False))
        plot_feature_importance(importance, results_dir)
    else:
        print(f"   Feature importance not found at {importance_path}")

    # ── Demand Forecast ──────────────────────────────────────────────────
    forecast_path = os.path.join(results_dir, "monthly_forecast.csv")
    if os.path.exists(forecast_path):
        forecast = pd.read_csv(forecast_path)
        print("\n DEMAND FORECAST SUMMARY (next 3 months)")
        print("-" * 60)
        summary = (
            forecast.groupby("MonthOffset")
            .agg(
                AvgDemand=("PredictedDemand", "mean"),
                TotalDemand=("PredictedDemand", "sum"),
                NumProducts=("StockCode", "nunique"),
            )
            .reset_index()
        )
        print(summary.to_string(index=False))
    else:
        print(f"   Forecast not found at {forecast_path}")

    # ── Model Info ───────────────────────────────────────────────────────
    model_path = os.path.join(results_dir, "demand_model.ubj")
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
        print(f"\n Model loaded successfully: {model_path}")
        print(f"   Model attributes: {model.attributes()}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


def plot_benchmark(benchmark: pd.DataFrame, results_dir: str):
    """Plot benchmark comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Consortium Model vs. Individual Company Models",
        fontsize=14,
        fontweight="bold",
    )

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        data = benchmark[["Dataset", metric]].copy()
        colors = [
            "#7c3aed" if "Consortium" in d and "all" in d
            else "#14b8a6" if "solo" in d
            else "#94a3b8"
            for d in data["Dataset"]
        ]
        short_labels = [
            d.replace("Consortium (all companies)", "Consortium")
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


def plot_feature_importance(importance: pd.DataFrame, results_dir: str):
    """Plot feature importance bar chart."""
    top = importance.head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color="#7c3aed")
    ax.set_xlabel("Importance (Gain)")
    ax.set_title("Top 10 Predictive Features - Consortium Model")
    plt.tight_layout()
    chart_path = os.path.join(results_dir, "feature_importance.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"  Saved chart: {chart_path}")


def main():
    datasets_dir = "datasets"
    results_dir = "results"

    if not os.path.isdir(results_dir):
        print(f"Results directory {results_dir} not found")
        return

    # Check for model and reports
    model_f = None
    for f in os.listdir(results_dir):
        if f.endswith(".ubj"):
            model_f = f

    if model_f is None:
        # Check datasets dir as fallback
        if os.path.isdir(datasets_dir):
            for f in os.listdir(datasets_dir):
                if f.endswith(".ubj"):
                    model_f = f
                    results_dir = datasets_dir

    load_results(results_dir)


if __name__ == "__main__":
    main()

