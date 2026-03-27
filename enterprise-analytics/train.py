"""
Enterprise Supply Chain Demand Forecasting - Training Algorithm

This algorithm runs inside a Trusted Execution Environment (TEE) on the
Prism AI / Cocos platform. It receives proprietary sales datasets from
multiple competing retailers and trains a unified demand forecasting model
without any party seeing another's raw data.

Scenario:
    Three retail companies contribute their transaction histories to jointly
    train an XGBoost model that predicts monthly product demand. The combined
    model outperforms any single-company model because it captures broader
    market signals, seasonal trends, and cross-geographic demand patterns.

Inputs (uploaded as datasets/):
    - company_1.csv, company_2.csv, company_3.csv

Outputs (saved to results/):
    - demand_model.ubj          : Trained XGBoost model
    - benchmark_report.csv      : Per-company vs. consortium accuracy
    - feature_importance.csv    : Top predictive features
    - monthly_forecast.csv      : 3-month forward demand forecast
"""

import os

os.environ["OPENBLAS_L2_SIZE"] = "1024"

import warnings

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATASETS_DIR = "datasets"
RESULTS_DIR = "results"


# ── Feature Engineering ──────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-series demand features from raw transaction data."""
    df = df.copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    df["Month"] = df["InvoiceDate"].dt.month
    df["DayOfWeek"] = df["InvoiceDate"].dt.dayofweek
    df["WeekOfYear"] = df["InvoiceDate"].dt.isocalendar().week.astype(int)

    # Aggregate to monthly product-level demand
    monthly = (
        df.groupby(["StockCode", "YearMonth", "Country", "Month", "WeekOfYear"])
        .agg(
            TotalQuantity=("Quantity", "sum"),
            TotalRevenue=("Revenue", "sum"),
            AvgPrice=("Price", "mean"),
            NumTransactions=("Invoice", "nunique"),
            NumCustomers=("Customer ID", "nunique"),
        )
        .reset_index()
    )

    # Encode categoricals
    le_stock = LabelEncoder()
    le_country = LabelEncoder()
    monthly["StockCode_enc"] = le_stock.fit_transform(monthly["StockCode"].astype(str))
    monthly["Country_enc"] = le_country.fit_transform(monthly["Country"].astype(str))

    # Sort for lag features
    monthly["YearMonth_str"] = monthly["YearMonth"].astype(str)
    monthly = monthly.sort_values(["StockCode", "YearMonth_str"])

    # Lag features (previous month demand)
    monthly["Lag1_Quantity"] = monthly.groupby("StockCode")["TotalQuantity"].shift(1)
    monthly["Lag2_Quantity"] = monthly.groupby("StockCode")["TotalQuantity"].shift(2)
    monthly["Lag1_Revenue"] = monthly.groupby("StockCode")["TotalRevenue"].shift(1)

    # Rolling averages
    monthly["Rolling3_Quantity"] = (
        monthly.groupby("StockCode")["TotalQuantity"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    monthly = monthly.dropna(subset=["Lag1_Quantity"])

    return monthly


FEATURE_COLS = [
    "StockCode_enc",
    "Country_enc",
    "Month",
    "WeekOfYear",
    "AvgPrice",
    "NumTransactions",
    "NumCustomers",
    "Lag1_Quantity",
    "Lag2_Quantity",
    "Lag1_Revenue",
    "Rolling3_Quantity",
]
TARGET_COL = "TotalQuantity"


# ── Training ─────────────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost demand forecasting model."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dval, "validation")],
        early_stopping_rounds=15,
        verbose_eval=50,
    )

    return model


def evaluate_model(model, X, y, label=""):
    """Evaluate model and return metrics dict."""
    dmatrix = xgb.DMatrix(X)
    preds = model.predict(dmatrix)
    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)
    print(f"  [{label}] MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    return {"Dataset": label, "MAE": mae, "RMSE": rmse, "R2": r2}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(DATASETS_DIR):
        print(f"Dataset directory {DATASETS_DIR} not found")
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load all company datasets ────────────────────────────────────────
    company_files = sorted(
        [f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")]
    )

    if not company_files:
        print("No CSV datasets found in datasets/")
        return

    print(f"Found {len(company_files)} company datasets: {company_files}")
    print("=" * 60)

    company_dfs = {}
    for f in company_files:
        name = os.path.splitext(f)[0]
        df = pd.read_csv(os.path.join(DATASETS_DIR, f))
        company_dfs[name] = df
        print(f"  {name}: {len(df)} transactions")

    # ── Build features per company and combined ──────────────────────────
    print("\nBuilding features...")
    company_features = {}
    all_features = []

    for name, df in company_dfs.items():
        features = build_features(df)
        company_features[name] = features
        all_features.append(features)
        print(f"  {name}: {len(features)} monthly demand records")

    combined = pd.concat(all_features, ignore_index=True)
    print(f"  Combined consortium: {len(combined)} monthly demand records")

    # ── Train consortium model (all companies together) ──────────────────
    print("\n" + "=" * 60)
    print("TRAINING CONSORTIUM MODEL (all companies)")
    print("=" * 60)

    X = combined[FEATURE_COLS].values
    y = combined[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    consortium_model = train_model(X_train, y_train, X_val, y_val)

    # ── Evaluate consortium model ────────────────────────────────────────
    print("\nConsortium model evaluation:")
    benchmark_rows = []
    consortium_metrics = evaluate_model(
        consortium_model, X_test, y_test, "Consortium (all companies)"
    )
    benchmark_rows.append(consortium_metrics)

    # Evaluate on each company's data separately
    for name, features in company_features.items():
        Xc = features[FEATURE_COLS].values
        yc = features[TARGET_COL].values
        metrics = evaluate_model(consortium_model, Xc, yc, f"Consortium on {name}")
        benchmark_rows.append(metrics)

    # ── Train individual company models for comparison ───────────────────
    print("\n" + "=" * 60)
    print("TRAINING INDIVIDUAL COMPANY MODELS (for benchmark)")
    print("=" * 60)

    for name, features in company_features.items():
        print(f"\n  Training model for {name}...")
        Xc = features[FEATURE_COLS].values
        yc = features[TARGET_COL].values

        if len(Xc) < 50:
            print(f"    Skipping {name}: insufficient data ({len(Xc)} records)")
            continue

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            Xc, yc, test_size=0.2, random_state=42
        )
        Xc_train, Xc_val, yc_train, yc_val = train_test_split(
            Xc_train, yc_train, test_size=0.15, random_state=42
        )

        individual_model = train_model(Xc_train, yc_train, Xc_val, yc_val)
        metrics = evaluate_model(
            individual_model, Xc_test, yc_test, f"{name} (solo model)"
        )
        benchmark_rows.append(metrics)

    # ── Save results ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # 1. Model
    model_path = os.path.join(RESULTS_DIR, "demand_model.ubj")
    consortium_model.save_model(model_path)
    print(f"  Saved model: {model_path}")

    # 2. Benchmark report
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_path = os.path.join(RESULTS_DIR, "benchmark_report.csv")
    benchmark_df.to_csv(benchmark_path, index=False)
    print(f"  Saved benchmark: {benchmark_path}")
    print("\n  BENCHMARK RESULTS:")
    print(benchmark_df.to_string(index=False))

    # 3. Feature importance
    importance = consortium_model.get_score(importance_type="gain")
    importance_df = pd.DataFrame(
        [{"Feature": FEATURE_COLS[int(k[1:])] if k.startswith("f") else k,
          "Importance": v}
         for k, v in importance.items()]
    ).sort_values("Importance", ascending=False)
    importance_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"\n  Saved feature importance: {importance_path}")
    print(importance_df.to_string(index=False))

    # 4. Forward forecast (next 3 months based on last known data)
    last_month_data = combined.sort_values("YearMonth_str").groupby("StockCode_enc").tail(1)
    forecast_rows = []
    for month_offset in range(1, 4):
        forecast_input = last_month_data[FEATURE_COLS].copy()
        forecast_input["Month"] = (forecast_input["Month"] + month_offset - 1) % 12 + 1
        dpred = xgb.DMatrix(forecast_input.values)
        preds = consortium_model.predict(dpred)
        for idx, (_, row) in enumerate(last_month_data.iterrows()):
            forecast_rows.append({
                "StockCode": row["StockCode"],
                "Country": row["Country"],
                "MonthOffset": month_offset,
                "PredictedDemand": max(0, preds[idx]),
            })

    forecast_df = pd.DataFrame(forecast_rows)
    forecast_path = os.path.join(RESULTS_DIR, "monthly_forecast.csv")
    forecast_df.to_csv(forecast_path, index=False)
    print(f"\n  Saved forecast: {forecast_path}")

    print("\n" + "=" * 60)
    print("COMPUTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

