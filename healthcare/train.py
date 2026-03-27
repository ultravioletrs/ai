"""
Healthcare - Multi-Hospital Patient Readmission Prediction (Training)

Runs inside a TEE on Prism AI / Cocos. Receives HIPAA-protected patient
encounter records from multiple hospitals and trains a unified readmission
risk model without any hospital seeing another's raw patient data.

Inputs  (datasets/): hospital_1.csv, hospital_2.csv, hospital_3.csv
Outputs (results/) : readmission_model.ubj, benchmark_report.csv,
                     feature_importance.csv, risk_distribution.csv
"""

import os
os.environ["OPENBLAS_L2_SIZE"] = "1024"

import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATASETS_DIR = "datasets"
RESULTS_DIR = "results"

CATEGORICAL_COLS = [
    "race", "gender", "age", "admission_type_id",
    "discharge_disposition_id", "admission_source_id",
    "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]

MEDICATION_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide",
    "glimepiride", "acetohexamide", "glipizide", "glyburide",
    "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
    "miglitol", "troglitazone", "tolazamide", "examide", "citoglipton", "insulin",
    "glyburide.metformin", "glipizide.metformin",
    "glimepiride.pioglitazone", "metformin.rosiglitazone",
    "metformin.pioglitazone",
]

NUMERIC_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]


def build_features(df):
    """Clean and engineer features from raw encounter data."""
    df = df.copy()
    df["readmitted_30d"] = (df["readmitted"] == "<30").astype(int)
    drop = ["encounter_id", "patient_nbr", "readmitted", "weight"]
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    df = df.replace("?", np.nan)
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
    for col in MEDICATION_COLS:
        if col in df.columns:
            df[col] = df[col].map(med_map).fillna(0).astype(int)
    df["num_med_changes"] = df[[c for c in MEDICATION_COLS if c in df.columns]].sum(axis=1)
    df["total_visits"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def get_feature_columns(df):
    """Return the list of feature columns (everything except the target)."""
    return [c for c in df.columns if c != "readmitted_30d"]


def train_model(X_train, y_train, X_val, y_val, feature_names=None):
    """Train XGBoost readmission classifier."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    params = {
        "objective": "binary:logistic", "eval_metric": "auc",
        "eta": 0.1, "max_depth": 6, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5,
        "scale_pos_weight": spw, "seed": 42,
    }
    model = xgb.train(params, dtrain, num_boost_round=300,
                      evals=[(dval, "validation")],
                      early_stopping_rounds=15, verbose_eval=50)
    return model


def evaluate_model(model, X, y, label="", feature_names=None):
    """Evaluate model and return metrics dict."""
    dmatrix = xgb.DMatrix(X, feature_names=feature_names)
    probs = model.predict(dmatrix)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = 0.0
    print(f"  [{label}] Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
    return {"Dataset": label, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "AUC": auc}


def main():
    if not os.path.isdir(DATASETS_DIR):
        print(f"Dataset directory {DATASETS_DIR} not found")
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load all hospital datasets ────────────────────────────────────────
    hospital_files = sorted([f for f in os.listdir(DATASETS_DIR) if f.endswith(".csv")])
    if not hospital_files:
        print("No CSV datasets found in datasets/")
        return
    print(f"Found {len(hospital_files)} hospital datasets: {hospital_files}")
    print("=" * 60)

    hospital_raw = {}
    for f in hospital_files:
        name = os.path.splitext(f)[0]
        df = pd.read_csv(os.path.join(DATASETS_DIR, f))
        hospital_raw[name] = df
        print(f"  {name}: {len(df)} encounters")

    # ── Build features per hospital and combined ──────────────────────────
    print("\nBuilding features...")
    hospital_features = {}
    all_features = []
    for name, df in hospital_raw.items():
        features = build_features(df)
        hospital_features[name] = features
        all_features.append(features)
        rate = features["readmitted_30d"].mean() * 100
        print(f"  {name}: {len(features)} encounters, {rate:.1f}% readmission rate")

    combined = pd.concat(all_features, ignore_index=True)
    rate = combined["readmitted_30d"].mean() * 100
    print(f"  Combined consortium: {len(combined)} encounters, {rate:.1f}% readmission rate")
    feature_cols = get_feature_columns(combined)
    print(f"  Using {len(feature_cols)} features")

    # ── Train consortium model (all hospitals together) ───────────────────
    print("\n" + "=" * 60)
    print("TRAINING CONSORTIUM MODEL (all hospitals)")
    print("=" * 60)
    X = combined[feature_cols].values
    y = combined["readmitted_30d"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)
    consortium_model = train_model(X_train, y_train, X_val, y_val,
                                   feature_names=feature_cols)

    # ── Evaluate consortium model ─────────────────────────────────────────
    print("\nConsortium model evaluation:")
    benchmark_rows = []
    m = evaluate_model(consortium_model, X_test, y_test,
                       "Consortium (all hospitals)", feature_names=feature_cols)
    benchmark_rows.append(m)
    for name, features in hospital_features.items():
        Xh = features[feature_cols].values
        yh = features["readmitted_30d"].values
        m = evaluate_model(consortium_model, Xh, yh,
                           f"Consortium on {name}", feature_names=feature_cols)
        benchmark_rows.append(m)

    # ── Train individual hospital models for comparison ───────────────────
    print("\n" + "=" * 60)
    print("TRAINING INDIVIDUAL HOSPITAL MODELS (for benchmark)")
    print("=" * 60)
    for name, features in hospital_features.items():
        print(f"\n  Training model for {name}...")
        Xh = features[feature_cols].values
        yh = features["readmitted_30d"].values
        if len(Xh) < 50:
            print(f"    Skipping {name}: insufficient data ({len(Xh)} encounters)")
            continue
        Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(
            Xh, yh, test_size=0.2, random_state=42, stratify=yh)
        Xh_tr, Xh_va, yh_tr, yh_va = train_test_split(
            Xh_tr, yh_tr, test_size=0.15, random_state=42, stratify=yh_tr)
        ind_model = train_model(Xh_tr, yh_tr, Xh_va, yh_va,
                                feature_names=feature_cols)
        m = evaluate_model(ind_model, Xh_te, yh_te,
                           f"{name} (solo model)", feature_names=feature_cols)
        benchmark_rows.append(m)

    # ── Save results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # 1. Model
    model_path = os.path.join(RESULTS_DIR, "readmission_model.ubj")
    consortium_model.save_model(model_path)
    print(f"  Saved model: {model_path}")

    # 2. Benchmark report
    bench_df = pd.DataFrame(benchmark_rows)
    bench_path = os.path.join(RESULTS_DIR, "benchmark_report.csv")
    bench_df.to_csv(bench_path, index=False)
    print(f"  Saved benchmark: {bench_path}")
    print("\n  BENCHMARK RESULTS:")
    print(bench_df.to_string(index=False))

    # 3. Feature importance
    importance = consortium_model.get_score(importance_type="gain")
    imp_df = pd.DataFrame(
        [{"Feature": k, "Importance": v} for k, v in importance.items()]
    ).sort_values("Importance", ascending=False)
    imp_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    imp_df.to_csv(imp_path, index=False)
    print(f"\n  Saved feature importance: {imp_path}")
    print(imp_df.head(15).to_string(index=False))

    # 4. Risk distribution
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    risk_probs = consortium_model.predict(dtest)
    risk_df = pd.DataFrame({
        "RiskScore": risk_probs,
        "RiskBucket": pd.cut(
            risk_probs, bins=[0, 0.1, 0.2, 0.3, 0.5, 1.0],
            labels=["Very Low", "Low", "Medium", "High", "Very High"]),
        "Actual": y_test,
    })
    risk_summary = risk_df.groupby("RiskBucket", observed=True).agg(
        Count=("RiskScore", "count"),
        AvgRisk=("RiskScore", "mean"),
        ActualReadmitRate=("Actual", "mean"),
    ).reset_index()
    risk_path = os.path.join(RESULTS_DIR, "risk_distribution.csv")
    risk_summary.to_csv(risk_path, index=False)
    print(f"\n  Saved risk distribution: {risk_path}")
    print(risk_summary.to_string(index=False))

    print("\n" + "=" * 60)
    print("COMPUTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

