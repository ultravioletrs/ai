"""
Healthcare - Data Preparation

Splits the UCI Diabetes 130-US Hospitals dataset into 3 hospital datasets
simulating a consortium where hospitals collaborate on readmission prediction
without exposing HIPAA-protected patient records.

Dataset: UCI Diabetes 130-US Hospitals for Years 1999-2008
Source:  https://www.kaggle.com/datasets/jimschacko/10-years-diabetes-dataset
"""

import argparse
import os
import random
import zipfile
import pandas as pd


def load_dataset(zip_path):
    """Load and clean the Diabetes 130-US Hospitals dataset from a zip."""
    with zipfile.ZipFile(zip_path, "r") as z:
        csv_files = [f for f in z.namelist() if f.endswith(".csv") and "diabetic" in f.lower()]
        if not csv_files:
            csv_files = [f for f in z.namelist() if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the zip archive")
        with z.open(csv_files[0]) as f:
            df = pd.read_csv(f)
    print(f"Loaded {len(df)} rows from {csv_files[0]}")
    if "discharge_disposition_id" in df.columns:
        exclude_discharge = [11, 13, 14, 19, 20, 21]
        df = df[~df["discharge_disposition_id"].isin(exclude_discharge)]
    if "patient_nbr" in df.columns:
        df = df.drop_duplicates(subset=["patient_nbr"], keep="first")
    print(f"After cleaning: {len(df)} rows, {df['patient_nbr'].nunique()} unique patients")
    return df


def split_by_hospital(df, n_hospitals=3, seed=42):
    """Split dataset into n disjoint hospital datasets by patient ID."""
    random.seed(seed)
    patients = list(df["patient_nbr"].unique())
    random.shuffle(patients)
    chunk_size = len(patients) // n_hospitals
    hospital_patients = []
    for i in range(n_hospitals):
        start = i * chunk_size
        end = start + chunk_size if i < n_hospitals - 1 else len(patients)
        hospital_patients.append(set(patients[start:end]))
    hospital_dfs = []
    for i, pat_set in enumerate(hospital_patients):
        hospital_df = df[df["patient_nbr"].isin(pat_set)].copy()
        hospital_dfs.append(hospital_df)
        readmit_30 = (hospital_df["readmitted"] == "<30").sum()
        readmit_rate = readmit_30 / len(hospital_df) * 100
        print(f"Hospital {i + 1}: {len(hospital_df)} encounters, "
              f"{len(pat_set)} patients, {readmit_rate:.1f}% 30-day readmission rate")
    return hospital_dfs


def save_datasets(hospital_dfs, output_dir):
    """Save each hospital dataset as a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    for i, df in enumerate(hospital_dfs):
        filename = f"hospital_{i + 1}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filepath} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare healthcare datasets from UCI Diabetes 130-US Hospitals")
    parser.add_argument("zipfile", type=str,
                        help="Path to the diabetes dataset zip file")
    parser.add_argument("-d", "--destination", type=str, default="datasets",
                        help="Output directory (default: datasets)")
    parser.add_argument("-n", "--num-hospitals", type=int, default=3,
                        help="Number of hospitals to split data into (default: 3)")
    args = parser.parse_args()
    df = load_dataset(args.zipfile)
    hospital_dfs = split_by_hospital(df, n_hospitals=args.num_hospitals)
    save_datasets(hospital_dfs, args.destination)
    print(f"\nDataset preparation complete. {args.num_hospitals} hospital datasets saved to '{args.destination}/'")


if __name__ == "__main__":
    main()

