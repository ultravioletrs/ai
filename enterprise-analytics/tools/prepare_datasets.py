"""
Enterprise Supply Chain Demand Forecasting - Data Preparation

Splits the UCI Online Retail II dataset into 3 separate company datasets
simulating a multi-party enterprise analytics consortium where retailers
collaborate on demand forecasting without exposing proprietary sales data.

Dataset: UCI Online Retail II
Source: https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

Each company dataset contains:
- Transaction history for a disjoint set of customers
- Invoice dates, product codes, quantities, and unit prices
- Country-level geographic distribution
"""

import argparse
import os
import random
import zipfile

import pandas as pd


def load_dataset(zip_path: str) -> pd.DataFrame:
    """Load and clean the Online Retail II dataset from a zip file."""
    with zipfile.ZipFile(zip_path, "r") as z:
        xlsx_files = [f for f in z.namelist() if f.endswith(".xlsx")]
        csv_files = [f for f in z.namelist() if f.endswith(".csv")]

        if xlsx_files:
            with z.open(xlsx_files[0]) as f:
                df = pd.read_excel(f, engine="openpyxl")
            data_file = xlsx_files[0]
        elif csv_files:
            with z.open(csv_files[0]) as f:
                df = pd.read_csv(f, encoding="utf-8")
            data_file = csv_files[0]
        else:
            raise FileNotFoundError("No .xlsx or .csv file found in the zip archive")

    print(f"Loaded {len(df)} rows from {data_file}")

    # Basic cleaning
    df = df.dropna(subset=["Customer ID", "Description"])
    df = df[df["Quantity"] > 0]
    df = df[df["Price"] > 0]
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["Revenue"] = df["Quantity"] * df["Price"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    print(f"After cleaning: {len(df)} rows, {df['Customer ID'].nunique()} customers")
    return df


def split_by_company(df: pd.DataFrame, n_companies: int = 3, seed: int = 42):
    """Split dataset into n disjoint company datasets by customer ID."""
    random.seed(seed)

    customers = list(df["Customer ID"].unique())
    random.shuffle(customers)

    chunk_size = len(customers) // n_companies
    company_customers = []
    for i in range(n_companies):
        start = i * chunk_size
        end = start + chunk_size if i < n_companies - 1 else len(customers)
        company_customers.append(set(customers[start:end]))

    company_dfs = []
    for i, cust_set in enumerate(company_customers):
        company_df = df[df["Customer ID"].isin(cust_set)].copy()
        company_dfs.append(company_df)
        print(
            f"Company {i + 1}: {len(company_df)} transactions, "
            f"{len(cust_set)} customers, "
            f"{company_df['Country'].nunique()} countries"
        )

    return company_dfs


def save_datasets(company_dfs: list, output_dir: str):
    """Save each company dataset as a CSV file."""
    os.makedirs(output_dir, exist_ok=True)

    for i, df in enumerate(company_dfs):
        filename = f"company_{i + 1}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Saved {filepath} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare enterprise analytics datasets from UCI Online Retail II"
    )
    parser.add_argument(
        "zipfile",
        type=str,
        help="Path to the online-retail-ii-uci.zip file",
    )
    parser.add_argument(
        "-d",
        "--destination",
        type=str,
        default="datasets",
        help="Output directory (default: datasets)",
    )
    parser.add_argument(
        "-n",
        "--num-companies",
        type=int,
        default=3,
        help="Number of companies to split data into (default: 3)",
    )
    args = parser.parse_args()

    df = load_dataset(args.zipfile)
    company_dfs = split_by_company(df, n_companies=args.num_companies)
    save_datasets(company_dfs, args.destination)

    print(f"\nDataset preparation complete. {args.num_companies} company datasets saved to '{args.destination}/'")


if __name__ == "__main__":
    main()

