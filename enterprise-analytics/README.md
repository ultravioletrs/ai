# Enterprise Supply Chain Demand Forecasting

Confidential multi-party analytics for enterprise supply chain optimization. Three competing retail companies collaboratively train a demand forecasting model inside a Trusted Execution Environment (TEE)—each contributes proprietary transaction data, but **no company ever sees another's raw data**.

This example demonstrates:

- **Secure Computation (aTLS)** — Attested TLS verifies the TEE hardware and software stack before any data is uploaded
- **Multi-Party Computation** — Three independent data providers each upload proprietary datasets into the same encrypted enclave
- **Real-World Data** — Uses the [UCI Online Retail II](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) dataset (real European e-commerce transactions) split across simulated companies
- **Enterprise Value** — Benchmark proves the consortium model outperforms any single-company model

## Table of Contents

- [Scenario](#scenario)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Setup Virtual Environment](#setup-virtual-environment)
- [Install](#install)
- [Train Model (Local)](#train-model-local)
- [Test Model (Local)](#test-model-local)
- [Testing with Prism (Multi-Party)](#testing-with-prism-multi-party)
- [Notes](#notes)

## Scenario

Three retail companies—**Company 1**, **Company 2**, and **Company 3**—compete in the same market. Each holds proprietary customer transaction data that is a trade secret. No company would willingly hand over raw sales data to a competitor or a third party.

However, they all recognize that a **consortium demand forecast** trained on the combined market data would be far more accurate than any model they could train alone. Prism AI makes this possible:

1. A **neutral Algorithm Provider** supplies the training code (`train.py`)
2. Each company acts as a **Data Provider**, uploading encrypted datasets into the TEE
3. The TEE runs the algorithm over all three datasets simultaneously
4. Only the **aggregated results** (trained model + benchmark report) exit the enclave
5. No company ever sees another's raw transactions

### What Gets Produced

| Output File | Description |
|---|---|
| `demand_model.ubj` | Trained XGBoost demand forecasting model |
| `benchmark_report.csv` | Consortium accuracy vs. individual company models |
| `feature_importance.csv` | Top predictive features ranked by gain |
| `monthly_forecast.csv` | 3-month forward demand prediction |

## Dataset

**UCI Online Retail II** — Real transactions from a UK-based online retailer (2009–2011).

- ~1 million transactions across 4,000+ customers and 4,000+ products
- Covers 43 countries
- Features: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country

The `prepare_datasets.py` tool splits this into 3 company datasets by customer ID, simulating the real-world scenario where each retailer owns a disjoint slice of the market.

**Source:** [Kaggle — Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Trusted Execution Environment (TEE)           │
│                   AMD SEV-SNP / Intel TDX Hardware               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    In-Enclave Agent                        │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │ Company 1   │  │ Company 2   │  │ Company 3   │      │  │
│  │  │ Dataset     │  │ Dataset     │  │ Dataset     │      │  │
│  │  │ (encrypted) │  │ (encrypted) │  │ (encrypted) │      │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────���      │  │
│  │         │                │                │              │  │
│  │         └────────────────┼────────────────┘              │  │
│  │                          ▼                               │  │
│  │               ┌──────────────────┐                       │  │
│  │               │   train.py       │                       │  │
│  │               │   (Algorithm)    │                       │  │
│  │               └────────┬─────────┘                       │  │
│  │                        ▼                                 │  │
│  │           ┌────────────────────────┐                     │  │
│  │           │  Results:              │                     │  │
│  │           │  • demand_model.ubj    │                     │  │
│  │           │  • benchmark_report    │                     │  │
│  │           │  • feature_importance  │                     │  │
│  │           │  • monthly_forecast    │                     │  │
│  │           └────────────────────────┘                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Memory encrypted by hardware • Host/cloud has zero access      │
└─────────────────────────────────────────────────────────────────┘
         ▲               ▲               ▲              │
    aTLS │          aTLS │          aTLS │              │ aTLS
         │               │               │              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ Company 1 │  │ Company 2 │  │ Company 3 │  │  Result   │
   │ (Data     │  │ (Data     │  │ (Data     │  │  Consumer │
   │  Provider)│  │  Provider)│  │  Provider)│  │           │
   └───────────┘  └───────────┘  └───────────┘  └───────────┘
```

**Key security guarantees:**

- **aTLS (Attested TLS):** Each participant verifies the TEE's hardware attestation before uploading. The cryptographic quote proves the enclave is genuine AMD SEV-SNP/Intel TDX hardware running the exact agreed-upon algorithm.
- **Memory encryption:** All data inside the TEE is encrypted by the CPU. The cloud provider, hypervisor, and host OS have zero access.
- **No raw data exits:** Only aggregated model weights and statistical reports leave the enclave. Individual transaction records are destroyed.

## Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Install

Fetch the data from Kaggle — [Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) dataset:

```bash
kaggle datasets download -d mashlyn/online-retail-ii-uci
```

To run the above command you need [kaggle cli](https://github.com/Kaggle/kaggle-api) installed and API credentials set up. Follow [this documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#kaggle-api).

You will get `online-retail-ii-uci.zip` in the folder.

Prepare the 3 company datasets:

```bash
python tools/prepare_datasets.py online-retail-ii-uci.zip -d datasets
```

Expected output:

```
Loaded 1067371 rows from online_retail_II.xlsx
After cleaning: 824364 rows, 4384 customers
Company 1: 273421 transactions, 1461 customers, 38 countries
Company 2: 276583 transactions, 1462 customers, 40 countries
Company 3: 274360 transactions, 1461 customers, 39 countries

Dataset preparation complete. 3 company datasets saved to 'datasets/'
```

## Train Model (Local)

To train the consortium model locally:

```bash
python train.py
```

The script loads all company CSVs from `datasets/`, builds time-series features, trains a consortium XGBoost model on the combined data, then benchmarks it against individual company models. Results are saved to `results/`.

## Test Model (Local)

Analyze the results and generate visualizations:

```bash
python predict.py
```

Output includes benchmark comparisons, feature importance charts, and demand forecast summaries.

## Testing with Prism (Multi-Party)

Prism provides a web-based interface for managing multi-party computations with full role-based access control. This is the recommended approach for enterprise deployments.

### Prerequisites

1. **Clone and start Prism:**

   ```bash
   git clone https://github.com/ultravioletrs/prism.git
   cd prism
   make run
   ```

2. **Prepare datasets** (follow the same steps as above)

3. **Build Cocos artifacts and generate keys:**

   ```bash
   cd cocos
   make all
   ./build/cocos-cli keys -k="rsa"
   ```

### Multi-Party Setup in Prism

This section shows how to configure a true multi-party computation where different participants have distinct roles:

#### 1. Create User Accounts

Create accounts for each participant in the consortium:

- **Algorithm Provider** — The neutral data scientist supplying the training code
- **Company 1 Data Provider** — Uploads company_1.csv
- **Company 2 Data Provider** — Uploads company_2.csv
- **Company 3 Data Provider** — Uploads company_3.csv
- **Result Consumer** — The consortium administrator who receives the output

#### 2. Create a Workspace

Create a workspace representing the consortium (e.g., "Retail Demand Consortium").

#### 3. Create a CVM

Create a Confidential VM and wait for it to come online.

#### 4. Create the Computation

Create the computation and set the name and description (e.g., "Q1 Demand Forecast — Multi-Retailer Consortium").

Generate sha3-256 checksums for all assets:

```bash
./build/cocos-cli checksum ../ai/enterprise-analytics/train.py
./build/cocos-cli checksum ../ai/enterprise-analytics/datasets/company_1.csv
./build/cocos-cli checksum ../ai/enterprise-analytics/datasets/company_2.csv
./build/cocos-cli checksum ../ai/enterprise-analytics/datasets/company_3.csv
```

#### 5. Add Computation Assets

Add the algorithm and dataset assets in Prism using the file names and checksums:

| Asset | File Name | Role |
|---|---|---|
| Algorithm | `train.py` | Algorithm Provider |
| Dataset 1 | `company_1.csv` | Data Provider (Company 1) |
| Dataset 2 | `company_2.csv` | Data Provider (Company 2) |
| Dataset 3 | `company_3.csv` | Data Provider (Company 3) |

#### 6. Assign Participant Roles

Use Prism's computation roles to assign each participant:

- The **Algorithm Provider** can upload the algorithm but cannot see the datasets
- Each **Data Provider** can upload only their own dataset
- The **Result Consumer** can download results but cannot see raw data or the algorithm

This enforces strict separation of concerns — no single participant has access to all assets.

#### 7. Upload Public Keys

Each participant uploads their public key (generated by `cocos-cli`) to enable encrypted uploads and result retrieval.

### Run the Computation

1. **Click "Run Computation"** and select an available CVM

2. **Copy the agent port** and export it:

   ```bash
   export AGENT_GRPC_URL=localhost:<AGENT_PORT>
   ```

3. **Algorithm Provider uploads the algorithm:**

   ```bash
   ./build/cocos-cli algo ../ai/enterprise-analytics/train.py ./private.pem -a python -r ../ai/enterprise-analytics/requirements.txt
   ```

4. **Each company uploads their dataset independently:**

   Company 1:
   ```bash
   ./build/cocos-cli data ../ai/enterprise-analytics/datasets/company_1.csv ./private.pem
   ```

   Company 2:
   ```bash
   ./build/cocos-cli data ../ai/enterprise-analytics/datasets/company_2.csv ./private.pem
   ```

   Company 3:
   ```bash
   ./build/cocos-cli data ../ai/enterprise-analytics/datasets/company_3.csv ./private.pem
   ```

5. **Monitor the computation** through the Prism web interface. Events will show algorithm upload, data uploads, computation running, and completion.

6. **Result Consumer downloads the results:**

   ```bash
   ./build/cocos-cli result ./private.pem
   ```

### Analyze Results

```bash
cp results.zip ../ai/enterprise-analytics/
cd ../ai/enterprise-analytics
unzip results.zip -d results
python predict.py
```

## Understanding the Security Model

### Attested TLS (aTLS) — How It Works

```
                                  aTLS Handshake
┌──────────┐                                          ┌──────────────┐
│  Client   │  1. TLS ClientHello ──────────────────▶ │  TEE Agent   │
│  (Data    │                                          │  (Enclave)   │
│  Provider)│  2. TLS ServerHello + Attestation  ◀──  │              │
│           │     Quote (signed by CPU hardware)       │              │
│           │                                          │              │
│           │  3. Client VERIFIES:                     │              │
│           │     ✓ Genuine AMD/Intel hardware         │              │
│           │     ✓ Correct software measurement       │              │
│           │     ✓ Enclave not tampered with          │              │
│           │                                          │              │
│           │  4. Encrypted data upload ─────────────▶ │  [Data is    │
│           │     (only if attestation passed)          │   decrypted  │
│           │                                          │   ONLY inside│
│           │                                          │   enclave]   │
└──────────┘                                          └──────────────┘
```

With aTLS, the TLS handshake includes a **hardware attestation quote** generated by the CPU's secure processor. This quote:

1. **Proves the hardware is genuine** — The attestation is signed by the CPU manufacturer's root key
2. **Includes a measurement of the software** — A SHA-256 hash of the entire software stack loaded into the enclave
3. **Cannot be forged** — Even a compromised hypervisor or cloud administrator cannot generate a valid quote

### Multi-Party Data Flow

```
 Company 1          Company 2          Company 3
     │                   │                   │
     │   aTLS + upload   │   aTLS + upload   │   aTLS + upload
     ▼                   ▼                   ▼
┌─────────────────────────────────────────────────┐
│                    TEE Enclave                   │
│                                                  │
│  company_1.csv  company_2.csv  company_3.csv    │
│       │              │              │            │
│       └──────────────┼──────────────┘            │
│                      ▼                           │
│              Combined DataFrame                  │
│                      │                           │
│              Feature Engineering                 │
│                      │                           │
│              XGBoost Training                    │
│                      │                           │
│              ┌─��─────┴───────┐                   │
│              ▼               ▼                   │
│        demand_model    benchmark_report          │
│        (no raw data)   (aggregated stats)        │
│                                                  │
│  ⚠ Raw CSVs destroyed after computation          │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼ aTLS download
                 Result Consumer
```

**Critical security properties:**

- Each company's data is encrypted in transit (aTLS) and at rest (hardware memory encryption)
- The algorithm cannot exfiltrate raw data — only the computation manifest's approved outputs leave the enclave
- Even the cloud provider and Prism platform operators have zero access to the data inside the TEE
- The benchmark report contains only aggregate metrics (MAE, RMSE, R²) — not individual transaction data

## Notes

- **Memory:** 8GB is sufficient for the Online Retail II dataset. Increase for larger enterprise datasets.
- **Runtime:** Training completes in approximately 2–5 minutes depending on hardware.
- **Scaling:** The same architecture supports any number of data providers. Simply add more `-data-paths` entries or additional Data Provider roles in Prism.
- **aTLS on real hardware:** When running on AMD SEV-SNP or Intel TDX servers, set `MANAGER_QEMU_ENABLE_SEV_SNP=true` and `-attested-tls-bool true` to get hardware-backed attestation. In development mode, attestation is simulated.
- **Dataset alternatives:** Any tabular sales/transaction dataset can be substituted. The feature engineering in `train.py` expects columns: `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, `Country`.

