# Healthcare — Multi-Hospital Patient Readmission Prediction

Confidential multi-party analytics for healthcare. Three competing hospitals collaboratively train a patient readmission risk model inside a Trusted Execution Environment (TEE)—each contributes HIPAA-protected patient records, but **no hospital ever sees another's raw data**.

This example demonstrates:

- **Secure Computation (aTLS)** — Attested TLS verifies the TEE hardware and software stack before any patient data is uploaded
- **Multi-Party Computation** — Three independent hospitals each upload proprietary EHR datasets into the same encrypted enclave
- **Real-World Data** — Uses the [UCI Diabetes 130-US Hospitals](https://www.kaggle.com/datasets/jimschacko/10-years-diabetes-dataset) dataset (~100K real patient encounters) split across simulated hospitals
- **Healthcare Value** — Benchmark proves the consortium model outperforms any single-hospital model at predicting 30-day readmissions

## Table of Contents

- [Scenario](#scenario)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Setup Virtual Environment](#setup-virtual-environment)
- [Install](#install)
- [Train Model (Local)](#train-model-local)
- [Test Model (Local)](#test-model-local)
- [Testing with Cocos (aTLS)](#testing-with-cocos-atls)
- [Testing with Prism (Multi-Party)](#testing-with-prism-multi-party)
- [Notes](#notes)

## Scenario

Three hospitals—**Hospital 1**, **Hospital 2**, and **Hospital 3**—each hold protected health information (PHI) governed by HIPAA regulations. No hospital can legally or ethically share raw patient records with another institution or a third party.

However, they all recognize that a **consortium readmission model** trained on combined patient populations would be far more accurate than any model they could train alone. Prism AI makes this possible:

1. A **neutral Algorithm Provider** supplies the training code (`train.py`)
2. Each hospital acts as a **Data Provider**, uploading encrypted patient datasets into the TEE
3. The TEE runs the algorithm over all three datasets simultaneously
4. Only the **aggregated results** (trained model + benchmark report) exit the enclave
5. No hospital ever sees another's raw patient records

### What Gets Produced

| Output File | Description |
|---|---|
| `readmission_model.ubj` | Trained XGBoost readmission risk classifier |
| `benchmark_report.csv` | Consortium accuracy vs. individual hospital models |
| `feature_importance.csv` | Top predictive features ranked by gain |
| `risk_distribution.csv` | Patient readmission risk bucket distribution |

## Dataset

**UCI Diabetes 130-US Hospitals** — Real patient encounters from 130 US hospitals (1999–2008).

- ~100,000 patient encounters across 10 years
- 50+ features including demographics, diagnoses, medications, and lab results
- Target: readmission within 30 days (binary classification)
- Features: race, gender, age, admission type, discharge disposition, diagnoses (ICD-9), 23 medication columns, lab results, number of procedures, time in hospital

The `prepare_datasets.py` tool splits this into 3 hospital datasets by patient ID, simulating the real-world scenario where each hospital owns a disjoint slice of the patient population.

**Source:** [Kaggle — 10 Years Diabetes Dataset](https://www.kaggle.com/datasets/jimschacko/10-years-diabetes-dataset)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Trusted Execution Environment (TEE)           │
│                   AMD SEV-SNP / Intel TDX Hardware               │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    In-Enclave Agent                        │  │
│  │                                                           │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │ Hospital 1  │  │ Hospital 2  │  │ Hospital 3  │      │  │
│  │  │ Patient EHR │  │ Patient EHR │  │ Patient EHR │      │  │
│  │  │ (encrypted) │  │ (encrypted) │  │ (encrypted) │      │  │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │  │
│  │         │                │                │              │  │
│  │         └────────────────┼────────────────┘              │  │
│  │                          ▼                               │  │
│  │               ┌──────────────────┐                       │  │
│  │               │   train.py       │                       │  │
│  │               │   (Algorithm)    │                       │  │
│  │               └────────┬─────────┘                       │  │
│  │                        ▼                                 │  │
│  │           ┌────────────────────────────┐                 │  │
│  │           │  Results:                  │                 │  │
│  │           │  • readmission_model.ubj   │                 │  │
│  │           │  • benchmark_report        │                 │  │
│  │           │  • feature_importance      │                 │  │
│  │           │  • risk_distribution       │                 │  │
│  │           └────────────────────────────┘                 │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Memory encrypted by hardware • Host/cloud has zero access      │
└─────────────────────────────────────────────────────────────────┘
         ▲               ▲               ▲              │
    aTLS │          aTLS │          aTLS │              │ aTLS
         │               │               │              ▼
   ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
   │Hospital 1 │  │Hospital 2 │  │Hospital 3 │  │  Result   │
   │ (Data     │  │ (Data     │  │ (Data     │  │  Consumer │
   │  Provider)│  │  Provider)│  │  Provider)│  │           │
   └───────────┘  └───────────┘  └───────────┘  └───────────┘
```

**Key security guarantees:**

- **aTLS (Attested TLS):** Each hospital verifies the TEE's hardware attestation before uploading. The cryptographic quote proves the enclave is genuine AMD SEV-SNP/Intel TDX hardware running the exact agreed-upon algorithm.
- **Memory encryption:** All patient data inside the TEE is encrypted by the CPU. The cloud provider, hypervisor, and host OS have zero access.
- **HIPAA compliance:** No raw patient records exit the enclave. Only aggregated model weights and statistical reports leave the enclave. Individual patient encounters are destroyed after computation.

## Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Install

Fetch the data from Kaggle — [10 Years Diabetes Dataset](https://www.kaggle.com/datasets/jimschacko/10-years-diabetes-dataset):

```bash
kaggle datasets download -d jimschacko/10-years-diabetes-dataset
```

To run the above command you need [kaggle cli](https://github.com/Kaggle/kaggle-api) installed and API credentials set up. Follow [this documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#kaggle-api).

You will get `10-years-diabetes-dataset.zip` in the folder.

Prepare the 3 hospital datasets:

```bash
python tools/prepare_datasets.py 10-years-diabetes-dataset.zip -d datasets
```

Expected output:

```
Loaded 101766 rows from diabetic_data.csv
After cleaning: 69651 rows, 69651 unique patients
Hospital 1: 23217 encounters, 23217 patients, 11.2% 30-day readmission rate
Hospital 2: 23217 encounters, 23217 patients, 11.0% 30-day readmission rate
Hospital 3: 23217 encounters, 23217 patients, 11.3% 30-day readmission rate

Dataset preparation complete. 3 hospital datasets saved to 'datasets/'
```

## Train Model (Local)

To train the consortium model locally:

```bash
python train.py
```

The script loads all hospital CSVs from `datasets/`, engineers clinical features (medication changes, visit history, diagnosis codes), trains a consortium XGBoost classifier on the combined data, then benchmarks it against individual hospital models. Results are saved to `results/`.

## Test Model (Local)

Analyze the results and generate visualizations:

```bash
python predict.py
```

Output includes benchmark comparisons, feature importance charts, and risk distribution summaries.

## Testing with Prism (Multi-Party)

Prism provides a web-based interface for managing multi-party computations with full role-based access control. This is the recommended approach for healthcare deployments where HIPAA compliance is required.

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
- **Hospital 1 Data Provider** — Uploads hospital_1.csv (HIPAA-covered entity)
- **Hospital 2 Data Provider** — Uploads hospital_2.csv (HIPAA-covered entity)
- **Hospital 3 Data Provider** — Uploads hospital_3.csv (HIPAA-covered entity)
- **Result Consumer** — The consortium administrator who receives the output

#### 2. Create a Workspace

Create a workspace representing the consortium (e.g., "Hospital Readmission Consortium").

#### 3. Create a CVM

Create a Confidential VM and wait for it to come online.

#### 4. Create the Computation

Create the computation and set the name and description (e.g., "30-Day Readmission Risk — Multi-Hospital Consortium").

Generate sha3-256 checksums for all assets:

```bash
./build/cocos-cli checksum ../ai/healthcare/train.py
./build/cocos-cli checksum ../ai/healthcare/datasets/hospital_1.csv
./build/cocos-cli checksum ../ai/healthcare/datasets/hospital_2.csv
./build/cocos-cli checksum ../ai/healthcare/datasets/hospital_3.csv
```

#### 5. Add Computation Assets

Add the algorithm and dataset assets in Prism using the file names and checksums:

| Asset | File Name | Role |
|---|---|---|
| Algorithm | `train.py` | Algorithm Provider |
| Dataset 1 | `hospital_1.csv` | Data Provider (Hospital 1) |
| Dataset 2 | `hospital_2.csv` | Data Provider (Hospital 2) |
| Dataset 3 | `hospital_3.csv` | Data Provider (Hospital 3) |

#### 6. Assign Participant Roles

Use Prism's computation roles to assign each participant:

- The **Algorithm Provider** can upload the algorithm but cannot see the patient datasets
- Each **Hospital Data Provider** can upload only their own dataset
- The **Result Consumer** can download results but cannot see raw patient data or the algorithm

This enforces strict separation of concerns — no single participant has access to all assets. This is critical for HIPAA compliance.

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
   ./build/cocos-cli algo ../ai/healthcare/train.py ./private.pem -a python -r ../ai/healthcare/requirements.txt
   ```

4. **Each hospital uploads their dataset independently:**

   Hospital 1:
   ```bash
   ./build/cocos-cli data ../ai/healthcare/datasets/hospital_1.csv ./private.pem
   ```

   Hospital 2:
   ```bash
   ./build/cocos-cli data ../ai/healthcare/datasets/hospital_2.csv ./private.pem
   ```

   Hospital 3:
   ```bash
   ./build/cocos-cli data ../ai/healthcare/datasets/hospital_3.csv ./private.pem
   ```

5. **Monitor the computation** through the Prism web interface. Events will show algorithm upload, data uploads, computation running, and completion.

6. **Result Consumer downloads the results:**

   ```bash
   ./build/cocos-cli result ./private.pem
   ```

### Analyze Results

```bash
cp results.zip ../ai/healthcare/
cd ../ai/healthcare
unzip results.zip -d results
python predict.py
```

## Understanding the Security Model

### Why This Matters for Healthcare

Traditional approaches to multi-hospital ML require one of:

1. **Data sharing agreements** — Hospitals send raw PHI to a central location. This creates massive HIPAA liability, requires BAAs, and is often rejected by legal/compliance teams.
2. **Federated learning** — Each hospital trains locally and shares model gradients. But gradient inversion attacks can reconstruct patient records from shared gradients.
3. **Synthetic data** — Hospitals generate fake data that mimics their real distributions. But synthetic data loses rare-but-critical edge cases and cannot guarantee privacy.

**Confidential computing solves all three problems:**

- Raw patient data is encrypted in hardware memory — the cloud provider cannot access it
- The algorithm runs inside a TEE with a cryptographically verifiable software stack
- Only aggregate outputs (model weights, statistical metrics) leave the enclave
- No gradient sharing, no synthetic data, no data movement to third parties

### Attested TLS (aTLS) — How It Works

```
                                  aTLS Handshake
┌──────────┐                                          ┌──────────────┐
│  Client   │  1. TLS ClientHello ──────────────────▶ │  TEE Agent   │
│  (Hospital│                                          │  (Enclave)   │
│   IT Dept)│  2. TLS ServerHello + Attestation  ◀──  │              │
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

### Multi-Party Data Flow

```
 Hospital 1         Hospital 2         Hospital 3
     │                   │                   │
     │   aTLS + upload   │   aTLS + upload   │   aTLS + upload
     ▼                   ▼                   ▼
┌─────────────────────────────────────────────────┐
│                    TEE Enclave                   │
│                                                  │
│  hospital_1.csv  hospital_2.csv  hospital_3.csv │
│       │              │              │            │
│       └──────────────┼──────────────┘            │
│                      ▼                           │
│              Combined DataFrame                  │
│                      │                           │
│              Feature Engineering                 │
│              (medications, diagnoses, visits)     │
│                      │                           │
│              XGBoost Classification              │
│              (readmission within 30 days)        │
│                      │                           │
│              ┌───────┴───────┐                   │
│              ▼               ▼                   │
│     readmission_model   benchmark_report         │
│     (no patient data)   (aggregated stats)       │
│                                                  │
│  ⚠ Raw CSVs destroyed after computation          │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼ aTLS download
                 Result Consumer
```

**Critical security properties:**

- Each hospital's patient data is encrypted in transit (aTLS) and at rest (hardware memory encryption)
- The algorithm cannot exfiltrate raw data — only the computation manifest's approved outputs leave the enclave
- Even the cloud provider and Prism platform operators have zero access to the data inside the TEE
- The benchmark report contains only aggregate metrics (Accuracy, AUC, F1) — not individual patient records
- Fully compatible with HIPAA Safe Harbor de-identification requirements