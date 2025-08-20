# Remaining Useful Life (RUL) Prediction with LSTM

This repository contains code and resources for predicting the Remaining Useful Life (RUL) of machinery using Long Short-Term Memory (LSTM) neural networks. The dataset used for this project is provided by NASA and was downloaded from Kaggle.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Virtual Environment](#setup-virtual-environment)
- [Install](#install)
- [Train Model](#train-model)
- [Test Model](#test-model)
- [Results](#results)
- [Testing with Cocos](#testing-with-cocos)
- [Testing with Prism](#testing-with-prism)
- [Notes](#notes)

## Introduction

Predicting the Remaining Useful Life (RUL) of machinery is crucial for maintenance planning and avoiding unexpected failures. This project leverages a Long Short-Term Memory (LSTM) neural network to predict RUL based on sensor data.

## Dataset

The dataset used in this project is from NASA's Prognostics Data Repository, available on Kaggle. It consists of sensor measurements from machinery over time, including data from various operational conditions.

### Experimental Scenario

This project serves as an experimental exploration into predictive maintenance using machine learning techniques. While the results are promising, it's important to note that this is an experimental setup. Download datasets:

- [NASA Dataset on Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)  
- [NASA website](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

The NASA CMAPSS dataset includes the following key components:

- **Engine ID**: Identifies the specific engine. Typical sensors include Fan Speed, Pressure, Temperature, Flow rates.
- **Cycle**: The time cycle for the recorded measurements.
- **Setting 1, 2, 3**: These are the operational settings that influence the engine's performance.
- **Sensor 1 to Sensor 21**: These columns represent various sensor measurements monitoring different parameters of the engine.

Dataset is divided into four data sets:

**Data Set: FD001**
- Train trajectories: 100
- Test trajectories: 100
- Conditions: ONE (Sea Level)
- Fault Modes: ONE (HPC Degradation)

**Data Set: FD002**
- Train trajectories: 260
- Test trajectories: 259
- Conditions: SIX 
- Fault Modes: ONE (HPC Degradation)

**Data Set: FD003**
- Train trajectories: 100
- Test trajectories: 100
- Conditions: ONE (Sea Level)
- Fault Modes: TWO (HPC Degradation, Fan Degradation)

**Data Set: FD004**
- Train trajectories: 248
- Test trajectories: 249
- Conditions: SIX 
- Fault Modes: TWO (HPC Degradation, Fan Degradation)

Each data set is further divided into training and test subsets. Each time series is from a different engine, the data can be considered to be from a fleet of engines of the same type. Each engine starts with different degrees of initial wear and manufacturing variation which is unknown to the user.

The goal of using this dataset is to leverage these measurements and settings to predict the Remaining Useful Life (RUL) of the engines accurately.

## Model Architecture

The model architecture is based on LSTM (Long Short Term Memory), which is well-suited for time series prediction tasks.

## Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyTorch can be installed from the [PyTorch website](https://pytorch.org/get-started/locally/). Follow the instructions to match your specific system configuration (e.g., CUDA version, OS).

## Install

Fetch the data from Kaggle - [NASA CMAPS Dataset](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) dataset

```bash
kaggle datasets download -d behrad3d/nasa-cmaps
```

To run the above command you would need [kaggle cli](https://github.com/Kaggle/kaggle-api) installed and API credentials setup. This can be done by following [this documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#kaggle-api).

You will get `nasa-cmaps.zip` in the folder

Extract the dataset:

```bash
unzip nasa-cmaps.zip -d datasets/
```

This will create `datasets` directory with the NASA CMAPSS dataset files.

## Train Model

To do the training, execute:

```bash
python RUL_training.py
```

## Test Model

Inference can be done using `pred_model.py`. The model file will be saved after training.

```bash
python pred_model.py
```

## Results

The performance of the model is evaluated using metrics such as Mean Squared Error (Train Loss), Validation Mean Squared Error (Val Loss) and R² Score (Coefficient of Determination)

After the training process is completed, the algorithm saves the trained model to a file. This allows you to reuse the model for predictions without needing to retrain it each time. The model is saved in a `pth` format. Additionally, it generates graphs of training loss and validation loss over epochs to help visualize the model's learning process.

![](images/val-r2.png)

Visualize Predictions: After running the script, it will generate a plot showing the predicted RUL versus the actual RUL. 

This plot helps in understanding how well the model predicts the Remaining Useful Life.
Here's an example of how the plot might look:

![](images/rul.png)

In this plot:
- The x-axis represents the time cycles.
- The y-axis represents the Remaining Useful Life (RUL).

The blue curve represents the actual RUL, while the red curve represents the predicted RUL. This visualization helps assess how well the model predicts the RUL compared to the ground truth.

## Testing with Cocos

Make sure you have the Cocos repository cloned and set up. This can be done by following the instructions in the [Cocos Documentation](https://docs.cocos.ultraviolet.rs/getting-started/)

### Prerequisites

1. **Clone the repositories:**

   ```bash
   git clone https://github.com/ultravioletrs/cocos.git
   git clone https://github.com/ultravioletrs/ai.git
   ```

2. **Navigate to the RUL directory:**

   ```bash
   cd ai/rul-turbofan
   ```

3. **Download and prepare the dataset:**

   Install kaggle-cli to download the dataset:

   ```bash
   pip install kaggle
   ```

   Set the [kaggle API key](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials) and download the dataset:

   ```bash
   kaggle datasets download -d behrad3d/nasa-cmaps
   ```

   Prepare the dataset:

   ```bash
   unzip nasa-cmaps.zip -d datasets/
   ```

4. **Change to cocos directory and build artifacts:**

   ```bash
   cd ../../cocos
   make all
   ```

5. **Generate keys:**

   ```bash
   ./build/cocos-cli keys -k="rsa"
   ```

### Finding Your Host IP Address

Find your host machine's IP address (avoid using localhost):

```bash
ip a
```

Look for your network interface (e.g., wlan0 for WiFi, eth0 for Ethernet) and note the inet address. For example, if you see `192.168.1.100`, use that as your `<YOUR_HOST_IP>`.

### Start Core Services

#### Start the Computation Management Server (CVMS)

From your cocos directory, start the CVMS server with the RUL algorithm and datasets:

```bash
cd cocos
HOST=<YOUR_HOST_IP> go run ./test/cvms/main.go -algo-path ../ai/rul-turbofan/RUL_training.py -public-key-path public.pem -attested-tls-bool false -data-paths ../ai/rul-turbofan/datasets/train_FD001.txt,../ai/rul-turbofan/datasets/test_FD001.txt,../ai/rul-turbofan/datasets/RUL_FD001.txt
```

Expected output:

```bash
{"time":"...","level":"INFO","msg":"cvms_test_server service gRPC server listening at <YOUR_HOST_IP>:7001 without TLS"}
```

#### Start the Manager

Navigate to the cocos/cmd/manager directory and start the Manager:

```bash
cd cocos/cmd/manager
sudo \
MANAGER_QEMU_SMP_MAXCPUS=4 \
MANAGER_QEMU_MEMORY_SIZE=8G \
MANAGER_GRPC_HOST=localhost \
MANAGER_GRPC_PORT=7002 \
MANAGER_LOG_LEVEL=debug \
MANAGER_QEMU_ENABLE_SEV_SNP=false \
MANAGER_QEMU_OVMF_CODE_FILE=/usr/share/edk2/x64/OVMF_CODE.fd \
MANAGER_QEMU_OVMF_VARS_FILE=/usr/share/edk2/x64/OVMF_VARS.fd \
go run main.go
```

Expected output:

```bash
{"time":"...","level":"INFO","msg":"Manager started without confidential computing support"}
{"time":"...","level":"INFO","msg":"manager service gRPC server listening at localhost:7002 without TLS"}
```

### Create CVM and Upload RUL Algorithm

#### Create CVM

From your cocos directory:

```bash
export MANAGER_GRPC_URL=localhost:7002
./build/cocos-cli create-vm --log-level debug --server-url "<YOUR_HOST_IP>:7001"
```

**Important:** Note the id and port from the cocos-cli output.

Expected output:

```bash
🔗 Connected to manager using  without TLS
🔗 Creating a new virtual machine
✅ Virtual machine created successfully with id <CVM_ID> and port <AGENT_PORT>
```

Expected CVMS server output:

```bash
&{message:"Method InitComputation for computation id 1 took ... to complete without errors"  computation_id:"1"  level:"INFO"  timestamp:{...}}
&{event_type:"ReceivingAlgorithm"  timestamp:{...}  computation_id:"1"  originator:"agent"  status:"InProgress"}
&{message:"agent service gRPC server listening at 10.0.2.15:<AGENT_PORT> without TLS"  computation_id:"1"  level:"INFO"  timestamp:{...}}
```

#### Export Agent gRPC URL

Set the AGENT_GRPC_URL using the port noted in the previous step (default 6100):

```bash
export AGENT_GRPC_URL=localhost:<AGENT_PORT>
```

#### Upload RUL Algorithm

From your cocos directory:

```bash
./build/cocos-cli algo ../ai/rul-turbofan/RUL_training.py ./private.pem -a python -r ../ai/rul-turbofan/requirements.txt
```

Expected output:

```bash
🔗 Connected to agent  without TLS
Uploading algorithm file: ../ai/rul-turbofan/RUL_training.py
🚀 Uploading algorithm [███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████] [100%]                               
Successfully uploaded algorithm! ✔ 
```

#### Upload RUL Datasets

Upload each dataset file:

```bash
./build/cocos-cli data ../ai/rul-turbofan/datasets/train_FD001.txt ./private.pem
./build/cocos-cli data ../ai/rul-turbofan/datasets/test_FD001.txt ./private.pem
./build/cocos-cli data ../ai/rul-turbofan/datasets/RUL_FD001.txt ./private.pem
```

Expected output for each upload:

```bash
🔗 Connected to agent  without TLS
Uploading dataset: ../ai/rul-turbofan/datasets/train_FD001.txt
📦 Uploading data [██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████] [100%]                               
Successfully uploaded dataset! ✔ 
```

Watch the CVMS server logs for training progress. Look for completion messages:

```bash
&{message:"Method Data took ... to complete without errors"  computation_id:"1"  level:"INFO"  timestamp:{...}}
&{event_type:"Running"  timestamp:{...}  computation_id:"1"  originator:"agent"  status:"Starting"}
&{event_type:"Running"  timestamp:{...}  computation_id:"1"  originator:"agent"  status:"Completed"}
&{event_type:"ConsumingResults"  timestamp:{...}  computation_id:"1"  originator:"agent"  status:"Ready"}
```

#### Download RUL Results

From your cocos directory:

```bash
./build/cocos-cli result ./private.pem
```

Expected output:

```bash
🔗 Connected to agent  without TLS
⏳ Retrieving computation result file
📥 Downloading result [██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████] [100%]                               
Computation result retrieved and saved successfully as results.zip! ✔ 
```

#### Test the Trained Model

1. Copy results to the RUL directory:

   ```bash
   cp results.zip ../ai/rul-turbofan/
   ```

2. Navigate to RUL directory and extract results:

   ```bash
   cd ../ai/rul-turbofan
   unzip results.zip -d results
   ```

3. Test the model:

   ```bash
   python pred_model.py
   ```

Expected output will show RUL prediction plots and performance metrics.

#### Remove RUL CVM

Use the `<CVM_ID>` obtained during CVM creation:

```bash
./build/cocos-cli remove-vm <CVM_ID>
```

Expected output:

```bash
🔗 Connected to manager using  without TLS
🔗 Removing virtual machine
✅ Virtual machine removed successfully
```

## Testing with Prism

Prism provides a web-based interface for managing Cocos computations.

### Prerequisites

1. **Clone and start Prism:**

   ```bash
   git clone https://github.com/ultravioletrs/prism.git
   cd prism
   make run
   ```

2. **Prepare RUL datasets** (follow the same steps as in the Cocos section above)

3. **Build Cocos artifacts:**

   ```bash
   cd cocos
   make all
   ```

4. **Generate keys:**

   ```bash
   ./build/cocos-cli keys -k="rsa"
   ```

### Prism Setup Process

1. **Create a user account**
2. **Create a workspace**
3. **Login to the created workspace**
4. **Create a CVM** and wait for it to come online
5. **Create the computation** and set a name and description.
6. Add participants using computation roles.

### Create RUL Computation

To create the computation in Prism, you'll need sha3-256 checksums for all datasets and the algorithm. Generate these from the cocos folder:

```bash
./build/cocos-cli checksum ../ai/rul-turbofan/datasets/train_FD001.txt
./build/cocos-cli checksum ../ai/rul-turbofan/datasets/test_FD001.txt
./build/cocos-cli checksum ../ai/rul-turbofan/datasets/RUL_FD001.txt
./build/cocos-cli checksum ../ai/rul-turbofan/RUL_training.py
```

Use the file names `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, and `RUL_training.py` when creating the computation asset in Prism. Link the assets to the computation.

### Upload Public Key

After creating the computation, upload your public key generated by `cocos-cli`. This enables you to upload datasets/algorithms and download results:

```bash
cat public.pem
```

### Run Computation

1. **Click "Run Computation"** and select an available cvm.
2. **Copy the agent port number** and export it:

   ```bash
   export AGENT_GRPC_URL=localhost:<AGENT_PORT>
   ```

3. **Upload the algorithm and datasets:**

   ```bash
   ./build/cocos-cli algo ../ai/rul-turbofan/RUL_training.py ./private.pem -a python -r ../ai/rul-turbofan/requirements.txt
   ./build/cocos-cli data ../ai/rul-turbofan/datasets/train_FD001.txt ./private.pem
   ./build/cocos-cli data ../ai/rul-turbofan/datasets/test_FD001.txt ./private.pem
   ./build/cocos-cli data ../ai/rul-turbofan/datasets/RUL_FD001.txt ./private.pem
   ```

4. **Monitor the computation** through the Prism interface until you receive an event indicating results are ready

5. **Download the results:**

   ```bash
   ./build/cocos-cli result ./private.pem
   ```

### Test Results

Follow the same testing steps as in the Cocos section:

```bash
cp results.zip ../ai/rul-turbofan/
cd ../ai/rul-turbofan
unzip results.zip -d results
python pred_model.py
```

## Notes

- The RUL model training with LSTM networks is moderately resource-intensive
- Memory allocation of 8GB should be sufficient for the Manager
- The model works with time-series sensor data from aircraft engines
- Results include visualization plots showing predicted vs actual RUL values
- The dataset contains multiple fault scenarios (FD001-FD004) - this example uses FD001
- LSTM networks are particularly well-suited for sequential data and time-series prediction tasks
