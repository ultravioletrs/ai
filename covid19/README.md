# COVID-19

PyTorch-based ML model for detecting COVID-19 based on chest X-ray images. The model classifies images into three categories: Normal, Viral Pneumonia, and COVID-19.

## Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

PyTorch can be installed from the [PyTorch website](https://pytorch.org/get-started/locally/). Follow the instructions to match your specific system configuration (e.g., CUDA version, OS).

For example, to install on a linux system with ROCm support, you can run:

```bash
pip3 install torch~=2.6.0 torchvision~=0.21.0 --index-url https://download.pytorch.org/whl/rocm6.0
```

## Install

Fetch the data from Kaggle - [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) dataset

```bash
kaggle datasets download -d tawsifurrahman/covid19-radiography-database
```

To run the above command you would need [kaggle cli](https://github.com/Kaggle/kaggle-api) installed and API credentials setup. This can be done by following [this documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#kaggle-api).

You will get `covid19-radiography-database.zip` in the folder

Run:

```bash
python tools/prepare_datasets.py covid19-radiography-database.zip -d datasets
```

This will create `datasets` directory with datasets divided in 3 hospitals (`h1`, `h2` and `h3`) plus additionally a `test` dataset, which will be used to test the produced model.

## Train Model

To do the training, execute:

```bash
python train.py
```

## Test Model

Inference can be done using `predict.py`. Any file in the `datasets/test` directory can be used for testing.

```bash
python predict.py --model results/model.pth --image datasets/test/COVID/COVID-2.png
```

## Testing with Cocos

Make sure you have the Cocos repository cloned and set up. This can be done by following the instructions in the [Cocos Documentation](https://docs.cocos.ultraviolet.rs/getting-started/)

### Prerequisites

1. **Clone the repositories:**

   ```bash
   git clone https://github.com/ultravioletrs/cocos.git
   git clone https://github.com/ultravioletrs/ai.git
   ```

2. **Navigate to the COVID-19 directory:**

   ```bash
   cd ai/covid19
   ```

3. **Download and prepare the dataset:**

   Install kaggle-cli to download the dataset:

   ```bash
   pip install kaggle
   ```

   Set the [kaggle API key](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials) and download the dataset:

   ```bash
   kaggle datasets download -d tawsifurrahman/covid19-radiography-database
   ```

   Prepare the dataset:

   ```bash
   python tools/prepare_datasets.py covid19-radiography-database.zip -d datasets
   ```

   Zip the hospital datasets for uploading:

   ```bash
   zip -r datasets/h1.zip datasets/h1
   zip -r datasets/h2.zip datasets/h2
   zip -r datasets/h3.zip datasets/h3
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

From your cocos directory, start the CVMS server with the COVID-19 algorithm and datasets:

```bash
cd cocos
HOST=<YOUR_HOST_IP> go run ./test/cvms/main.go -algo-path ../ai/covid19/train.py -public-key-path public.pem -attested-tls-bool false -data-paths ../ai/covid19/datasets/h1.zip,../ai/covid19/datasets/h2.zip,../ai/covid19/datasets/h3.zip
```

Expected output:

```bash
{"time":"...","level":"INFO","msg":"cvms_test_server service gRPC server listening at <YOUR_HOST_IP>:7001 without TLS"}
```

#### Start the Manager

Navigate to the cocos/cmd/manager directory and start the Manager (increase memory for COVID-19 training):

```bash
cd cocos/cmd/manager
sudo \
MANAGER_QEMU_SMP_MAXCPUS=4 \
MANAGER_QEMU_MEMORY_SIZE=25G \
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

### Create CVM and Upload COVID-19 Algorithm

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

#### Upload COVID-19 Algorithm

From your cocos directory:

```bash
./build/cocos-cli algo ../ai/covid19/train.py ./private.pem -a python -r ../ai/covid19/requirements.txt
```

Expected output:

```bash
🔗 Connected to agent  without TLS
Uploading algorithm file: ../ai/covid19/train.py
🚀 Uploading algorithm [███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████] [100%]                               
Successfully uploaded algorithm! ✔ 
```

#### Upload COVID-19 Datasets

Upload each hospital dataset:

```bash
./build/cocos-cli data ../ai/covid19/datasets/h1.zip ./private.pem -d
./build/cocos-cli data ../ai/covid19/datasets/h2.zip ./private.pem -d
./build/cocos-cli data ../ai/covid19/datasets/h3.zip ./private.pem -d
```

Expected output for each upload:

```bash
🔗 Connected to agent  without TLS
Uploading dataset: ../ai/covid19/datasets/h1.zip
📦 Uploading data [██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████] [100%]                               
Successfully uploaded dataset! ✔ 
```

Watch the CVMS server logs for training progress. The COVID-19 training process may take significant time due to the large dataset and model complexity. Look for completion messages:

```bash
&{message:"Method Data took ... to complete without errors"  computation_id:"1"  level:"INFO"  timestamp:{...}}
&{event_type:"Running"  timestamp:{...}  computation_id:"1"  originator:"agent"  status:"Starting"}
&{event_type:"Running"  timestamp:{...}  computation_id:"1"  originator:"agent"  status:"Completed"}
&{event_type:"ConsumingResults"  timestamp:{...}  computation_id:"1"  originator:"agent"  status:"Ready"}
```

#### Download COVID-19 Results

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

1. Copy results to the COVID-19 directory:

   ```bash
   cp results.zip ../ai/covid19/
   ```

2. Navigate to COVID-19 directory and extract results:

   ```bash
   cd ../ai/covid19
   unzip results.zip -d results
   ```

3. Test the model with test data (use any image from the test dataset):

   ```bash
   python predict.py --model results/model.pth --image datasets/test/COVID/COVID-2.png
   ```

Expected output will show the predicted class (Normal, Viral Pneumonia, or COVID-19).

#### Remove COVID-19 CVM

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

2. **Prepare COVID-19 datasets** (follow the same steps as in the Cocos section above)

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

### Create COVID-19 Computation

To create the computation in Prism, you'll need sha3-256 checksums for all datasets and the algorithm. Generate these from the cocos folder:

```bash
./build/cocos-cli checksum ../ai/covid19/datasets/h1.zip
./build/cocos-cli checksum ../ai/covid19/datasets/h2.zip
./build/cocos-cli checksum ../ai/covid19/datasets/h3.zip
./build/cocos-cli checksum ../ai/covid19/train.py
```

Use the file names `h1.zip`, `h2.zip`, `h3.zip`, and `train.py` when creating the computation asset in Prism. Link the assets to the computation.

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
   ./build/cocos-cli algo ../ai/covid19/train.py ./private.pem -a python -r ../ai/covid19/requirements.txt
   ./build/cocos-cli data ../ai/covid19/datasets/h1.zip ./private.pem
   ./build/cocos-cli data ../ai/covid19/datasets/h2.zip ./private.pem
   ./build/cocos-cli data ../ai/covid19/datasets/h3.zip ./private.pem
   ```

4. **Monitor the computation** through the Prism interface until you receive an event indicating results are ready

5. **Download the results:**

   ```bash
   ./build/cocos-cli result ./private.pem
   ```

### Test Results

Follow the same testing steps as in the Cocos section:

```bash
cp results.zip ../ai/covid19/
cd ../ai/covid19
unzip results.zip -d results
python predict.py --model results/model.pth --image datasets/test/COVID/COVID-2.png
```

## Notes

- The COVID-19 model training is computationally intensive and may require significant time and resources
- Ensure adequate memory allocation (25GB recommended) when running the Manager
- The model works with chest X-ray images and classifies them into three categories
- Test images are available in the `datasets/test` directory after running the preparation script
