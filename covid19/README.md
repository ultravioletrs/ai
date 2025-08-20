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

Inference can be done using `predict.py`. Anyfile in the `datasets/test` directory can be used for testing.

```bash
python predict.py --model results/model.pth --image datasets/test/COVID/COVID-2.png
```

## Testing with Cocos

Make sure you have the Cocos repository cloned and eos buildroot installed. This can be done by following the instructions in the [Cocos Documentation](https://docs.cocos.ultraviolet.rs/getting-started/)

Clone the ai repository which has the COVID-19 model:

```bash
git clone https://github.com/ultravioletrs/ai.git
```

```bash
cd ai/covid19
```

Download the data from [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) dataset.

You can install kaggle-cli to download the dataset:

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

Zip the folders:

```bash
zip -r datasets/h1.zip datasets/h1
zip -r datasets/h2.zip datasets/h2
zip -r datasets/h3.zip datasets/h3
```

Change the directory to cocos:

```bash
cd ../../cocos
```

Build cocos artifacts:

```bash
make all
```

Before running the computation server, we need to issue certificates for the computation server and the client. This can be done by running the following commands:

```bash
./build/cocos-cli keys -k="rsa"
```

Run the computation server:

```bash
HOST=192.168.100.27 go run ./test/cvms/main.go -algo-path ../ai/covid19/train.py -public-key-path public.pem -attested-tls-bool false -data-paths ../ai/covid19/datasets/h1.zip,../ai/covid19/datasets/h2.zip,../ai/covid19/datasets/h3.zip
```

On another terminal, run manager:

```bash
cd cmd/manager
```

Make sure you have the `bzImage` and `rootfs.cpio.gz` in the `cmd/manager/img` directory.

```bash
sudo \
MANAGER_QEMU_SMP_MAXCPUS=4 \
MANAGER_QEMU_MEMORY_SIZE=25G \
MANAGER_GRPC_HOST=localhost \
MANAGER_GRPC_PORT= 7002 \
MANAGER_LOG_LEVEL=debug \
MANAGER_QEMU_ENABLE_SEV_SNP=false \
MANAGER_QEMU_OVMF_CODE_FILE=/usr/share/edk2/x64/OVMF_CODE.fd \
MANAGER_QEMU_OVMF_VARS_FILE=/usr/share/edk2/x64/OVMF_VARS.fd \
go run main.go
```

After sometime you will see the computation server will output a port number. This port number is the port on which the computation server is running. You can use this port number to run the client.

The logs will look like this:

```bash
{"time":"2025-08-20T14:36:27.156171875+03:00","level":"INFO","msg":"cvms_test_server service gRPC server listening at 192.168.100.27:7001 without TLS"}
{"time":"2025-08-20T14:45:31.016663955+03:00","level":"DEBUG","msg":"received who am on ip address 192.168.100.27:56466"}
&{message:"TEE device not found"  level:"INFO"  timestamp:{seconds:1755690331  nanos:7261868}}
&{}
&{message:"Method InitComputation for computation id 1 took 3.015µs to complete without errors"  computation_id:"1"  level:"INFO"  timestamp:{seconds:1755690331  nanos:345076481}}
&{computation_id:"1"}
&{message:"agent service gRPC server listening at 10.0.2.15:7002 without TLS"  computation_id:"1"  level:"INFO"  timestamp:{seconds:1755690331  nanos:345224719}}
&{event_type:"ReceivingAlgorithm"  timestamp:{seconds:1755690331  nanos:345227613}  computation_id:"1"  originator:"agent"  status:"InProgress"}
```

On another terminal, upload the artifacts to the computation server:

```bash
export AGENT_GRPC_URL=localhost:<port_number>
```

```bash
./build/cocos-cli algo ../ai/covid19/train.py ./private.pem -a python -r ../ai/covid19/requirements.txt
```

Upload the data to the computation server:

```bash
./build/cocos-cli data ../ai/covid19/datasets/h1.zip ./private.pem -d
./build/cocos-cli data ../ai/covid19/datasets/h2.zip ./private.pem -d
./build/cocos-cli data ../ai/covid19/datasets/h3.zip ./private.pem -d
```

When the results are ready, download the results:

```bash
./build/cocos-cli result ./private.pem
```

The above will generate a `results.zip` file. Copy this file to the ai directory:

```bash
cp results.bin ../ai/covid19/
```

Test the model with the test data:

```bash
cd ../ai/covid19
```

```bash
unzip results.zip -d results
```

The image can be any image from the test dataset:

```bash
python predict.py --model results/model.pth --image datasets/test/COVID/COVID-2.png
```

## Testing with Prism

Make sure you have the Cocos repository cloned and eos buildroot installed. This can be done by following the instructions in the [Cocos Documentation](https://docs.cocos.ultraviolet.rs/getting-started/)

Clone the ai repository which has the COVID-19 model:

```bash
git clone https://github.com/ultravioletrs/ai.git
```

```bash
cd ai/covid19
```

Download the data from [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) dataset.

You can install kaggle-cli to download the dataset:

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

Zip the folders:

```bash
zip -r datasets/h1.zip datasets/h1
zip -r datasets/h2.zip datasets/h2
zip -r datasets/h3.zip datasets/h3
```

Start prism

```bash
git clone https://github.com/ultravioletrs/prism.git
```

```bash
cd prism
```

```bash
make run
```

The following recording will demonstrate how to setup prism - https://jam.dev/c/8067f697-4eaa-407f-875a-17119e4f3901

Build cocos artifacts:

```bash
make all
```

Before running the computation server, we need to issue certificates for the computation server and the client. This can be done by running the following commands:

```bash
./build/cocos-cli keys -k="rsa"
```

You need to have done the following:

- Create a user at `localhost:9095`
- Create a workspace
- Login to the created workspace
- Create a cvm and wait for it to come online

- Create the covid computation. To get the filehash for all the files go to `cocos` folder and use the cocos-cli. For the file names use `h1.zip`, `h2.zip`, `h3.zip` and `train.py`

  ```bash
  ./build/cocos-cli checksum ../ai/covid19/datasets/h1.zip
  ```

  ```bash
  ./build/cocos-cli checksum ../ai/covid19/datasets/h2.zip
  ```

  ```bash
  ./build/cocos-cli checksum ../ai/covid19/datasets/h3.zip
  ```

  ```bash
  ./build/cocos-cli checksum ../ai/covid19/train.py
  ```

- After the computation has been created upload your public key generate by `cocos-cli`. This key will enable you to upload the datatsets and algorithms and also download the results.

  ```bash
  ./build/cocos-cli keys
  ```

- Click run computation and wait for the vm to be provisioned.Copy the aggent port number and export `AGENT_GRPC_URL`

  ```bash
  export AGENT_GRPC_URL=localhost:<port_number>
  ```

- After vm has been provisioned upload the datasets and the algorithm

  ```bash
  ./build/cocos-cli algo ../ai/covid19/train.py ./private.pem -a python -r ../ai/covid19/requirements.txt
  ```

  ```bash
  ./build/cocos-cli data ../ai/covid19/datasets/h1.zip ./private.pem -d
  ```

  ```bash
  ./build/cocos-cli data ../ai/covid19/datasets/h2.zip ./private.pem -d
  ```

  ```bash
  ./build/cocos-cli data ../ai/covid19/datasets/h3.zip ./private.pem -d
  ```

- The computation will run and you will get an event that the results are ready. You can download the results by running the following command:

  ```bash
  ./build/cocos-cli results ./private.pem
  ```

The above will generate a `results.zip` file. Copy this file to the ai directory:

```bash
cp results.bin ../ai/covid19/
```

Test the model with the test data:

```bash
cd ../ai/covid19
```

```bash
unzip results.zip -d results
```

The image can be any image from the test dataset:

```bash
python predict.py --model results/model.pth --image datasets/test/COVID/COVID-2.png
```
