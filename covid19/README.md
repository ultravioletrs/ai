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
pip3 install torch~=2.3.1 torchvision~=0.18.1 --index-url https://download.pytorch.org/whl/rocm6.0
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

Currently, the most suitable way to test it with cocos is to use the virtual machine that has AMDSEV enabled. Login to the virtual machine and run the following commands:

```bash
mkdir -p covid19-test && cd covid19-test
```

Clone the ai repository which has the COVID-19 model:

```bash
git clone https://github.com/rodneyosodo/uv-ai.git
```

Clone cocos repository:

```bash
git clone https://github.com/ultravioletrs/cocos.git
```

Copy the covid19 training file to the cocos repository:

```bash
cp ai/covid19/covid19.py cocos/
```

also copy the requirements file:

```bash
cp ai/covid19/requirements.txt cocos/
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
mv covid19-radiography-database.zip ai/covid19/
```

```bash
cd ai/covid19
```

```bash
python tools/prepare_datasets.py covid19-radiography-database.zip -d data
```

Zip the folders:

```bash
zip -r data/h1.zip data/h1
zip -r data/h2.zip data/h2
zip -r data/h3.zip data/h3
```

Copy the zipped files to cocos:

```bash
mkdir -p ../../cocos/data
```

```bash
cp data/h1.zip data/h2.zip data/h3.zip ../../cocos/data/
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
go run ./test/computations/main.go ./covid19.py public.pem false data/h1.zip data/h2.zip data/h3.zip
```

On another terminal, run manager:

```bash
cd cmd/manager
```

Copy bzImage and rootfs to the `img` directory:

```bash
mkdir -p img tmp
```

```bash
sudo cp /home/sammy/buildroot/output/images/bzImage /home/cocosai/titan/titan.cpio.gz img/
```

Rename `titan.cpio.gz` to `rootfs.cpio.gz`:

```bash
mv img/titan.cpio.gz img/rootfs.cpio.gz
```

```bash
sudo MANAGER_QEMU_USE_SUDO=true MANAGER_LOG_LEVEL=debug MANAGER_QEMU_ENABLE_SEV=false MANAGER_QEMU_ENABLE_SEV_SNP=true MANAGER_QEMU_SEV_CBITPOS=51 MANAGER_QEMU_KERNEL_HASH=true MANAGER_QEMU_CPU=EPYC-v4 MANAGER_QEMU_OVMF_CODE_FILE=/home/cocosai/danko/AMDSEV/ovmf/Build/AmdSev/DEBUG_GCC5/FV/OVMF.fd MANAGER_QEMU_BIN_PATH=/home/cocosai/danko/AMDSEV/usr/local/bin/qemu-system-x86_64 MANAGER_GRPC_URL=localhost:7001 MANAGER_QEMU_MEMORY_SIZE=25G go run main.go
```

After sometime you will see the computation server will output a port number. This port number is the port on which the computation server is running. You can use this port number to run the client.

The logs will look like this:

```bash
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1721816170 nanos:825593350} computation_id:"1" originator:"manager" status:"starting"}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1721816170 nanos:826050932} computation_id:"1" originator:"manager" status:"in-progress"}
received agent log
&{message:"char device redirected to /dev/pts/17 (label compat_monitor0)\n" computation_id:"1" level:"debug" timestamp:{seconds:1721816170 nanos:927805046}}
received agent log
&{message:"qemu-system-x86_64: warning: Number of hotpluggable cpus requested (64) exceeds the recommended cpus supported by KVM (24)\n" computation_id:"1" level:"error" timestamp:{seconds:1721816170 nanos:953823551}}
received agent log
&{message:"S" computation_id:"1" level:"debug" timestamp:{seconds:1721816172 nanos:583261451}}
received agent log
&{message:"e" computation_id:"1" level:"debug" timestamp:{seconds:1721816172 nanos:583288633}}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1721816191 nanos:936892540} computation_id:"1" originator:"manager" status:"complete"}
received runRes
&{agent_port:"46589" computation_id:"1"}
received agent log
&{message:"Transition: receivingManifest -> receivingManifest\n" computation_id:"1" level:"DEBUG" timestamp:{seconds:1721816191 nanos:933814929}}
received agent log
&{message:"agent service gRPC server listening at :7002 without TLS" computation_id:"1" level:"INFO" timestamp:{seconds:1721816191 nanos:934464476}}
received agent event
&{event_type:"receivingAlgorithm" timestamp:{seconds:1721816191 nanos:934190831} computation_id:"1" originator:"agent" status:"in-progress"}
```

On another terminal, upload the artifacts to the computation server:

```bash
export AGENT_GRPC_URL=localhost:<port_number>
```

```bash
./build/cocos-cli algo ./covid19.py ./private.pem --algorithm python -r ./requirements.txt --python-runtime python
```

Upload the data to the computation server:

```bash
./build/cocos-cli data data/h1.zip ./private.pem
./build/cocos-cli data data/h2.zip ./private.pem
./build/cocos-cli data data/h3.zip ./private.pem
```

When the results are ready, download the results:

```bash
./build/cocos-cli result ./private.pem
```

The above will generate a `result.bin` file. Copy this file to the ai directory:

```bash
cp result.bin ../ai/covid19/
```

Test the model with the test data:

```bash
cd ../ai/covid19
```

```bash
python predict.py --model result.bin --image data/test/COVID/COVID-2.png
```
