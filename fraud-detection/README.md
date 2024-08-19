# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques, specifically the XGBoost model. The dataset used in this project was downloaded from Kaggle.
Given the class imbalance ratio, script measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC).

## Dataset
The dataset used in this project contains transactions made by European cardholders over a period of two days in September 2013. It has been modified using Principal Component Analysis (PCA) for confidentiality reasons. 
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
The dataset includes the following features:

- `Time`: Number of seconds elapsed between this transaction and the first transaction in the dataset.
- `V1, V2, ..., V28`: Principal components obtained from PCA.
- `Amount`: Transaction amount.
- `Class`: Label indicating whether the transaction is fraudulent (1) or not (0).


Fetch the data from Kaggle - [Fraud Detection Database](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
```

To run the above command you would need [kaggle cli](https://github.com/Kaggle/kaggle-api) installed and API credentials setup. This can be done by following [this documentation](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#kaggle-api).
You will get `creditcardfraud.zip` in the folder

Run:
```bash
unzip creditcardfraud.zip -d datasets/
```
This will extract the contents of `creditcardfraud.zip` into `datasets`  directory


### Train Model

To train model run script execute:
```bash
python fraud-detection.py
``` 
Script will produce model `fraud_model.ubj`
### Test Model
Inference can be done using `prediction.py`. Make sure to move the `fraud_model.ubj` file to the datasets directory as well.

```bash
python prediction.py
```
Results will be `.png` images that represents confusion matrix and AUPRC
![](images/AUPRC.png)
![](images/C-matrix.png)

## Testing with Cocos

Make sure you have the Cocos repository cloned and eos buildroot installed. This can be done by following the instructions in the [Cocos Documentation](https://docs.cocos.ultraviolet.rs/getting-started/)

Clone the ai repository which has the fraud detection algorithm:

```bash
git clone https://github.com/ultravioletrs/ai.git
```

```bash
cd ai/fraud-detection
```

Download the data from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3).

Extract the `creditcard.csv` file into the datasets directory

```bash
unzip archive.zip -d datasets
```

Change the directory to cocos:

```bash
cd ../../cocos
```

Build cocos artifacts:

```bash
make all
```

Before running the computation server, we need to issue public and private key pairs. This can be done by running the following commands:

```bash
./build/cocos-cli keys -k rsa
```

Run the computation server:

```bash
go run ./test/computations/main.go ../ai/fraud-detection/ fraud-detection.py  public.pem false ../ai/fraud-detection/datasets/creditcard.csv
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
MANAGER_GRPC_URL=localhost:7001 \
MANAGER_LOG_LEVEL=debug \
MANAGER_QEMU_ENABLE_SEV_SNP=false \
MANAGER_QEMU_OVMF_CODE_FILE=/usr/share/edk2/x64/OVMF_CODE.fd \
MANAGER_QEMU_OVMF_VARS_FILE=/usr/share/edk2/x64/OVMF_VARS.fd \
go run main.go
```

After sometime you will see the computation server will output a port number. This port number is the port on which the computation server is running. You can use this port number to run the client.

The logs will look like this:

```bash
{"time":"2024-08-19T10:23:50.068445068+03:00","level":"INFO","msg":"manager_test_server service gRPC server listening at :7001 without TLS"}
{"time":"2024-08-19T10:24:17.767534539+03:00","level":"DEBUG","msg":"received who am on ip address [::1]:45608"}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1724052258 nanos:76069455} computation_id:"1" originator:"manager" status:"starting"}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1724052258 nanos:76390596} computation_id:"1" originator:"manager" status:"in-progress"}
received agent log
&{message:"char device redirected to /dev/pts/5 (label compat_monitor0)\n" computation_id:"1" level:"debug" timestamp:{seconds:1724052258 nanos:140448274}}
received agent log
&{message:"qemu-system-x86_64: warning: host doesn't support requested feature: CPUID.80000001H:EDX.mmxext [bit 22]\nqemu-system-x86_64: warning: host doesn't support requested feature: CPUID.80000001H:EDX.fxsr-opt [bit 25]\nqemu-system-x86_64: warning: host doesn't support requested feature: CPUID.80000001H:ECX.cr8legacy [bit 4]\nqemu-system-x86_64: warning: host doesn't support requested feature: CPUID.80000001H:ECX.sse4a [bit 6]\nqemu-system-x86_64: warning: host doesn't support requested feature: CPUID.80000001H:ECX.misalignsse [bit 7]\nqemu-system-x86_64: warning: host doesn't support requested feature: CPUID.80000001H:ECX.osvw [bit 9]\n" computation_id:"1" level:"error" timestamp:{seconds:1724052258 nanos:177482826}}
received agent log
&{message:"\x1b[2J\x1b[01;01H" computation_id:"1" level:"debug" timestamp:{seconds:1724052258 nanos:485734327}}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1724052271 nanos:480190393} computation_id:"1" originator:"manager" status:"complete"}
received runRes
&{agent_port:"6050" computation_id:"1"}
received agent log
&{message:"Transition: receivingManifest -> receivingManifest\n" computation_id:"1" level:"DEBUG" timestamp:{seconds:1724052271 nanos:479293635}}
received agent event
&{event_type:"receivingAlgorithm" timestamp:{seconds:1724052271 nanos:480207098} computation_id:"1" originator:"agent" status:"in-progress"}
received agent log
&{message:"agent service gRPC server listening at :7002 without TLS" computation_id:"1" level:"INFO" timestamp:{seconds:1724052271 nanos:480676615}}
received agent event
&{event_type:"receivingData" timestamp:{seconds:1724052647 nanos:92491532} computation_id:"1" originator:"agent" status:"in-progress"}
received agent log
&{message:"Transition: receivingData -> receivingData\n" computation_id:"1" level:"DEBUG" timestamp:{seconds:1724052647 nanos:92466438}}
received agent event
&{event_type:"running" timestamp:{seconds:1724052722 nanos:889675666} computation_id:"1" originator:"agent" status:"in-progress"}
received agent log
&{message:"computation run started" computation_id:"1" level:"DEBUG" timestamp:{seconds:1724052722 nanos:889653708}}
received agent log
&{message:"Collecting pandas~=2.2.2 (from -r /tmp/requirements.txt3799616143 (line 1))\n" computation_id:"1" level:"DEBUG" timestamp:{seconds:1724052725 nanos:908283024}}
received agent log
&{message:"  Downloading pandas-2.2.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (19 kB)\n" computation_id:"1" level:"DEBUG" timestamp:{seconds:1724052726 nanos:447295301}}
received agent log

```

On another terminal, upload the artifacts to the computation server:

```bash
export AGENT_GRPC_URL=localhost:<port_number>
```

```bash
./build/cocos-cli algo ../ai/fraud-detection/fraud-detection.py ./private.pem -a python -r ../ai/fraud-detection/requirements.txt

```

Output:
```
2024/08/19 10:30:27 Uploading algorithm binary: ../../ai/fraud-detection/fraud-detection.py
Uploading algorithm...  100% [===================================================================================>] 
2024/08/19 10:30:47 Successfully uploaded algorithm
```


Upload the data to the computation server:

```bash
./build/cocos-cli data ../ai/fraud-detection/datasets/creditcard.csv ./private.pem
```

Output:
```
2024/08/19 10:31:41 Uploading dataset CSV: ../../ai/fraud-detection/datasets/creditcard.csv
Uploading data...  100% [========================================================================================>] 
2024/08/19 10:32:02 Successfully uploaded dataset
```

When the results are ready, download the results:

```bash
./build/cocos-cli result ./private.pem
```
Output:
```
2024/08/19 10:55:01 Retrieving computation result file
2024/08/19 10:55:21 Computation result retrieved and saved successfully!
```

The above will generate a `results.zip` file. Copy this file to the ai directory:

```bash
cp results.zip ../ai/fraud-detection/
```

Test the model with the test data:

```bash
cd ../ai/fraud-detection
```

```bash
unzip results.zip -d results
```

The image can be any image from the test dataset:

```bash
python prediction.py
```

## Testing with Prism

Make sure you have the Cocos repository cloned and eos buildroot installed. This can be done by following the instructions in the [Cocos Documentation](https://docs.cocos.ultraviolet.rs/getting-started/)

Clone the ai repository which has the fraud detection algorithm:

```bash
git clone https://github.com/ultravioletrs/ai.git
```

```bash
cd ai/fraud-detection
```
For detailed instructions on how to fetch datasets from Kaggle, please refer to the [Datasets](#dataset)  section.

In your browser launch to PRISM SaaS at https://prism.ultraviolet.rs.

- Create a user at https://prism.ultraviolet.rs.
- Create a workspace
- Login to the created workspace
- Create a backend.
- Issue Certs for the backend, request download and download the certs
- Unzip the folder and copy the contents to the managers `cmd/manager/` directory under `cocos` folder
- Start the manager with the backend address.

The following recording will demonstrate how to setup prism - https://jam.dev/c/8067f697-4eaa-407f-875a-17119e4f3901

Build cocos artifacts:

```bash
make all
```

Before running the computation server, we need to issue certificates for the computation server and the client. This is done from the PRISM SaaS.
Public/Private key pairs are needed for the users that will provide the algorithm, dataset and consume the results.

This can be done by running the following commands:

```bash
./build/cocos-cli keys -k rsa
```

You need to have done the following:


  ```bash
  cd cmd/manager
  ```

  Make sure you have the `bzImage` and `rootfs.cpio.gz` in the `cmd/manager/img` directory.

  ```bash
  sudo \
  MANAGER_QEMU_SMP_MAXCPUS=4 \
  MANAGER_QEMU_MEMORY_SIZE=25G \
  MANAGER_GRPC_URL=localhost:7011 \
  MANAGER_LOG_LEVEL=debug \
  MANAGER_QEMU_ENABLE_SEV_SNP=false \
  MANAGER_QEMU_OVMF_CODE_FILE=/usr/share/edk2/x64/OVMF_CODE.fd \
  MANAGER_QEMU_OVMF_VARS_FILE=/usr/share/edk2/x64/OVMF_VARS.fd \
  MANAGER_GRPC_CLIENT_CERT=cert.pem \
  MANAGER_GRPC_CLIENT_KEY=key.pem \
  MANAGER_GRPC_SERVER_CA_CERTS=ca.pem \
  go run main.go
  ```

- Create the fraud detection computation. To get the filehash for all the files go to `cocos` folder and use the cocos-cli. For the file names use `creditcard.csv` and `fraud-detection.py`

  ```bash
  ./build/cocos-cli file-hash ../ai/fraud-detection/datasets/creditcard
  ```

  ```bash
  ./build/cocos-cli file-hash ../ai/fraud-detection/fraud-detection.py
  ```

- After the computation has been created, each user needs to upload their public key generated by `cocos-cli`. This key will enable the respective user to upload the datatsets and algorithms and also download the results.

  ```bash
  ./build/cocos-cli keys -k rsa
  ```

- Click run computation and wait for the vm to be provisioned.Copy the aggent port number and export `AGENT_GRPC_URL`

  ```bash
  export AGENT_GRPC_URL=localhost:<port_number>
  ```

- After vm has been provisioned upload the datasets and the algorithm

  ```bash
  ./build/cocos-cli algo ../ai/fraud-detection/fraud-detection.py ./private.pem -a python -r ../ai/fraud-detection/requirements.txt
  ```

  ```bash
  ./build/cocos-cli data ../ai/fraud-detection/creditcard.csv ./private.pem
  ```

- The computation will run, and you will get an event that the results are ready. You can download the results by running the following command:

  ```bash
  ./build/cocos-cli results ./private.pem
  ```

The above will generate a `results.zip` file. Copy this file to the ai directory:

```bash
cp results.bin ../ai/fraud-detection/
```

Test the model with the test data:

```bash
cd ../ai/fraud-detection
```

```bash
unzip results.zip -d results
```

The image can be any image from the test dataset:

```bash
python prediction.py
```
