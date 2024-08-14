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

You can download the dataset `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3).
Create a new directory `datasets`, and add the downloaded `creditcard.csv` to the directory.

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
./build/cocos-cli keys -k="rsa"
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
&{message:"\nXGBoost Evaluation Metrics on Validation Set:\nAccuracy: 0.9995, Precision: 0.7887, Recall: 0.8750, F1-score: 0.8296\n\nXGBoost Evaluation Metrics on Test Set:\nAccuracy: 0.9995, Precision: 0.8252, Recall: 0.8673, F1-score: 0.8458\n"  computation_id:"1"  level:"DEBUG"  timestamp:{seconds:1723543333  nanos:759194638}}
received agent event
&{event_type:"running"  timestamp:{seconds:1723543333  nanos:984903280}  computation_id:"1"  originator:"agent"  status:"complete"}
received agent event
&{event_type:"resultsReady"  timestamp:{seconds:1723543333  nanos:996398342}  computation_id:"1"  originator:"agent"  status:"in-progress"}
received agent log
&{message:"Transition: resultsReady -> resultsReady\n"  computation_id:"1"  level:"DEBUG"  timestamp:{seconds:1723543333  nanos:996384676}}
received agent log
&{message:"Method Result took 992ns to complete without errors"  computation_id:"1"  level:"INFO"  timestamp:{seconds:1723543549  nanos:842501995}}
received agent event
&{event_type:"complete"  timestamp:{seconds:1723543549  nanos:844679773}  computation_id:"1"  originator:"agent"  status:"in-progress"}
received agent log
&{message:"Transition: complete -> complete\n"  computation_id:"1"  level:"DEBUG"  timestamp:{seconds:1723543549  nanos:844659716}}
```

On another terminal, upload the artifacts to the computation server:

```bash
export AGENT_GRPC_URL=localhost:<port_number>
```

```bash
./build/cocos-cli algo ../ai/fraud-detection/fraud-detection.py ./private.pem -a python -r ../ai/fraud-detection/requirements.txt
```

Upload the data to the computation server:

```bash
./build/cocos-cli data ../ai/fraud-detection/datasets/creditcard.csv ./private.pem
```

When the results are ready, download the results:

```bash
./build/cocos-cli result ./private.pem
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

Download the data from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/download?datasetVersionNumber=3).

You can install kaggle-cli to download the dataset:

```bash
pip install kaggle
```

Set the [kaggle API key](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md#api-credentials) and download the dataset:

```bash
unzip archive.zip -d datasets
```

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
./build/cocos-cli keys -k="rsa"
```

You need to have done the following:


  ```bash
  cd cmd/manager
  ```

  Make sure you have the `bzImage` and `rootfs.cpio.gz` in the `cmd/manager/img` directory.

  ```bash
  sudo \                                                                                                                                      (main|…1⚑2)
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
  ./build/cocos-cli keys -k="rsa"
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

- The computation will run and you will get an event that the results are ready. You can download the results by running the following command:

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
