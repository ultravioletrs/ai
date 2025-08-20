# COCOS

Cocos AI is an open-source system designed for running confidential workloads. It features a Confidential VM (CVM) manager, an in-enclave Agent, and a Command Line Interface (CLI) for secure communication with the enclave.

This folder contains some machine learning examples to help you get started with Cocos. Currently cocos supports running algorithms as binary targets in the enclave and also wasm modules. With rust we are able to build the same algorithms as binary targets and wasm modules by changing the target we are building for.

The following documentation helps you to run the examples in the enclave.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Running Examples with CVMS](#running-examples-with-cvms)
- [WASM Module Example](#wasm-module-example)
- [Notes](#notes)

## Prerequisites

- Git
- Make
- Virtualization enabled in the BIOS
- [Qemu](https://www.qemu.org/download/)
- Vsock kernel module
- [Cocos](https://github.com/ultravioletrs/cocos)
- [Rust](https://www.rust-lang.org/tools/install)
- [Go](https://golang.org/doc/install)

## Setup

### Build QEMU Images

1. **Clone the repositories:**

   ```bash
   git clone https://github.com/ultravioletrs/cocos.git
   git clone https://github.com/buildroot/buildroot.git
   ```

2. **Prepare cocos directory:**

   ```bash
   mkdir -p cocos/cmd/manager/img && mkdir -p cocos/cmd/manager/tmp
   ```

3. **Build the cocos qemu image:**

   ```bash
   cd buildroot
   make BR2_EXTERNAL=../cocos/hal/linux cocos_defconfig
   make -j4 && cp output/images/bzImage output/images/rootfs.cpio.gz ../cocos/cmd/manager/img
   ```

   The above commands will build the cocos qemu image and copy the kernel image and rootfs to the manager image directory.

4. **Generate key pair:**

   ```bash
   cd ../cocos
   make all
   ./build/cocos-cli keys -k="rsa"
   ```

### Build Example Algorithm

For the addition example, build the addition algorithm:

```bash
cd burn-algorithms
cargo build --release --bin addition --features cocos
cp target/release/addition ../cocos/
cd ../cocos
```

## Running Examples with CVMS

The modern approach uses the CVMS (Computation Management Server) for streamlined workflow management.

### Finding Your Host IP Address

Find your host machine's IP address (avoid using localhost):

```bash
ip a
```

Look for your network interface (e.g., wlan0 for WiFi, eth0 for Ethernet) and note the inet address. For example, if you see `192.168.1.100`, use that as your `<YOUR_HOST_IP>`.

### Start Core Services

#### Start the Computation Management Server (CVMS)

From your cocos directory, start the CVMS server with the addition algorithm:

```bash
HOST=<YOUR_HOST_IP> go run ./test/cvms/main.go -algo-path ./addition -public-key-path public.pem -attested-tls-bool false
```

Expected output:

```bash
{"time":"...","level":"INFO","msg":"cvms_test_server service gRPC server listening at <YOUR_HOST_IP>:7001 without TLS"}
```

#### Start the Manager

Navigate to the cocos/cmd/manager directory and start the Manager:

```bash
cd cmd/manager
sudo \
MANAGER_QEMU_SMP_MAXCPUS=4 \
MANAGER_QEMU_MEMORY_SIZE=4G \
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

### Create CVM and Upload Algorithm

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

#### Export Agent gRPC URL

Set the AGENT_GRPC_URL using the port noted in the previous step:

```bash
export AGENT_GRPC_URL=localhost:<AGENT_PORT>
```

#### Upload Addition Algorithm

From your cocos directory:

```bash
./build/cocos-cli algo ./addition ./private.pem
```

Expected output:

```bash
🔗 Connected to agent  without TLS
Uploading algorithm file: ./addition
🚀 Uploading algorithm [███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████] [100%]                               
Successfully uploaded algorithm! ✔ 
```

#### Upload Dataset (if required)

For algorithms that require datasets:

```bash
./build/cocos-cli data <dataset-file> ./private.pem
```

**Note:** The addition example doesn't require a dataset, so this step can be skipped.

#### Download Results

After the computation completes:

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

#### Extract and View Results

```bash
unzip results.zip
cat results/result.txt
```

For the addition example, you can also read the result using the built binary:

```bash
cd burn-algorithms
cargo build --release --bin addition --features read
./target/release/addition ../cocos/results/result.txt
```

Expected output:

```bash
"[5.141593, 4.0, 5.0, 8.141593]"
```

#### Remove CVM

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

## WASM Module Example

### Build WASM Module

For the addition inference example:

```bash
cd burn-algorithms/addition-inference
cargo build --release --target wasm32-wasip1 --features cocos
cp ../target/wasm32-wasip1/release/addition-inference.wasm ../../../cocos/
cd ../../../cocos
```

### Run with CVMS

Start the CVMS server with the WASM module:

```bash
HOST=<YOUR_HOST_IP> go run ./test/cvms/main.go -algo-path ./addition-inference.wasm -public-key-path public.pem -attested-tls-bool false
```

Follow the same CVM creation and management steps as above, but upload the algorithm with the WASM flag:

```bash
./build/cocos-cli algo ./addition-inference.wasm ./private.pem -a wasm
```

## Notes

- **Memory Requirements**: 4GB is sufficient for most basic examples; increase as needed for complex algorithms
- **WASM Support**: Both binary and WASM modules are supported, with WASM providing better portability
- **Security**: The enclave provides confidential computing capabilities for sensitive workloads
- **Datasets**: Not all algorithms require datasets; the addition example works without external data
- **Results Format**: Results are packaged in ZIP files and can contain multiple output files

### Terminal Recordings

- Binary example: [![asciicast](https://asciinema.org/a/LzH6RLi1r69hYBhx3qaOIn4ER.svg)](https://asciinema.org/a/LzH6RLi1r69hYBhx3qaOIn4ER)
- WASM example: [![asciicast](https://asciinema.org/a/vKnxV4A9HXloD8g1xJRvw7h3T.svg)](https://asciinema.org/a/vKnxV4A9HXloD8g1xJRvw7h3T)
