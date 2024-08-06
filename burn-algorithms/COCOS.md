# COCOS

Cocos AI is an open-source system designed for running confidential workloads. It features a Confidential VM (CVM) manager, an in-enclave Agent, and a Command Line Interface (CLI) for secure communication with the enclave.

This folder contains some machine learning examples to help you get started with Cocos. Currently cocos supports running algorithms as binary targets in the enclave and also wasm modules. With rust we are able to build the same algorithms as binary targets and wasm modules by changing the target we are building for.

The following documentation helps you to run the examples in the enclave.

## Prerequisites

- Git
- Make
- Virtualization enabled in the BIOS
- [Qemu](https://www.qemu.org/download/)
- Vsock kernel module
- [Cocos](https://github.com/ultravioletrs/cocos)
- [Rust](https://www.rust-lang.org/tools/install)
- [Go](https://golang.org/doc/install)

### Build qemu images

Clone the cocos and buildroot repositories.

```bash
git clone https://github.com/ultravioletrs/cocos.git
```

prepare cocos directory.

```bash
mkdir -p cocos/cmd/manager/img && mkdir -p cocos/cmd/manager/tmp
```

```bash
git clone https://github.com/buildroot/buildroot.git
```

Change the directory to buildroot.

```bash
cd buildroot
```

Build the cocos qemu image.

```bash
make BR2_EXTERNAL=../cocos/hal/linux cocos_defconfig
```

```bash
make -j4 && cp output/images/bzImage output/images/rootfs.cpio.gz ../cocos/cmd/manager/img
```

The above commands will build the cocos qemu image and copy the kernel image and rootfs to the manager image directory.

### Get Key Pair

Generates a new public/private key pair using an algorithm of the users choice(This happens in the cocos directory).

If you are not in the cocos directory, change the directory to cocos.

```bash
cd ../cocos
```

```bash
make cli
```

```bash
./build/cocos-cli keys -k="rsa"
```

## Running the examples

Start computation server(this happens in the cocos directory).

```bash
go run ./test/computations/main.go <path_to_algorithm_file> public.pem false <path_to_data_files...>
```

For the addition example we can build the addition algorithm(this happens in the burn-algorithms directory).

```bash
cargo build --release --bin addition --features cocos
```

Copy the built binary from `/target/release/addition` to the directoy where you will run the computation server.

```bash
cp target/release/addition ../../cocos
```

For example, to run the `addition` algorithm, run the following command (this happens in the cocos directory). Since the addition algorithm does not require any dataset, the dataset path is empty.

```bash
go run ./test/computations/main.go ./addition public.pem false
```

Start the manager.

```bash
cd cmd/manager
```

The manager requires the vhost_vsock kernel module to be loaded. Load the module with the following command.

```bash
sudo modprobe vhost_vsock
```

```bash
sudo \
MANAGER_QEMU_SMP_MAXCPUS=4 \
MANAGER_GRPC_URL=localhost:7001 \
MANAGER_LOG_LEVEL=debug \
MANAGER_QEMU_USE_SUDO=false  \
MANAGER_QEMU_ENABLE_SEV=false \
MANAGER_QEMU_SEV_CBITPOS=51 \
MANAGER_QEMU_ENABLE_SEV_SNP=false \
MANAGER_QEMU_OVMF_CODE_FILE=/usr/share/edk2/x64/OVMF_CODE.fd \
MANAGER_QEMU_OVMF_VARS_FILE=/usr/share/edk2/x64/OVMF_VARS.fd \
go run main.go
```

This will start on a specific port called `agent_port`, which will be in the manager logs.

For example,

```bash
{"time":"2024-07-26T11:45:08.503149211+03:00","level":"INFO","msg":"manager_test_server s
ervice gRPC server listening at :7001 without TLS"}
{"time":"2024-07-26T11:45:14.827479501+03:00","level":"DEBUG","msg":"received who am on i
p address [::1]:47936"}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1721983514 nanos:832365721} computation_id
:"1" originator:"manager" status:"starting"}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1721983514 nanos:833034946} computation_id
:"1" originator:"manager" status:"in-progress"}
received agent log
&{message:"char device redirected to /dev/pts/15 (label compat_monitor0)\n" computation_i
d:"1" level:"debug" timestamp:{seconds:1721983514 nanos:849595083}}
received agent log
&{message:"\x1b[2J\x1b[0" computation_id:"1" level:"debug" timestamp:{seconds:1721983515
nanos:215753406}}
received agent event
&{event_type:"vm-provision" timestamp:{seconds:1721983527 nanos:970098872} computation_id
:"1" originator:"manager" status:"complete"}
received runRes
&{agent_port:"43045" computation_id:"1"}
received agent log
&{message:"Transition: receivingManifest -> receivingManifest\n" computation_id:"1" level
:"DEBUG" timestamp:{seconds:1721983527 nanos:966911139}}
received agent log
```

## Uploading the algorithm and dataset

```bash
export AGENT_GRPC_URL=localhost:43045
```

Upload the algorithm to the enclave.

```bash
./build/cocos-cli algo ./addition ./private.pem
```

Upload the dataset to the enclave. Since this algorithm does not require a dataset, we can skip this step.

```bash
./build/cocos-cli dataset <dataset-file> ./private.pem
```

## Downloading the results

After the computation has been completed, you can download the results from the enclave.

```bash
./build/cocos-cli result ./private.pem
```

This will generate a `result.zip` file in the current directory. Unzip the file to get the result.

```bash
unzip result.zip
```

For the addition example, we can read the output from the `result.txt` file which contains the result of the addition algorithm. This file is gotten from the `result.zip` file.

To read the result, run the following command.

```bash
cat result.txt
```

You can also build the addition algorithm with the `read` feature to read the result from the enclave.

```bash
cargo build --release --bin addition --features read
```

Run the binary with the `result.txt` file as an argument.

```bash
./target/release/addition ../../cocos/results.txt
```

This will output the result of the addition algorithm.

```bash
"[5.141593, 4.0, 5.0, 8.141593]"
```

Terminal recording of the above steps:

[![asciicast](https://asciinema.org/a/LzH6RLi1r69hYBhx3qaOIn4ER.svg)](https://asciinema.org/a/LzH6RLi1r69hYBhx3qaOIn4ER)

## Wasm Module

For the addition inference example we can build the addition algorithm(this happens in the burn-algorithms/addition-inference directory).

```bash
cargo build --release --target wasm32-wasip1 --features cocos
```

Copy the built wasm module from `/target/wasm32-wasip1/release/addition_inference.wasm` to the directoy where you will run the computation server.

```bash
cp ../target/wasm32-wasip1/release/addition-inference.wasm ../../../cocos
```

For example, to run the `addition-inference` algorithm, run the following command (this happens in the cocos directory). Since the addition-inference algorithm does not require any dataset, the dataset path is empty.

```bash
go run ./test/computations/main.go ./addition-inference.wasm public.pem true
```

Start the manager.

```bash
cd cmd/manager
```

```bash
sudo \
MANAGER_QEMU_SMP_MAXCPUS=4 \
MANAGER_GRPC_URL=localhost:7001 \
MANAGER_LOG_LEVEL=debug \
MANAGER_QEMU_USE_SUDO=false  \
MANAGER_QEMU_ENABLE_SEV=false \
MANAGER_QEMU_SEV_CBITPOS=51 \
MANAGER_QEMU_ENABLE_SEV_SNP=false \
MANAGER_QEMU_OVMF_CODE_FILE=/usr/share/edk2/x64/OVMF_CODE.fd \
MANAGER_QEMU_OVMF_VARS_FILE=/usr/share/edk2/x64/OVMF_VARS.fd \
go run main.go
```

This will start on a specific port called `agent_port`, which will be in the manager logs.

For example,

```bash
{"time":"2024-08-06T10:54:53.42640029+03:00","level":"INFO","msg":"manager_test_server service gRPC server listening at :7001 without TLS"}
{"time":"2024-08-06T10:54:55.953576985+03:00","level":"DEBUG","msg":"received who am on ip address [::1]:50528"}
received agent event
&{event_type:"vm-provision"  timestamp:{seconds:1722930895  nanos:957553381}  computation_id:"1"  originator:"manager"  status:"starting"}
received agent event
&{event_type:"vm-provision"  timestamp:{seconds:1722930895  nanos:958021704}  computation_id:"1"  originator:"manager"  status:"in-progress"}
received agent log
&{message:"char device redirected to /dev/pts/10 (label compat_monitor0)\n"  computation_id:"1"  level:"debug"  timestamp:{seconds:1722930896  nanos:39152844}}
received agent log
&{message:"\x1b["  computation_id:"1"  level:"debug"  timestamp:{seconds:1722930898  nanos:319429985}}
received agent event
&{event_type:"vm-provision"  timestamp:{seconds:1722930911  nanos:886580521}  computation_id:"1"  originator:"manager"  status:"complete"}
received runRes
&{agent_port:"46593"  computation_id:"1"}
received agent log
&{message:"Transition: receivingManifest -> receivingManifest\n"  computation_id:"1"  level:"DEBUG"  timestamp:{seconds:1722930911  nanos:859764366}}
received agent event
```

```bash
export AGENT_GRPC_URL=localhost:46593
```

Upload the algorithm to the enclave. The `-a wasm` flag is used to specify that the algorithm is a wasm module.

```bash
./build/cocos-cli algo ./addition-inference.wasm ./private.pem -a wasm
```

Upload the dataset to the enclave. Since this algorithm does not require a dataset, we can skip this step.

```bash
./build/cocos-cli dataset <dataset-file> ./private.pem
```

After the computation has been completed, you can download the results from the enclave.

```bash
./build/cocos-cli result ./private.pem
```

This will generate a `result.zip` file in the current directory. Unzip the file to get the result.

```bash
unzip result.zip
```

For the addition example, we can read the output from the `results/results.txt` file which contains the result of the addition algorithm. This file is gotten from the `result.zip` file.

To read the result, run the following command.

```bash
cat results/results.txt
```

Terminal recording of the above steps:

[![asciicast](https://asciinema.org/a/vKnxV4A9HXloD8g1xJRvw7h3T.svg)](https://asciinema.org/a/vKnxV4A9HXloD8g1xJRvw7h3T)

## Conclusion

This documentation has shown how to run the addition algorithm in the enclave using the Cocos system. The addition algorithm was built as a binary target and a wasm module. The results were downloaded from the enclave and read to get the output of the addition algorithm. This process can be followed to run other algorithms in the enclave using the Cocos system.
