# COCOS

Cocos AI is an open-source system designed for running confidential workloads. It features a Confidential VM (CVM) manager, an in-enclave Agent, and a Command Line Interface (CLI) for secure communication with the enclave.

This folder contains some machine learning examples to help you get started with Cocos. Currently cocos supports running algorithms as binary targets in the enclave and also wasm modules. With rust we are able to build the same algorithms as binary targets and wasm modules by changing the target we are building for.

The following documentation helps you to run the examples in the enclave.

## Prerequisites

- [Cocos](https://github.com/ultravioletrs/cocos)
- [Rust](https://www.rust-lang.org/tools/install)

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

Generates a new public/private key pair using an algorithm of the users choice(This happens in the cocos directory).

```bash
make cli
```

```bash
./build/cocos-cli keys -k="rsa"
```

Start computation server(this happens in the cocos directory).

```bash
go run ./test/computations/main.go <algo_file> public.pem false <datafiles>
```

For the addition example we can build the addition algorithm(this happens in the burn-algorithms directory).

```bash
cargo build --release --bin addition --features cocos
```

Copy the built binary from `/target/release/addition` to the directoy where you will run the computation server.

For example, to run the `addition` algorithm, run the following command.

```bash
go run ./test/computations/main.go ./addition public.pem false ""
```

The dataset path is empty because the addition algorithm does not require any dataset.

Start the manager.

```bash
cd cmd/manager
```

```bash
sudo MANAGER_QEMU_SMP_MAXCPUS=4 MANAGER_GRPC_URL=localhost:7001 MANAGER_LOG_LEVEL=debug MANAGER_QEMU_USE_SUDO=false  MANAGER_QEMU_ENABLE_SEV=false MANAGER_QEMU_SEV_CBITPOS=51 MANAGER_QEMU_ENABLE_SEV_SNP=false MANAGER_QEMU_OVMF_CODE_FILE=/usr/share/edk2/x64/ OVMF_CODE.fd MANAGER_QEMU_OVMF_VARS_FILE=/usr/share/edk2/x64/OVMF_VARS.fd go run main.go
```

The will start on a specific port called `agent_port`, which will be in the manager logs.

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

After the computation has been completed, you can download the results from the enclave.

```bash
./build/cocos-cli result ./private.pem
```

This will generate a `result.bin` file in the current directory.

For the addition example, we can read the output from the `result.bin` file.

We build with `read` feature to read the result from the enclave.

```bash
cargo build --release --bin addition --features read
```

Run the binary with the `result.bin` file as an argument.

```bash
./target/release/addition ./result.bin
```

This will output the result of the addition algorithm.

```bash
"[5.141593, 4.0, 5.0, 8.141593]"
```
