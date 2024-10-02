# Burn Algorithms

The need to run machine learning algorithms either training or making inference from a single algorithm has been a challenge with Python based models This is where Rust comes in. Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. Unlike C++ or C, Rust is memory safe by default.

Burn Algorithms is a collection machine learning algorithms that is written in Rust using [burn](https://burn.dev/). Burn is a deep learning Framework for Rust that is designed to be extremely flexible, compute efficient and highly portable. Burn strives to be as fast as possible on as many hardwares as possible, with robust implementations. With Burn, you can run your machine learning models on the CPU, GPU, WebAssembly, and other hardwares.

These examples are used to be provide starting point on using [cocos](https://github.com/ultravioletrs/cocos) system to run machine learning algorithms. [Cocos AI](https://docs.cocos.ultraviolet.rs/) is an open-source system designed for running confidential workloads. It features a Confidential VM (CVM) manager, an in-enclave Agent, and a Command Line Interface (CLI) for secure communication with the enclave.

Currently cocos supports running algorithms as binary targets in the enclave and also wasm modules. With rust we are able to build the same algorithms as binary targets and wasm modules by changing the target we are building for.

## Examples

The following examples are available:

- [x] Matrix Addition
- [x] Classification problem - (Iris dataset)
- [x] Regression problem - (Wine Quality dataset)
- [x] Image Classification - (MNIST dataset) - (WIP)
- [x] Image Classification - (CIFAR-10 dataset)
- [x] Text Classification - (AG News dataset)
- [x] Text Classification - (IMDB dataset)

## Training

### Addition

This is a simple matrix addition example. It adds two matrices and prints the result.

To run the example, use the following command (you should be in the `burn-algorithms` directory):

```bash
cargo run --release --bin addition-wgpu --features wgpu
```

The output should be something like:

```bash
[5.141593, 4.0, 5.0, 8.141593]
```

From this example, we have used the `wgpu` feature, which means we are targeting the WebGPU backend. If you want to run the example on the CPU, you can use the following command:

You can also run the example using the `ndarray` feature:

```bash
cargo run --release --bin addition-ndarray --features ndarray
```

The output should be something like:

```bash
[5.141593, 4.0, 5.0, 8.141593]
```

A detailed explanation of how to run the example on cocos can be found in the [COCOS.md](./COCOS.md) file.

```bash
cargo build --release --bin addition-cocos --features cocos
```

This will generate a binary target that can be run on cocos.

You can run it on your local machine using the following command:

```bash
./target/release/addition-cocos
```

This generates a file called `result.txt` that contains the result of the addition and stores it in the `results` folder.

```bash
cat results/results.txt
```

To read results from cocos, you can use the following command:

```bash
cargo build --release --bin addition-read --features read
```

```bash
./target/release/addition-read ./results/results.txt
```

### Iris dataset

The dataset is already downloaded and stored in the `datasets` folder inside the `iris` folder.

Run the following command to train the model on `wgpu` or `cpu`. This should be run on the `burn-algorithms/iris` folder.

```bash
cargo run --release --bin iris-wgpu --features wgpu
```

The output should be something like:

```bash
Train Dataset Size: 120
Test Dataset Size: 30
======================== Learner Summary ========================
Model:
ClassificationModel {
  input_layer: Linear {d_input: 4, d_output: 128, bias: true, params: 640}
  hidden_layer: Linear {d_input: 128, d_output: 64, bias: true, params: 8256}
  activation: Relu
  output_layer: Linear {d_input: 64, d_output: 3, bias: true, params: 195}
  params: 9091
}
Total Epochs: 48


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Loss     | 0.066    | 47       | 1.116    | 1        |
| Train | Accuracy | 40.833   | 1        | 99.167   | 27       |
| Valid | Loss     | 0.041    | 43       | 0.905    | 1        |
| Valid | Accuracy | 66.667   | 1        | 100.000  | 43       |
```

or

```bash
cargo run --release --bin iris-ndarray --features ndarray
```

The output should be something like:

```bash
Train Dataset Size: 120
Test Dataset Size: 30
======================== Learner Summary ========================
Model:
ClassificationModel {
  input_layer: Linear {d_input: 4, d_output: 128, bias: true, params: 640}
  hidden_layer: Linear {d_input: 128, d_output: 64, bias: true, params: 8256}
  activation: Relu
  output_layer: Linear {d_input: 64, d_output: 3, bias: true, params: 195}
  params: 9091
}
Total Epochs: 48


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Loss     | 0.062    | 48       | 1.041    | 1        |
| Train | Accuracy | 47.500   | 1        | 98.333   | 47       |
| Valid | Loss     | 0.034    | 48       | 0.837    | 1        |
| Valid | Accuracy | 66.667   | 1        | 100.000  | 48       |
```

### Wine Quality dataset

First, download the dataset from <https://archive.ics.uci.edu/dataset/186/wine+quality> and extract it in the `data` folder inside the `winequality` folder.

Download the dataset

```bash
wget https://archive.ics.uci.edu/static/public/186/wine+quality.zip -P winequality/data
```

Extract the dataset

```bash
unzip winequality/data/wine+quality.zip -d winequality/data && rm winequality/data/wine+quality.zip
```

Then, run the following command to train the model on wgpu or ndrray. This should be run on the `burn-algorithms` folder.

```bash
cargo run --release --bin winequality-wgpu --features wgpu
```

The output should be something like:

```bash
Train Dataset Size: 3918
Test Dataset Size: 980
======================== Learner Summary ========================
Model:
RegressionModel {
  input_layer: Linear {d_input: 11, d_output: 1, bias: true, params: 12}
  params: 12
}
Total Epochs: 81


| Split | Metric | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------|----------|----------|----------|----------|
| Train | Loss   | 0.024    | 75       | 0.095    | 1        |
| Valid | Loss   | 0.023    | 76       | 0.063    | 1        |
```

or

```bash
cargo run --release --bin winequality-ndarray --features ndarray
```

The output should be something like:

```bash
Train Dataset Size: 3918
Test Dataset Size: 980
======================== Learner Summary ========================
Model:
RegressionModel {
  input_layer: Linear {d_input: 11, d_output: 1, bias: true, params: 12}
  params: 12
}
Total Epochs: 81


| Split | Metric | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------|----------|----------|----------|----------|
| Train | Loss   | 0.025    | 77       | 0.148    | 1        |
| Valid | Loss   | 0.024    | 76       | 0.090    | 1        |
```

To run the example for cocos, you can use the following command:

```bash
cargo build --release --bin winequality-cocos --features cocos
```

This will generate a binary target that can be run on cocos.

To prepare the cocos dataset, you can use the following command:

```bash
mkdir -p datasets && cd winequality && zip -r dataset.zip data && cd ../ && mv winequality/dataset.zip datasets/winequality
```

You can run it on your local machine using the following command:

```bash
./target/release/winequality-cocos
```

This generates a model at `results/model.bin`.

### MNIST dataset

First, download the dataset from <https://www.kaggle.com/datasets/playlist/mnistzip/data?select=mnist_png> and extract it in the `data` folder inside the `mnist` folder.

```bash
mkdir -p mnist/data/
unzip mnistzip.zip -d mnist/data/
```

The folder structure would be something like:

```bash
mnist
├── Cargo.toml
├── data
│   └── mnist_png
│       ├── train
│       │   ├── 0
│       │   ├── 1
│       │   ├── 2
│       │   ├── 3
│       │   ├── 4
│       │   ├── 5
│       │   ├── 6
│       │   ├── 7
│       │   ├── 8
│       │   └── 9
│       └── valid
│           ├── 0
│           ├── 1
│           ├── 2
│           ├── 3
│           ├── 4
│           ├── 5
│           ├── 6
│           ├── 7
│           ├── 8
│           └── 9
```

Then, run the following command to train the model on wgpu or ndrray.

```bash
cargo run --release --bin mnist-wgpu --features wgpu
```

The output should be something like:

```bash
======================== Learner Summary ========================
Model:
Model {
  conv1: ConvBlock {
    conv: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 80}
    norm: BatchNorm {num_features: 8, momentum: 0.1, epsilon: 0.00001, params: 32}
    activation: Gelu
    params: 112
  }
  conv2: ConvBlock {
    conv: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 1168}
    norm: BatchNorm {num_features: 16, momentum: 0.1, epsilon: 0.00001, params: 64}
    activation: Gelu
    params: 1232
  }
  conv3: ConvBlock {
    conv: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 3480}
    norm: BatchNorm {num_features: 24, momentum: 0.1, epsilon: 0.00001, params: 96}
    activation: Gelu
    params: 3576
  }
  dropout: Dropout {prob: 0.5}
  fc1: Linear {d_input: 11616, d_output: 32, bias: false, params: 371712}
  fc2: Linear {d_input: 32, d_output: 10, bias: false, params: 320}
  activation: Gelu
  params: 376952
}
Total Epochs: 10


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Loss     | 0.015    | 10       | 0.252    | 1        |
| Train | Accuracy | 93.775   | 1        | 99.548   | 10       |
| Valid | Loss     | 0.033    | 7        | 0.079    | 1        |
| Valid | Accuracy | 97.740   | 1        | 98.990   | 10       |
```

or

```bash
cargo run --release --bin mnist-ndarray --features ndarray
```

The output should be something like:

```bash
======================== Learner Summary ========================
Model:
Model {
  conv1: ConvBlock {
    conv: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 80}
    norm: BatchNorm {num_features: 8, momentum: 0.1, epsilon: 0.00001, params: 32}
    activation: Gelu
    params: 112
  }
  conv2: ConvBlock {
    conv: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 1168}
    norm: BatchNorm {num_features: 16, momentum: 0.1, epsilon: 0.00001, params: 64}
    activation: Gelu
    params: 1232
  }
  conv3: ConvBlock {
    conv: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Valid, params: 3480}
    norm: BatchNorm {num_features: 24, momentum: 0.1, epsilon: 0.00001, params: 96}
    activation: Gelu
    params: 3576
  }
  dropout: Dropout {prob: 0.5}
  fc1: Linear {d_input: 11616, d_output: 32, bias: false, params: 371712}
  fc2: Linear {d_input: 32, d_output: 10, bias: false, params: 320}
  activation: Gelu
  params: 376952
}
Total Epochs: 2


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 93.425   | 1        | 97.884   | 2        |
| Train | Loss     | 0.075    | 2        | 0.249    | 1        |
| Valid | Accuracy | 97.680   | 1        | 97.680   | 1        |
| Valid | Loss     | 0.080    | 1        | 0.080    | 1        |
```

### Cifar-10 dataset

First, download the dataset from <https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz> and extract it in the `data` folder inside the `cifar10` folder.

Download the dataset

```bash
wget https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz -P cifar10/data
```

Extract the dataset

```bash
tar -xvzf cifar10/data/cifar10.tgz -C cifar10/data
```

Then, run the following command to train the model on wgpu or ndrray.

```bash
cargo run --release --bin cifar10-wgpu --features wgpu
```

The output should be something like:

```bash
======================== Learner Summary ========================
Model:
Cnn {
  activation: Relu
  dropout: Dropout {prob: 0.3}
  pool: MaxPool2d {kernel_size: [2, 2], stride: [2, 2], padding: Valid, dilation: [1, 1]}
  conv1: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Same, params: 896}
  conv2: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Same, params: 9248}
  conv3: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Same, params: 18496}
  conv4: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Same, params: 36928}
  conv5: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Same, params: 73856}
  conv6: Conv2d {stride: [1, 1], kernel_size: [3, 3], dilation: [1, 1], groups: 1, padding: Same, params: 147584}
  fc1: Linear {d_input: 2048, d_output: 128, bias: true, params: 262272}
  fc2: Linear {d_input: 128, d_output: 10, bias: true, params: 1290}
  params: 550570
}
Total Epochs: 2

| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 16.800   | 1        | 28.009   | 2        |
| Train | Loss     | 1.898    | 2        | 2.190    | 1        |
| Valid | Accuracy | 27.540   | 1        | 27.540   | 1        |
| Valid | Loss     | 1.945    | 1        | 1.945    | 1        |
```

or

```bash
cargo run --release --bin cifar10-ndarray --features ndarray
```

The output should be something like:

```bash

```

### AG News

First, download the dataset from <https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz> and extract it in the `data` folder inside the `imdb` folder.

Download the dataset

```bash
wget https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz -P agnews/data
```

Extract the dataset

```bash
tar -xvzf agnews/data/ag_news_csv.tgz -C agnews/data
```

Add header to the csv files // label,title,description

```bash
sed -i '1s/^/label,title,description\n/' agnews/data/ag_news_csv/train.csv
sed -i '1s/^/label,title,description\n/' agnews/data/ag_news_csv/test.csv
```

Then, run the following command to train the model on wgpu or ndrray.

```bash
cargo run --release --bin agnews-wgpu --features wgpu
```

The output should be something like:

```bash

```

or

```bash
cargo run --release --bin agnews-ndarray --features ndarray
```

The output should be something like:

```bash

```

### IMDB

First, download the dataset from <https://huggingface.co/datasets/scikit-learn/imdb> and extract it in the `data` folder inside the `imdb` folder.

Download the dataset

**Make sure you have git-lfs installed (<https://git-lfs.com>)**

```bash
git lfs install
```

```bash
git clone https://huggingface.co/datasets/scikit-learn/imdb imdb/data
```

Then, run the following command to train the model on wgpu or ndrray.

```bash
cargo run --release --bin imdb-wgpu --features wgpu
```

The output should be something like:

```bash

```

or

```bash
cargo run --release --bin imdb-ndarray --features ndarray
```

The output should be something like:

```bash

```

## Inference

For inference, we use the [wasmtime](https://wasmtime.dev/) runtime to run the WebAssembly binary generated from the Rust code. This is because it runs webassembly code outside the browser and can be used as a command-line utility.

We need it to run outside the browser because we are running machine learning models that require a lot of computation and memory. The browser might not be able to handle the computation and memory requirements of the models. Above that, we are building this to be used inside a Trusted Execution Environment (TEE) like Intel SGX, which is a hardware-based security technology that can help protect your data and code while it is being processed.

For the following examples to work, you need to have the `wasmtime` runtime installed on your machine. You can download it from <https://wasmtime.dev/>. Also, make sure you have the `wasm32-wasip1` target installed on your machine.

```bash
rustup target add wasm32-wasip1
```

To install `wasmtime` on a unix machine, you can use the following command:

```bash
curl https://wasmtime.dev/install.sh -sSf | bash
```

### Addition Inference

This is a simple matrix addition example that runs on WebAssembly. It adds two matrices and prints the result.

To run the example, use the following command:

From `burn-algorithms` root directory run:

```bash
cd addition-inference
```

```bash
cargo build --release --target wasm32-wasip1 --features input
```

```bash
wasmtime ../target/wasm32-wasip1/release/addition-inference.wasm '[[2.0, 3.0], [4.0, 5.0]]' '[[1.0, 1.0], [1.0, 1.0]]'
```

Matrix A: `[[2.0, 3.0], [4.0, 5.0]]`
Matrix B: `[[1.0, 1.0], [1.0, 1.0]]`

The output should be something like:

```bash
Data { value: [3.0, 4.0, 5.0, 6.0], shape: Shape { dims: [2, 2] } }
```

If you want to run inference as compile binary, you can use the following command:

```bash
cargo build --release --features input
```

```bash
../target/release/addition-inference '[[2.0, 3.0], [4.0, 5.0]]' '[[1.0, 1.0], [1.0, 1.0]]'
```

If you don't want to provide the input data as an argument, you can use the following command:

```bash
cargo build --release
```

```bash
../target/release/addition-inference
```

This also applies to wasm32-wasip1 target.

```bash
cargo build --release --target wasm32-wasip1
```

```bash
wasmtime ../target/wasm32-wasip1/release/addition-inference.wasm
```

### Iris dataset classification Inference

This is a simple classification example that runs on WebAssembly. It classifies the Iris dataset and prints the result.

To run the example, use the following command:

From `burn-algorithms` root directory run:

```bash
cd iris-inference
```

```bash
cargo build --release --target wasm32-wasip1 --bin iris-inference
```

```bash
wasmtime ../target/wasm32-wasip1/release/iris-inference.wasm '{"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}'
```

The first and only argument is the input data in JSON format for example:

```bash
{"sepal_length": 7.0,"sepal_width": 3.2,"petal_length": 4.7,"petal_width": 1.4}
```

The output should be something like:

```bash
Iris-versicolor
```

If you want to run inference as compile binary, you can use the following command:

```bash
cargo build --release
```

```bash
../target/release/iris-inference '{"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4}'
```

### MNIST Image classification Inference

This is a simple image classification example that runs on WebAssembly. It classifies the MNIST dataset and prints the result.

To run the example, use the following command:

From `burn-algorithms` root directory run:

```bash
cargo r --release --bin convert-image mnist-inference/4.png
```

This will convert the image to any array that can be used as input to the model.

The output should be something like:

```bash
  Compiling convert-image v0.1.0 (/home/rodneyosodo/code/ultraviolet/ai/burn-algorithms/convert-image)
  Finished `release` profile [optimized] target(s) in 1.93s
  Running `target/release/convert-image mnist-inference/0.png`

[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 67.0, 232.0, 39.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 81.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0, 180.0, 39.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 153.0, 210.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 220.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.0, 254.0, 162.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 222.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 183.0, 254.0, 125.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 245.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 198.0, 254.0, 56.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0, 254.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 23.0, 231.0, 254.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 159.0, 254.0, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 163.0, 254.0, 216.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 159.0, 254.0, 67.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 86.0, 178.0, 248.0, 254.0, 91.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 159.0, 254.0, 85.0, 0.0, 0.0, 0.0, 47.0, 49.0, 116.0, 144.0, 150.0, 241.0, 243.0, 234.0, 179.0, 241.0, 252.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 150.0, 253.0, 237.0, 207.0, 207.0, 207.0, 253.0, 254.0, 250.0, 240.0, 198.0, 143.0, 91.0, 28.0, 5.0, 233.0, 250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 119.0, 177.0, 177.0, 177.0, 177.0, 177.0, 98.0, 56.0, 0.0, 0.0, 0.0, 0.0, 0.0, 102.0, 254.0, 220.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 137.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 57.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 57.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 255.0, 94.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 96.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 153.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 255.0, 153.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 254.0, 153.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

Use the output as input to the model.

From `burn-algorithms` root directory run:

```bash
cd mnist-inference
```

```bash
cargo build --release --target wasm32-wasip1 --bin mnist-inference
```

```bash
wasmtime ../target/wasm32-wasip1/release/mnist-inference.wasm '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 67.0, 232.0, 39.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 62.0, 81.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0, 180.0, 39.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 153.0, 210.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 220.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 27.0, 254.0, 162.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 222.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 183.0, 254.0, 125.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 245.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 198.0, 254.0, 56.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 120.0, 254.0, 163.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 23.0, 231.0, 254.0, 29.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 159.0, 254.0, 120.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 163.0, 254.0, 216.0, 16.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 159.0, 254.0, 67.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 14.0, 86.0, 178.0, 248.0, 254.0, 91.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 159.0, 254.0, 85.0, 0.0, 0.0, 0.0, 47.0, 49.0, 116.0, 144.0, 150.0, 241.0, 243.0, 234.0, 179.0, 241.0, 252.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 150.0, 253.0, 237.0, 207.0, 207.0, 207.0, 253.0, 254.0, 250.0, 240.0, 198.0, 143.0, 91.0, 28.0, 5.0, 233.0, 250.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 119.0, 177.0, 177.0, 177.0, 177.0, 177.0, 98.0, 56.0, 0.0, 0.0, 0.0, 0.0, 0.0, 102.0, 254.0, 220.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 137.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 57.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 57.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 255.0, 94.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 96.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 254.0, 153.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 169.0, 255.0, 153.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 96.0, 254.0, 153.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]'
```

The first and only argument is the input data in the form of a JSON array.

The output should be something like:

```bash
4
```

If you want to run inference as compile binary, you can use the following command:

```bash
cargo build --release
```

```bash
../target/release/mnist-inference <IMAGE_PIXEL_ARRAY>
```

### Wine Quality

From `burn-algorithms` root directory run:

```bash
cd winequality-inference
```

```bash
cargo build --release --target wasm32-wasip1 --bin winequality-inference
```

```bash
wasmtime ../target/wasm32-wasip1/release/winequality-inference.wasm '{"fixed_acidity": 5.5,"volatile_acidity": 0.29,"citric_acid": 0.3,"residual_sugar": 1.1,"chlorides": 0.022,"free_sulfur_dioxide": 20,"total_sulfur_dioxide": 110,"density": 0.98869,"ph": 3.34,"sulphates": 0.38,"alcohol": 12.8}'
```

The first and only argument is the input data in the form of a JSON object.

The output should be something like:

```bash
0.5728824
```

If you want to run inference as compile binary, you can use the following command:

```bash
cargo build --release
```

```bash
../target/release/winequality-inference '{"fixed_acidity": 5.5,"volatile_acidity": 0.29,"citric_acid": 0.3,"residual_sugar": 1.1,"chlorides": 0.022,"free_sulfur_dioxide": 20,"total_sulfur_dioxide": 110,"density": 0.98869,"ph": 3.34,"sulphates": 0.38,"alcohol": 12.8}'
```

## References

- [Burn-Github](https://burn.dev/)
- [Burn-Documentation](https://burn.dev/docs/burn/)
- [Burn-Book](https://burn.dev/book/)
- [Wasmtime](https://wasmtime.dev/)
- [wasm32-wasip1](https://doc.rust-lang.org/rustc/platform-support/wasm32-wasip1.html#wasm32-wasip1)
