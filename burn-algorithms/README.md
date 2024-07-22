# Burn Algorithms

The need to run machine learning algorithms either training or making inference from a single algorithm has been a challenge with Python based models This is where Rust comes in. Rust is a systems programming language that runs blazingly fast, prevents segfaults, and guarantees thread safety. Unlike C++ or C, Rust is memory safe by default.

Burn Algorithms is a collection machine learning algorithms that is written in Rust using [burn](https://burn.dev/). Burn is a deep learning Framework for Rust that is designed to be extremely flexible, compute efficient and highly portable. Burn strives to be as fast as possible on as many hardwares as possible, with robust implementations. With Burn, you can run your machine learning models on the CPU, GPU, WebAssembly, and other hardwares.

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

To run the example, use the following command:

```bash
cargo run --release --bin addition --features wgpu
```

The output should be something like:

```bash
    Finished `release` profile [optimized] target(s) in 0.16s
    Running `target/release/addition`

Tensor {
  data:
[[5.141593, 4.0],
 [5.0, 8.141593]],
  shape:  [2, 2],
  device:  BestAvailable,
  backend:  "fusion<jit<wgpu>>",
  kind:  "Float",
  dtype:  "f32",
}
```

You can also run the example using the `ndarray` feature:

```bash
cargo run --release --bin addition --features ndarray
```

The output should be something like:

```bash
    Finished `release` profile [optimized] target(s) in 0.16s
    Running `target/release/addition`
Tensor {
  data:
[[5.141593, 4.0],
 [5.0, 8.141593]],
  shape:  [2, 2],
  device:  Cpu,
  backend:  "ndarray",
  kind:  "Float",
  dtype:  "f32",
}
```

### Iris dataset

The dataset is already downloaded and stored in the `data` folder inside the `iris` folder.

Run the following command to train the model on wgpu or ndrray.

```bash
cargo run --release --bin iris --features wgpu
```

The output should be something like:

```bash
  Finished `release` profile [optimized] target(s) in 0.19s
  Running `target/release/iris`
Train Dataset Size: 120
Test Dataset Size: 30
======================== Learner Summary ========================
Model: ClassificationModel[num_params=9091]
Total Epochs: 100


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 35.833   | 1        | 98.333   | 100      |
| Train | Loss     | 0.053    | 99       | 1.141    | 1        |
| Valid | Accuracy | 56.667   | 2        | 100.000  | 92       |
| Valid | Loss     | 0.031    | 92       | 0.950    | 1        |
```

or

```bash
cargo run --release --bin iris --features ndarray
```

The output should be something like:

```bash
  Finished `release` profile [optimized] target(s) in 0.19s
  Running `target/release/iris`
Train Dataset Size: 120
Test Dataset Size: 30
======================== Learner Summary ========================
Model: ClassificationModel[num_params=9091]
Total Epochs: 100


| Split | Metric   | Min.     | Epoch    | Max.     | Epoch    |
|-------|----------|----------|----------|----------|----------|
| Train | Accuracy | 47.500   | 1        | 99.167   | 23       |
| Train | Loss     | 0.053    | 99       | 1.041    | 1        |
| Valid | Accuracy | 66.667   | 1        | 100.000  | 92       |
| Valid | Loss     | 0.031    | 92       | 0.837    | 1        |
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

Then, run the following command to train the model on wgpu or ndrray.

```bash
cargo run --release --bin winequality --features wgpu
```

The output should be something like:

```bash
  Finished `release` profile [optimized] target(s) in 0.16s
  Running `target/release/winequality`
Train Dataset Size: 3918
Test Dataset Size: 980
======================== Learner Summary ========================
Model: RegressionModel[num_params=12]
Total Epochs: 100


| Split | Metric | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------|----------|----------|----------|----------|
| Train | Loss   | 0.028    | 100      | 0.120    | 1        |
| Valid | Loss   | 0.042    | 100      | 0.078    | 1        |
```

or

```bash
cargo run --release --bin winequality --features ndarray
```

The output should be something like:

```bash
  Finished `release` profile [optimized] target(s) in 0.16s
  Running `target/release/winequality`
Train Dataset Size: 3918
Test Dataset Size: 980
======================== Learner Summary ========================
Model: RegressionModel[num_params=12]
Total Epochs: 100


| Split | Metric | Min.     | Epoch    | Max.     | Epoch    |
|-------|--------|----------|----------|----------|----------|
| Train | Loss   | 0.028    | 100      | 0.171    | 1        |
| Valid | Loss   | 0.042    | 100      | 0.110    | 1        |
```

### MNIST dataset

First, download the dataset from <https://www.kaggle.com/datasets/playlist/mnistzip/data?select=mnist_png> and extract it in the `data` folder inside the `mnist` folder.

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
cargo run --release --bin mnist --features wgpu
```

The output should be something like:

```bash

```

or

```bash
cargo run --release --bin mnist --features ndarray
```

The output should be something like:

```bash

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
cargo run --release --bin cifar10 --features wgpu
```

The output should be something like:

```bash

```

or

```bash
cargo run --release --bin cifar10 --features ndarray
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

Then, run the following command to train the model on wgpu or ndrray.

```bash
cargo run --release --bin agnews --features wgpu
```

The output should be something like:

```bash

```

or

```bash
cargo run --release --bin agnews --features ndarray
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
cargo run --release --bin imdb --features wgpu
```

The output should be something like:

```bash

```

or

```bash
cargo run --release --bin imdb --features ndarray
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
