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

## Addition

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

## Iris dataset

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

## Wine Quality dataset

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

## MNIST dataset

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

## Cifar-10 dataset

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

## AG News

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

## IMDB

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

## References

- [Burn-Github](https://burn.dev/)
- [Burn-Documentation](https://burn.dev/docs/burn/)
- [Burn-Book](https://burn.dev/book/)
