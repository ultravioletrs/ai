# Burn Algorithms

- [x] Matrix Addition
- [x] Classification problem - (Iris dataset)
- [x] Regression problem - (Wine Quality dataset)
- [x] Image Classification - (MNIST dataset)
- [x] Image Classification - (CIFAR-10 dataset)
- [x] Text Classification - (AG News dataset)
- [x] Text Classification - (IMDB dataset)

## Addition

This is a simple matrix addition example. It adds two matrices and prints the result.

To run the example, use the following command:

```bash
cargo run --release --bin addition --features wgpu
```

The output should be:

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

The output should be:

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
cargo run --bin cifar10 --features wgpu
```

or

```bash
cargo run --bin cifar10 --features ndarray
```
