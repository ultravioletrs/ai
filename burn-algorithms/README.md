# Burn Algorithms

- [x] Matrix Addition
- [x] Classification problem - (Iris dataset)
- [x] Regression problem - (Wine Quality dataset)
- [x] Image Classification - (MNIST dataset)
- [x] Image Classification - (CIFAR-10 dataset)
- [x] Text Classification - (AG News dataset)
- [x] Text Classification - (IMDB dataset)

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
cargo run --bin cifar10 --features ndrray
```
