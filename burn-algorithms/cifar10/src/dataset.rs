use burn::data::dataset::vision::ImageFolderDataset;
use std::path::{Path, PathBuf};

pub trait CIFAR10Loader {
    fn cifar10_train() -> Self;
    fn cifar10_test() -> Self;
}

impl CIFAR10Loader for ImageFolderDataset {
    fn cifar10_train() -> Self {
        let root = data_path();

        Self::new_classification(root.join("train")).unwrap()
    }

    fn cifar10_test() -> Self {
        let root = data_path();

        Self::new_classification(root.join("test")).unwrap()
    }
}

fn data_path() -> PathBuf {
    let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
    let cifar_dir = example_dir.join("data/cifar10");

    let labels_file = cifar_dir.join("labels.txt");
    if !labels_file.exists() {
        panic!("Downloading CIFAR-10 dataset...");
    }

    cifar_dir
}
