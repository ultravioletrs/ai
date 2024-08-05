use burn::data::dataset::vision::ImageFolderDataset;
use flate2::read::GzDecoder;
use std::fs::File;
use std::path::{Path, PathBuf};
use tar::Archive;

pub trait CIFAR10Loader {
    fn cifar10_train(data_path: &PathBuf) -> Self;
    fn cifar10_test(data_path: &PathBuf) -> Self;
}

impl CIFAR10Loader for ImageFolderDataset {
    fn cifar10_train(data_path: &PathBuf) -> Self {
        Self::new_classification(data_path.join("train")).unwrap()
    }

    fn cifar10_test(data_path: &PathBuf) -> Self {
        Self::new_classification(data_path.join("test")).unwrap()
    }
}

pub fn data_path() -> PathBuf {
    let data_dir = if cfg!(feature = "cocos") {
        let datasets_dir = Path::new("datasets");
        let files = std::fs::read_dir(datasets_dir).unwrap();
        let tarball_without_ext = files
            .map(|f| f.unwrap().path())
            .next()
            .expect("No file found in the directory");
        let tarball = tarball_without_ext.with_extension("tgz");
        std::fs::copy(tarball_without_ext.as_path(), &tarball).unwrap();
        std::fs::remove_file(tarball_without_ext.as_path()).unwrap();
        let tarball_file = File::open(&tarball).unwrap();
        let tar = GzDecoder::new(tarball_file);
        let mut archive = Archive::new(tar);
        archive.unpack(datasets_dir).unwrap();

        let cifar_dir = datasets_dir.join("cifar10");

        let labels_file = cifar_dir.join("labels.txt");
        if !labels_file.exists() {
            panic!("Download the CIFAR-10 dataset from https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz and place it in the data directory");
        }

        cifar_dir
    } else {
        let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let cifar_dir = example_dir.join("data/cifar10");

        let labels_file = cifar_dir.join("labels.txt");
        if !labels_file.exists() {
            panic!("Download the CIFAR-10 dataset from https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz and place it in the data directory");
        }

        cifar_dir
    };

    data_dir
}
