use std::path::{Path, PathBuf};

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            transform::{PartialDataset, ShuffledDataset},
            Dataset, InMemDataset,
        },
    },
    prelude::*,
};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IrisItem {
    #[serde(rename = "SepalLengthCm")]
    pub sepal_length: f64,

    #[serde(rename = "SepalWidthCm")]
    pub sepal_width: f64,

    #[serde(rename = "PetalLengthCm")]
    pub petal_length: f64,

    #[serde(rename = "PetalWidthCm")]
    pub petal_width: f64,

    #[serde(rename = "Species")]
    pub species: String,
}

type ShuffledData = ShuffledDataset<InMemDataset<IrisItem>, IrisItem>;
type PartialData = PartialDataset<ShuffledData, IrisItem>;

pub struct IrisDataset {
    dataset: PartialData,
}

fn class_label(class: &str) -> i8 {
    match class {
        "Iris-setosa" => 0,
        "Iris-versicolor" => 1,
        "Iris-virginica" => 2,
        _ => panic!("Invalid class"),
    }
}

impl Dataset<IrisItem> for IrisDataset {
    fn get(&self, index: usize) -> Option<IrisItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl IrisDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let path = IrisDataset::read();

        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');

        let dataset = InMemDataset::from_csv(path, rdr).unwrap();

        let len = dataset.len();

        let dataset = ShuffledDataset::with_seed(dataset, 42);

        // The dataset from HuggingFace has only train split, so we manually split the train dataset into train
        // and test in a 80-20 ratio
        let filtered_dataset = match split {
            "train" => PartialData::new(dataset, 0, len * 8 / 10),
            "test" => PartialData::new(dataset, len * 8 / 10, len),
            _ => panic!("Invalid split type"),
        };

        Self {
            dataset: filtered_dataset,
        }
    }

    fn read() -> PathBuf {
        let iris_dir = Path::new("datasets");
        let files = std::fs::read_dir(iris_dir).expect("Failed to read directory");
        let csv_file = files
            .map(|f| f.expect("Failed to read file").path())
            .next()
            .expect("No file found in the directory");

        if !csv_file.exists() {
            panic!("Download the Iris dataset from https://www.kaggle.com/datasets/saurabh00007/iriscsv and place it in the data directory");
        }

        csv_file
    }
}

#[derive(Clone, Debug)]
pub struct IrisBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct IrisBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> IrisBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<IrisItem, IrisBatch<B>> for IrisBatcher<B> {
    fn batch(&self, items: Vec<IrisItem>) -> IrisBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.sepal_length as f32,
                    item.sepal_width as f32,
                    item.petal_length as f32,
                    item.petal_width as f32,
                ],
                &self.device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data(
                    TensorData::from([(class_label(&item.species) as i64).elem::<B::IntElem>()]),
                    &self.device,
                )
            })
            .collect();

        let targets = Tensor::cat(targets, 0);

        IrisBatch { inputs, targets }
    }
}
