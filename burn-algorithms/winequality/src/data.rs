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
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WineQualityItem {
    #[serde(rename = "fixed acidity")]
    pub fixed_acidity: f64,

    #[serde(rename = "volatile acidity")]
    pub volatile_acidity: f64,

    #[serde(rename = "citric acid")]
    pub citric_acid: f64,

    #[serde(rename = "residual sugar")]
    pub residual_sugar: f64,

    #[serde(rename = "chlorides")]
    pub chlorides: f64,

    #[serde(rename = "free sulfur dioxide")]
    pub free_sulfur_dioxide: f64,

    #[serde(rename = "total sulfur dioxide")]
    pub total_sulfur_dioxide: f64,

    #[serde(rename = "density")]
    pub density: f64,

    #[serde(rename = "pH")]
    pub ph: f64,

    #[serde(rename = "sulphates")]
    pub sulphates: f64,

    #[serde(rename = "alcohol")]
    pub alcohol: f64,

    #[serde(rename = "quality")]
    pub quality: i64,
}

type ShuffledData = ShuffledDataset<InMemDataset<WineQualityItem>, WineQualityItem>;
type PartialData = PartialDataset<ShuffledData, WineQualityItem>;

pub struct WineQualityDataset {
    dataset: PartialData,
}

impl Dataset<WineQualityItem> for WineQualityDataset {
    fn get(&self, index: usize) -> Option<WineQualityItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl WineQualityDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let path = WineQualityDataset::read();

        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b';');

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
        let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let wine_dir = example_dir.join("data/");

        let csv_file = wine_dir.join("winequality-white.csv");
        if !csv_file.exists() {
            panic!("Download the Wine Quality dataset from https://archive.ics.uci.edu/dataset/186/wine+quality and place it in the data directory");
        }

        csv_file
    }
}

#[derive(Clone, Debug)]
pub struct WineQualityBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct WineQualityBatch<B: Backend> {
    pub inputs: Tensor<B, 2>,
    pub targets: Tensor<B, 1>,
}

impl<B: Backend> WineQualityBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }

    pub fn min_max_norm<const D: usize>(&self, inp: Tensor<B, D>) -> Tensor<B, D> {
        let min = inp.clone().min_dim(0);
        let max = inp.clone().max_dim(0);
        (inp.clone() - min.clone()).div(max - min)
    }
}

impl<B: Backend> Batcher<WineQualityItem, WineQualityBatch<B>> for WineQualityBatcher<B> {
    fn batch(&self, items: Vec<WineQualityItem>) -> WineQualityBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    item.fixed_acidity as f32,
                    item.volatile_acidity as f32,
                    item.citric_acid as f32,
                    item.residual_sugar as f32,
                    item.chlorides as f32,
                    item.free_sulfur_dioxide as f32,
                    item.total_sulfur_dioxide as f32,
                    item.density as f32,
                    item.ph as f32,
                    item.sulphates as f32,
                    item.alcohol as f32,
                ],
                &self.device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);
        let inputs = self.min_max_norm(inputs);

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats([item.quality as f32], &self.device))
            .collect();

        let targets = Tensor::cat(targets, 0);
        let targets = self.min_max_norm(targets);

        WineQualityBatch { inputs, targets }
    }
}
