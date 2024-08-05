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
use dircpy::copy_dir;
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
    pub fn train(path: &PathBuf) -> Self {
        Self::new("train", path)
    }

    pub fn test(path: &PathBuf) -> Self {
        Self::new("test", path)
    }

    pub fn new(split: &str, path: &PathBuf) -> Self {
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

    pub fn read() -> PathBuf {
        let csv_file = if cfg!(feature = "cocos") {
            let wine_dir = Path::new("datasets");
            let files = std::fs::read_dir(wine_dir).expect("Failed to read directory");
            let zipped_file_without_ext = files
                .map(|f| f.expect("Failed to read file").path())
                .next()
                .expect("No file found in the directory");
            let zipped_file = zipped_file_without_ext.with_extension("zip");
            std::fs::copy(zipped_file_without_ext.as_path(), &zipped_file)
                .expect("Failed to copy file");
            std::fs::remove_file(zipped_file_without_ext.as_path()).expect("Failed to remove file");
            simple_zip::zip::Decompress::local_buffer(&zipped_file);
            let src = wine_dir
                .parent()
                .expect("Failed to get parent")
                .join("data");
            copy_dir(src, wine_dir).expect("Failed to copy directory");
            wine_dir.join("winequality-white.csv")
        } else {
            let example_dir = Path::new(file!())
                .parent()
                .expect("Failed to get parent")
                .parent()
                .expect("Failed to get parent");
            let wine_dir = example_dir.join("data/");

            wine_dir.join("winequality-white.csv")
        };
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
}

impl<B: Backend> Batcher<WineQualityItem, WineQualityBatch<B>> for WineQualityBatcher<B> {
    fn batch(&self, items: Vec<WineQualityItem>) -> WineQualityBatch<B> {
        let mut inputs: Vec<Tensor<B, 2>> = Vec::new();

        // The constants are the min and max values of the dataset
        for item in items.iter() {
            let input_tensor = Tensor::<B, 1>::from_floats(
                [
                    (item.fixed_acidity as f32 - 3.8) / (14.2 - 3.8),
                    (item.volatile_acidity as f32 - 0.08) / (1.1 - 0.08),
                    (item.citric_acid as f32 - 0.0) / (1.66 - 0.0),
                    (item.residual_sugar as f32 - 0.6) / (65.8 - 0.6),
                    (item.chlorides as f32 - 0.009) / (0.346 - 0.009),
                    (item.free_sulfur_dioxide as f32 - 2.0) / (289.0 - 2.0),
                    (item.total_sulfur_dioxide as f32 - 9.0) / (440.0 - 9.0),
                    (item.density as f32 - 0.98711) / (1.03898 - 0.98711),
                    (item.ph as f32 - 2.72) / (3.82 - 2.72),
                    (item.sulphates as f32 - 0.22) / (1.08 - 0.22),
                    (item.alcohol as f32 - 8.0) / (14.2 - 8.0),
                ],
                &self.device,
            );

            inputs.push(input_tensor.unsqueeze());
        }

        let inputs = Tensor::cat(inputs, 0);

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1>::from_floats(
                    [(item.quality as f32 - 3.0) / (9.0 - 3.0)],
                    &self.device,
                )
            })
            .collect();

        let targets = Tensor::cat(targets, 0);

        WineQualityBatch { inputs, targets }
    }
}
