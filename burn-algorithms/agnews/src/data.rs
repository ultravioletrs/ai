use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use burn::data::dataset::InMemDataset;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};
use derive_new::new;
use flate2::read::GzDecoder;
use nn::attention::generate_padding_mask;
use tar::Archive;

#[derive(new, Clone, Debug)]
pub struct ClassificationItem {
    pub text: String,
    pub label: usize,
}

pub trait ClassificationDataset: Dataset<ClassificationItem> {
    fn num_classes() -> usize;
    fn class_name(label: usize) -> String;
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct AgNewsItem {
    pub title: String,
    pub description: String,
    pub label: usize,
}

pub struct AgNewsDataset {
    dataset: InMemDataset<AgNewsItem>,
}

impl Dataset<ClassificationItem> for AgNewsDataset {
    fn get(&self, index: usize) -> Option<ClassificationItem> {
        self.dataset.get(index).map(|item| {
            ClassificationItem::new(format!("{} {}", item.title, item.description), item.label)
        })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl AgNewsDataset {
    pub fn train(agnews_dir: &Path) -> Self {
        Self::new(agnews_dir, "train.csv")
    }

    pub fn test(agnews_dir: &Path) -> Self {
        Self::new(agnews_dir, "test.csv")
    }

    pub fn new(agnews_dir: &Path, split: &str) -> Self {
        let dataset = Self::read(agnews_dir, split);
        Self { dataset }
    }

    pub fn data_path() -> PathBuf {
        let data_dir = if cfg!(feature = "cocos") {
            let datasets_dir = Path::new("datasets");
            let files = std::fs::read_dir(datasets_dir).expect("Failed to read directory");
            let tarball_without_ext = files
                .map(|f| f.expect("Failed to read file").path())
                .next()
                .expect("No file found in the directory");
            let tarball = tarball_without_ext.with_extension("tgz");
            std::fs::copy(tarball_without_ext.as_path(), &tarball).expect("Failed to copy file");
            std::fs::remove_file(tarball_without_ext.as_path()).expect("Failed to remove file");
            let tarball_file = File::open(&tarball).expect("Failed to open file");
            let tar = GzDecoder::new(tarball_file);
            let mut archive = Archive::new(tar);
            archive
                .unpack(datasets_dir)
                .expect("Failed to unpack tarball");

            let agnews_dir = datasets_dir.join("ag_news_csv");

            let labels_file = agnews_dir.join("classes.txt");
            if !labels_file.exists() {
                panic!("Download the CIFAR-10 dataset from https://s3.amazonaws.com/fast-ai-sample/cifar10.tgz and place it in the data directory");
            }

            agnews_dir
        } else {
            let example_dir = Path::new(file!())
                .parent()
                .expect("Failed to get parent")
                .parent()
                .expect("Failed to get parent");
            example_dir.join("data/ag_news_csv/")
        };

        data_dir
    }

    fn read(agnews_dir: &Path, file_name: &str) -> InMemDataset<AgNewsItem> {
        let csv_file = agnews_dir.join(file_name);

        if !csv_file.exists() {
            panic!("Download the AG News dataset from https://s3.amazonaws.com/fast-ai-nlp/ag_news_csv.tgz and place it in the data directory");
        }

        let mut rdr = csv::ReaderBuilder::new();
        let rdr = rdr.delimiter(b',');

        InMemDataset::from_csv(csv_file, rdr).unwrap()
    }
}

impl ClassificationDataset for AgNewsDataset {
    fn num_classes() -> usize {
        4
    }

    fn class_name(label: usize) -> String {
        match label {
            0 => "World",
            1 => "Sports",
            2 => "Business",
            3 => "Technology",
            _ => panic!("invalid class"),
        }
        .to_string()
    }
}

pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str) -> Vec<usize>;

    fn decode(&self, tokens: &[usize]) -> String;

    fn vocab_size(&self) -> usize;

    fn pad_token(&self) -> usize;

    fn pad_token_value(&self) -> String {
        self.decode(&[self.pad_token()])
    }
}

pub struct BertCasedTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl Default for BertCasedTokenizer {
    fn default() -> Self {
        Self {
            tokenizer: tokenizers::Tokenizer::from_pretrained("bert-base-cased", None).unwrap(),
        }
    }
}

impl Tokenizer for BertCasedTokenizer {
    fn encode(&self, value: &str) -> Vec<usize> {
        let tokens = self.tokenizer.encode(value, true).unwrap();
        tokens.get_ids().iter().map(|t| *t as usize).collect()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        let tokens = tokens.iter().map(|t| *t as u32).collect::<Vec<u32>>();
        self.tokenizer.decode(&tokens, false).unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.tokenizer.token_to_id("[PAD]").unwrap() as usize
    }
}

#[derive(Clone, new)]
pub struct ClassificationBatcher<B: Backend> {
    tokenizer: Arc<dyn Tokenizer>,
    device: B::Device,
    max_seq_length: usize,
}

#[derive(Debug, Clone, new)]
pub struct ClassificationTrainingBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub labels: Tensor<B, 1, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

#[derive(Debug, Clone, new)]
pub struct ClassificationInferenceBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask_pad: Tensor<B, 2, Bool>,
}

impl<B: Backend> Batcher<ClassificationItem, ClassificationTrainingBatch<B>>
    for ClassificationBatcher<B>
{
    fn batch(&self, items: Vec<ClassificationItem>) -> ClassificationTrainingBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());
        let mut labels_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item.text));
            labels_list.push(Tensor::from_data(
                TensorData::from([(item.label as i64).elem::<B::IntElem>()]),
                &self.device,
            ));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &self.device,
        );

        ClassificationTrainingBatch {
            tokens: mask.tensor,
            labels: Tensor::cat(labels_list, 0),
            mask_pad: mask.mask,
        }
    }
}

impl<B: Backend> Batcher<String, ClassificationInferenceBatch<B>> for ClassificationBatcher<B> {
    fn batch(&self, items: Vec<String>) -> ClassificationInferenceBatch<B> {
        let mut tokens_list = Vec::with_capacity(items.len());

        for item in items {
            tokens_list.push(self.tokenizer.encode(&item));
        }

        let mask = generate_padding_mask(
            self.tokenizer.pad_token(),
            tokens_list,
            Some(self.max_seq_length),
            &B::Device::default(),
        );

        ClassificationInferenceBatch {
            tokens: mask.tensor.to_device(&self.device),
            mask_pad: mask.mask.to_device(&self.device),
        }
    }
}
