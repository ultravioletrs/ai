use std::path::Path;
use std::sync::Arc;

use burn::data::dataset::InMemDataset;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::{backend::Backend, Tensor},
};
use derive_new::new;
use nn::attention::generate_padding_mask;

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
    pub fn train() -> Self {
        Self::new("train.csv")
    }

    pub fn test() -> Self {
        Self::new("test.csv")
    }

    pub fn new(split: &str) -> Self {
        let dataset = Self::read(split);
        Self { dataset }
    }

    fn read(file_name: &str) -> InMemDataset<AgNewsItem> {
        let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
        let iris_dir = example_dir.join("data/ag_news_csv/");

        let csv_file = iris_dir.join(file_name);

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
                Data::from([(item.label as i64).elem()]),
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
