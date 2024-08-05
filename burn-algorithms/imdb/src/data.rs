use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{
            transform::{PartialDataset, ShuffledDataset},
            Dataset, InMemDataset,
        },
    },
    prelude::*,
    tensor::{backend::Backend, Tensor},
};
use derive_new::new;
use nn::attention::generate_padding_mask;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(new, Clone, Debug)]
pub struct ClassificationItem {
    pub text: String,
    pub label: u8,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IMDBItem {
    pub review: String,
    pub sentiment: String,
}

fn sentiment_to_label(sentiment: String) -> u8 {
    match sentiment.as_str() {
        "negative" => 0,
        "positive" => 1,
        _ => panic!("invalid class"),
    }
}

type ShuffledData = ShuffledDataset<InMemDataset<IMDBItem>, IMDBItem>;
type PartialData = PartialDataset<ShuffledData, IMDBItem>;

pub struct IMDBDataset {
    dataset: PartialData,
}

impl Dataset<ClassificationItem> for IMDBDataset {
    fn get(&self, index: usize) -> Option<ClassificationItem> {
        self.dataset
            .get(index)
            .map(|item| ClassificationItem::new(item.review, sentiment_to_label(item.sentiment)))
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl IMDBDataset {
    pub fn train() -> Self {
        Self::new("train")
    }

    pub fn test() -> Self {
        Self::new("test")
    }

    pub fn new(split: &str) -> Self {
        let path = IMDBDataset::read();

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
        let csv_file = if cfg!(feature = "cocos") {
            let imdb_dir = Path::new("datasets");
            let files = std::fs::read_dir(imdb_dir).unwrap();
            files
                .map(|f| f.unwrap().path())
                .next()
                .expect("No file found in the directory")
        } else {
            let example_dir = Path::new(file!()).parent().unwrap().parent().unwrap();
            let imdb_dir = example_dir.join("data/");
            imdb_dir.join("IMDB Dataset.csv")
        };

        if !csv_file.exists() {
            panic!("Download the IMDB review dataset from https://huggingface.co/datasets/scikit-learn/imdb and place it in the data directory");
        }

        csv_file
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
