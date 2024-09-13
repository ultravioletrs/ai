use burn::{
    config::Config,
    module::Module,
    nn::{
        loss::CrossEntropyLossConfig,
        transformer::{TransformerEncoder, TransformerEncoderConfig, TransformerEncoderInput},
        Embedding, EmbeddingConfig, Linear, LinearConfig,
    },
    tensor::{
        activation::softmax,
        backend::{AutodiffBackend, Backend},
        Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::data::{ClassificationInferenceBatch, ClassificationTrainingBatch};

#[derive(Config)]
pub struct ModelConfig {
    pub transformer: TransformerEncoderConfig,
    pub num_classes: usize,
    pub vocab_size: usize,
    pub max_length: usize,
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding_token: Embedding<B>,
    embedding_position: Embedding<B>,
    transformer: TransformerEncoder<B>,
    output: Linear<B>,
    num_classes: usize,
    max_length: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        let embedding_token =
            EmbeddingConfig::new(self.vocab_size, self.transformer.d_model).init(device);
        let embedding_position =
            EmbeddingConfig::new(self.max_length, self.transformer.d_model).init(device);

        let transformer = self.transformer.init(device);
        let output = LinearConfig::new(self.transformer.d_model, self.num_classes).init(device);

        Model {
            embedding_token,
            embedding_position,
            transformer,
            output,
            num_classes: self.num_classes,
            max_length: self.max_length,
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, item: ClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        let tokens = item.tokens.to_device(device);
        let labels = item.labels.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(&[batch_size]);
        let embedding_positions = self.embedding_position.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);

        let output_classification = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.num_classes]);

        let loss = CrossEntropyLossConfig::new()
            .init(&output_classification.device())
            .forward(output_classification.clone(), labels.clone());

        ClassificationOutput {
            loss,
            output: output_classification,
            targets: labels,
        }
    }

    pub fn infer(&self, item: ClassificationInferenceBatch<B>) -> Tensor<B, 2> {
        let [batch_size, seq_length] = item.tokens.dims();
        let device = &self.embedding_token.devices()[0];

        let tokens = item.tokens.to_device(device);
        let mask_pad = item.mask_pad.to_device(device);

        let index_positions = Tensor::arange(0..seq_length as i64, device)
            .reshape([1, seq_length])
            .repeat(&[batch_size]);
        let embedding_positions = self.embedding_position.forward(index_positions);
        let embedding_tokens = self.embedding_token.forward(tokens);
        let embedding = (embedding_positions + embedding_tokens) / 2;

        let encoded = self
            .transformer
            .forward(TransformerEncoderInput::new(embedding).mask_pad(mask_pad));
        let output = self.output.forward(encoded);
        let output = output
            .slice([0..batch_size, 0..1])
            .reshape([batch_size, self.num_classes]);

        softmax(output, 1)
    }
}

impl<B: AutodiffBackend> TrainStep<ClassificationTrainingBatch<B>, ClassificationOutput<B>>
    for Model<B>
{
    fn step(&self, item: ClassificationTrainingBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<ClassificationTrainingBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, item: ClassificationTrainingBatch<B>) -> ClassificationOutput<B> {
        self.forward(item)
    }
}
