use crate::data::IrisBatch;
use burn::{
    nn::loss::CrossEntropyLossConfig,
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct ClassificationModel<B: Backend> {
    input_layer: nn::Linear<B>,
    hidden_layer: nn::Linear<B>,
    activation: nn::Relu,
    output_layer: nn::Linear<B>,
}

#[derive(Config)]
pub struct ClassificationModelConfig {
    pub input_size: usize,
    pub hidden_size: usize,
}

impl ClassificationModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> ClassificationModel<B> {
        let input_layer = nn::LinearConfig::new(self.input_size, self.hidden_size)
            .with_bias(true)
            .init(device);
        let hidden_layer = nn::LinearConfig::new(self.hidden_size, self.hidden_size / 2)
            .with_bias(true)
            .init(device);
        let output_layer = nn::LinearConfig::new(self.hidden_size / 2, 3)
            .with_bias(true)
            .init(device);

        ClassificationModel {
            input_layer,
            hidden_layer,
            activation: nn::Relu::new(),
            output_layer,
        }
    }
}

impl<B: Backend> ClassificationModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        let x = self.input_layer.forward(x);
        let x = self.activation.forward(x);
        let x = self.hidden_layer.forward(x);
        let x = self.activation.forward(x);
        self.output_layer.forward(x)
    }

    pub fn forward_classification(&self, item: IrisBatch<B>) -> ClassificationOutput<B> {
        let targets = item.targets;
        let output = self.forward(item.inputs);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<IrisBatch<B>, ClassificationOutput<B>>
    for ClassificationModel<B>
{
    fn step(&self, item: IrisBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<IrisBatch<B>, ClassificationOutput<B>> for ClassificationModel<B> {
    fn step(&self, item: IrisBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(item)
    }
}
