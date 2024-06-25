use crate::data::WineQualityBatch;
use burn::{
    nn::{
        loss::{MseLoss, Reduction::Mean},
        Linear, LinearConfig,
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{RegressionOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct RegressionModel<B: Backend> {
    input_layer: Linear<B>,
}

#[derive(Config)]
pub struct RegressionModelConfig {
    pub num_features: usize,
}

impl RegressionModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RegressionModel<B> {
        let input_layer = LinearConfig::new(self.num_features, 1)
            .with_bias(true)
            .init(device);

        RegressionModel { input_layer }
    }
}

impl<B: Backend> RegressionModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        self.input_layer.forward(x)
    }

    pub fn forward_step(&self, item: WineQualityBatch<B>) -> RegressionOutput<B> {
        let targets: Tensor<B, 2> = item.targets.unsqueeze();
        let output: Tensor<B, 2> = self.forward(item.inputs);

        let loss = MseLoss::new().forward(output.clone(), targets.clone(), Mean);

        RegressionOutput {
            loss,
            output,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<WineQualityBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: WineQualityBatch<B>) -> TrainOutput<RegressionOutput<B>> {
        let item = self.forward_step(item);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<WineQualityBatch<B>, RegressionOutput<B>> for RegressionModel<B> {
    fn step(&self, item: WineQualityBatch<B>) -> RegressionOutput<B> {
        self.forward_step(item)
    }
}
