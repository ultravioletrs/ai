// Copied from the https://github.com/ultravioletrs/ai/blob/main/burn-algorithms/mnist/src/model.rs package

use burn::{
    module::Module,
    prelude::*,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
use nn::{BatchNorm, PaddingConfig2d};

static STATE_ENCODED: &[u8] = include_bytes!("../../artifacts/mnist/model.bin");
const NUM_CLASSES: usize = 10;
pub type NDBackend = burn::backend::ndarray::NdArray<f32>;

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: ConvBlock<B>,
    conv2: ConvBlock<B>,
    conv3: ConvBlock<B>,
    dropout: nn::Dropout,
    fc1: nn::Linear<B>,
    fc2: nn::Linear<B>,
    activation: nn::Gelu,
}

impl<B: Backend> Default for Model<B> {
    fn default() -> Self {
        let device = B::Device::default();
        Self::new(&device)
    }
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = ConvBlock::new([1, 8], [3, 3], device); // out: [Batch,8,26,26]
        let conv2 = ConvBlock::new([8, 16], [3, 3], device); // out: [Batch,16,24x24]
        let conv3 = ConvBlock::new([16, 24], [3, 3], device); // out: [Batch,24,22x22]
        let hidden_size = 24 * 22 * 22;
        let fc1 = nn::LinearConfig::new(hidden_size, 32)
            .with_bias(false)
            .init(device);
        let fc2 = nn::LinearConfig::new(32, NUM_CLASSES)
            .with_bias(false)
            .init(device);

        let dropout = nn::DropoutConfig::new(0.5).init();

        Self {
            conv1,
            conv2,
            conv3,
            dropout,
            fc1,
            fc2,
            activation: nn::Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = input.dims();

        let x = input.reshape([batch_size, 1, height, width]).detach();
        let x = self.conv1.forward(x);
        let x = self.conv2.forward(x);
        let x = self.conv3.forward(x);

        let [batch_size, channels, height, width] = x.dims();
        let x = x.reshape([batch_size, channels * height * width]);

        let x = self.dropout.forward(x);
        let x = self.fc1.forward(x);
        let x = self.activation.forward(x);

        self.fc2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    conv: nn::conv::Conv2d<B>,
    norm: BatchNorm<B, 2>,
    activation: nn::Gelu,
}

impl<B: Backend> ConvBlock<B> {
    pub fn new(channels: [usize; 2], kernel_size: [usize; 2], device: &B::Device) -> Self {
        let conv = nn::conv::Conv2dConfig::new(channels, kernel_size)
            .with_padding(PaddingConfig2d::Valid)
            .init(device);
        let norm = nn::BatchNormConfig::new(channels[1]).init(device);

        Self {
            conv,
            norm,
            activation: nn::Gelu::new(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv.forward(input);
        let x = self.norm.forward(x);

        self.activation.forward(x)
    }
}

pub fn build_and_load_model() -> Model<NDBackend> {
    let model: Model<NDBackend> = Model::new(&Default::default());
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec(), &Default::default())
        .expect("Failed to decode state");

    model.load_record(record)
}
