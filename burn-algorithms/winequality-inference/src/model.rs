// Copied from the https://github.com/ultravioletrs/ai/blob/main/burn-algorithms/iris/src/model.rs package

use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    prelude::*,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

pub type NDBackend = burn::backend::ndarray::NdArray<f32>;

static STATE_ENCODED: &[u8] = include_bytes!("../../artifacts/winequality/model.bin");

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input_layer: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn new(device: &B::Device) -> Self {
        let input_layer = LinearConfig::new(11, 1).with_bias(true).init(device);

        Model { input_layer }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = input.detach();
        self.input_layer.forward(x)
    }
}

pub fn build_and_load_model() -> Result<Model<NDBackend>, String> {
    let model: Model<NDBackend> = Model::new(&Default::default());
    let record = match BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec(), &Default::default())
    {
        Ok(record) => record,
        Err(e) => return Err(format!("Failed to load model: {}", e)),
    };

    Ok(model.load_record(record))
}
