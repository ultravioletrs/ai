use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{decay::WeightDecayConfig, AdamConfig},
    tensor::backend::AutodiffBackend,
};

use agnews::{data::AgNewsDataset, training::TrainingConfig};

#[cfg(feature = "cocos")]
static ARTIFACT_DIR: &str = "results";

#[cfg(not(feature = "cocos"))]
static ARTIFACT_DIR: &str = "artifacts/agnews/";


pub fn launch<B: AutodiffBackend>(devices: B::Device) {
    let config = TrainingConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    let data_path = AgNewsDataset::data_path();

    agnews::training::train::<B, AgNewsDataset>(
        devices,
        AgNewsDataset::train(&data_path),
        AgNewsDataset::test(&data_path),
        config,
        ARTIFACT_DIR,
    );
}

#[cfg(not(feature = "wgpu"))]
mod ndarray {
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };

    use crate::launch;

    pub fn run() {
        let devices = NdArrayDevice::default();
        launch::<Autodiff<NdArray>>(devices);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use crate::launch;
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };

    pub fn run() {
        let device = WgpuDevice::default();
        launch::<Autodiff<Wgpu>>(device);
    }
}

fn main() {
    #[cfg(not(feature = "wgpu"))]
    ndarray::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
