use burn::{
    nn::transformer::TransformerEncoderConfig,
    optim::{decay::WeightDecayConfig, AdamConfig},
    tensor::backend::AutodiffBackend,
};

use agnews::{data::AgNewsDataset, training::ExperimentConfig};

pub fn launch<B: AutodiffBackend>(devices: B::Device) {
    let config = ExperimentConfig::new(
        TransformerEncoderConfig::new(256, 1024, 8, 4)
            .with_norm_first(true)
            .with_quiet_softmax(true),
        AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5))),
    );

    agnews::training::train::<B, AgNewsDataset>(
        devices,
        AgNewsDataset::train(),
        AgNewsDataset::test(),
        config,
        "imdb/artifacts/",
    );
}

#[cfg(feature = "ndarray")]
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
    #[cfg(feature = "ndarray")]
    ndarray::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
