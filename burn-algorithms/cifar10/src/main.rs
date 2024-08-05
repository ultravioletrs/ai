#[cfg(not(feature = "wgpu"))]
mod ndarray {
    use burn::{
        backend::{
            ndarray::{NdArray, NdArrayDevice},
            Autodiff,
        },
        optim::{momentum::MomentumConfig, SgdConfig},
    };
    use cifar10::training::{train, TrainingConfig};

    pub fn run() {
        train::<Autodiff<NdArray>>(
            TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.,
                nesterov: false,
            }))),
            NdArrayDevice::default(),
        );
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::{
        backend::{
            wgpu::{Wgpu, WgpuDevice},
            Autodiff,
        },
        optim::{momentum::MomentumConfig, SgdConfig},
    };
    use cifar10::training::{train, TrainingConfig};

    pub fn run() {
        train::<Autodiff<Wgpu>>(
            TrainingConfig::new(SgdConfig::new().with_momentum(Some(MomentumConfig {
                momentum: 0.9,
                dampening: 0.,
                nesterov: false,
            }))),
            WgpuDevice::default(),
        );
    }
}

fn main() {
    #[cfg(not(feature = "wgpu"))]
    ndarray::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
