#[cfg(not(feature = "wgpu"))]
mod ndarray {
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };
    use mnist::training;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        training::run::<Autodiff<NdArray>>(device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };
    use mnist::training;

    pub fn run() {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device);
    }
}

fn main() {
    #[cfg(not(feature = "wgpu"))]
    ndarray::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
