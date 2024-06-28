use burn::tensor::{backend::Backend, Tensor};
use std::f32::consts::PI;

pub fn addition<B: Backend>() {
    let device = Default::default();

    let tensor1: Tensor<B, 2> = Tensor::from_floats([[2., 3.], [4., 5.]], &device);
    let tensor2: Tensor<B, 2> = Tensor::from_floats([[PI, 1.], [1., PI]], &device);

    println!("{:}", tensor1 + tensor2);
}

#[cfg(feature = "ndarray")]
pub fn run_ndarray() {
    addition::<burn::backend::NdArray>();
}

#[cfg(feature = "wgpu")]
pub fn run_wgpu() {
    addition::<burn::backend::Wgpu>();
}

fn main() {
    #[cfg(feature = "ndarray")]
    run_ndarray();
    #[cfg(feature = "wgpu")]
    run_wgpu();
}
