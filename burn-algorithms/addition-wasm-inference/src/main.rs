// Copied from: https://github.com/ultravioletrs/ai/blob/main/burn-algorithms/addition/src/main.rs

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};
use futures::executor;
use serde_json;

pub async fn addition<B: Backend>(
    a: [[f32; 2]; 2],
    b: [[f32; 2]; 2],
) -> Data<<B as Backend>::FloatElem, 2> {
    let device = Default::default();

    let tensor1: Tensor<B, 2> = Tensor::from_floats(a, &device);
    let tensor2: Tensor<B, 2> = Tensor::from_floats(b, &device);

    let result = tensor1 + tensor2;

    #[cfg(not(target_family = "wasm"))]
    return result.into_data();

    #[cfg(target_family = "wasm")]
    return result.into_data().await;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let a: [[f32; 2]; 2] = serde_json::from_str(&args[1]).expect("Please provide a valid JSON input for example: [[1.0, 2.0], [3.0, 4.0]]");
    let b: [[f32; 2]; 2] = serde_json::from_str(&args[2]).expect("Please provide a valid JSON input for example: [[1.0, 2.0], [3.0, 4.0]]");
    let result: Data<f32, 2> = executor::block_on(addition::<burn::backend::NdArray>(a, b));
    println!("{:}", result);
}
