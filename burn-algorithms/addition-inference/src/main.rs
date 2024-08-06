// Copied from: https://github.com/ultravioletrs/ai/blob/main/burn-algorithms/addition/src/main.rs

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};
use futures::executor;

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
    let a = [[2., 3.], [4., 5.]];
    let b = [[std::f32::consts::PI, 1.], [1., std::f32::consts::PI]];
    let result = executor::block_on(addition::<burn::backend::NdArray>(a, b));
    if cfg!(feature = "cocos") {
        match lib::save_results_to_file(result.to_string(), "results/results.txt".to_string()) {
            Ok(_) => (),
            Err(e) => {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }
    } else {
        println!("{:}", result);
    }
}
