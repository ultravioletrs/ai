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
    let mut a = [[2., 3.], [4., 5.]];
    let mut b = [[std::f32::consts::PI, 1.], [1., std::f32::consts::PI]];
    if cfg!(feature = "input") {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 3 {
            eprintln!("Provide two JSON inputs for example: '[[1.0, 2.0], [3.0, 4.0]]' '[[1.0, 2.0], [3.0, 4.0]]'");
            std::process::exit(1);
        }
        a = match serde_json::from_str(&args[1]) {
            Ok(a) => a,
            Err(e) => {
                eprintln!(
                    "Provide a valid JSON input for example: '[[1.0, 2.0], [3.0, 4.0]]': {}",
                    e
                );
                std::process::exit(1);
            }
        };
        b = match serde_json::from_str(&args[2]) {
            Ok(b) => b,
            Err(e) => {
                eprintln!(
                    "Provide a valid JSON input for example: '[[1.0, 2.0], [3.0, 4.0]]': {}",
                    e
                );
                std::process::exit(1);
            }
        };
    }
    println!(
        "{:?}",
        executor::block_on(addition::<burn::backend::NdArray>(a, b))
    );
}
