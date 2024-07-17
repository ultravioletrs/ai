use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};


pub fn addition<B: Backend>(
    mut a: [[f32; 2]; 2],
    mut b: [[f32; 2]; 2],
) -> Result<Data<<B as Backend>::FloatElem, 2>, String> {
    if cfg!(feature = "input") {
        let args: Vec<String> = std::env::args().collect();
        if args.len() < 3 {
            return Err("Please provide two JSON inputs for example: '[[1.0, 2.0], [3.0, 4.0]]' '[[1.0, 2.0], [3.0, 4.0]]'".to_string());
        }
        a = match serde_json::from_str(&args[1]) {
            Ok(a) => a,
            Err(e) => {
                return Err(format!(
                    "Please provide a valid JSON input for example: '[[1.0, 2.0], [3.0, 4.0]]': {}",
                    e
                ))
            }
        };
        b = match serde_json::from_str(&args[2]) {
            Ok(b) => b,
            Err(e) => {
                return Err(format!(
                    "Please provide a valid JSON input for example: '[[1.0, 2.0], [3.0, 4.0]]': {}",
                    e
                ))
            }
        };
    }
    let device = Default::default();

    let tensor1: Tensor<B, 2> = Tensor::from_floats(a, &device);
    let tensor2: Tensor<B, 2> = Tensor::from_floats(b, &device);

    let result = tensor1 + tensor2;

    Ok(result.into_data())
}

#[cfg(any(feature = "ndarray", feature = "cocos"))]
pub fn run() {
    let a = [[2., 3.], [4., 5.]];
    let b = [[std::f32::consts::PI, 1.], [1., std::f32::consts::PI]];
    match addition::<burn::backend::NdArray>(a, b) {
        Ok(result) => println!("{:}", result),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
}

#[cfg(feature = "wgpu")]
pub fn run() {
    let a = [[2., 3.], [4., 5.]];
    let b = [[std::f32::consts::PI, 1.], [1., std::f32::consts::PI]];
    match addition::<burn::backend::Wgpu>(a, b) {
        Ok(result) => println!("{:}", result),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
}

fn main() {
    #[cfg(any(feature = "ndarray", feature = "cocos"))]
    run();
    #[cfg(feature = "wgpu")]
    run();
}
