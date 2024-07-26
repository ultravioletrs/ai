#[cfg(any(feature = "ndarray", feature = "cocos"))]
use std::{io::Write, os::unix::net::UnixStream};

use burn::{
    prelude::*,
    tensor::{backend::Backend, Tensor},
};

pub fn addition<B: Backend>(
    a: [[f32; 2]; 2],
    b: [[f32; 2]; 2],
) -> Result<Data<<B as Backend>::FloatElem, 2>, String> {
    let device = Default::default();

    let tensor1: Tensor<B, 2> = Tensor::from_floats(a, &device);
    let tensor2: Tensor<B, 2> = Tensor::from_floats(b, &device);

    let result = tensor1 + tensor2;

    Ok(result.into_data())
}

#[cfg(any(feature = "ndarray", feature = "cocos"))]
pub fn send_data_via_socket(result: Data<f32, 2>) -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        return Err("Provide unix socket path".to_string());
    }

    let mut stream = match UnixStream::connect(&args[1]) {
        Ok(stream) => stream,
        Err(e) => return Err(e.to_string()),
    };

    let data = result.to_string();
    match stream.write_all(data.as_bytes()) {
        Ok(_) => (),
        Err(e) => return Err(e.to_string()),
    };

    Ok(())
}

#[cfg(any(feature = "ndarray", feature = "cocos"))]
pub fn run() {
    let a = [[2., 3.], [4., 5.]];
    let b = [[std::f32::consts::PI, 1.], [1., std::f32::consts::PI]];
    match addition::<burn::backend::NdArray>(a, b) {
        Ok(result) => {
            if cfg!(feature = "cocos") {
                match send_data_via_socket(result) {
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
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };
}

#[cfg(feature = "wgpu")]
pub fn run_wgpu() {
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

#[cfg(feature = "read")]
pub fn read_binary_results() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: read-results <results-binary-file>");
        std::process::exit(1);
    }
    let results = std::fs::read(&args[1]).expect("Failed to read file");
    println!("{:?}", String::from_utf8_lossy(&results));
}

fn main() {
    #[cfg(any(feature = "ndarray", feature = "cocos"))]
    run();
    #[cfg(feature = "wgpu")]
    run_wgpu();
    #[cfg(feature = "read")]
    read_binary_results();
}
