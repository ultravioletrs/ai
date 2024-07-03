use futures::executor;
use mnist_wasm_inference::infer::inference;
use serde_json;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input: Vec<f32> = serde_json::from_str(&args[1]).unwrap();
    let result = executor::block_on(inference(input.as_slice()));
    println!("{}", result);
}
