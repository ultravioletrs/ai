use futures::executor;
use mnist_inference::infer::inference;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input: Vec<f32> = serde_json::from_str(&args[1])
        .expect("Please provide a valid JSON input for example: [0.0, 0.0, ..., 0.0]");
    let result = executor::block_on(inference(input.as_slice()));
    println!("{}", result);
}
