use futures::executor;
use mnist_inference::infer::inference;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <JSON input>", args[0]);
        std::process::exit(1);
    }
    let input: Vec<f32> = match serde_json::from_str(&args[1]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!(
                "Provide a valid JSON input for example: [0.0, 0.0, ..., 0.0]: {}",
                e
            );
            std::process::exit(1);
        }
    };
    let result = executor::block_on(inference(input.as_slice()));
    println!("{}", result);
}
