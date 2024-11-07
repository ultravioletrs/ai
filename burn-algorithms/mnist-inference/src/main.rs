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
                "Invalid JSON input: {}. Provide a valid JSON input for example: [0.0, 0.0, ..., 0.0]",
                e
            );
            std::process::exit(1);
        }
    };
    match executor::block_on(inference(input.as_slice())) {
        Ok(result) => {
            if cfg!(feature = "cocos") {
                match lib::save_results_to_file(
                    result.to_string(),
                    "results/results.txt".to_string(),
                ) {
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
        Err(e) => eprintln!("{}", e),
    };
}
