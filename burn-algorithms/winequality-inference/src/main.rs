use futures::executor;
use winequality_inference::infer::{inference, WineQualityItem};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <JSON input>", args[0]);
        std::process::exit(1);
    }
    let deserialized: WineQualityItem = match serde_json::from_str(&args[1]) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Invalid JSON input: {}. Provide a valid JSON input for example: {{\"fixed_acidity\": 7,\"volatile_acidity\": 0.27,\"citric_acid\": 0.36,\"residual_sugar\": 20.7,\"chlorides\": 0.045,\"free_sulfur_dioxide\": 45,\"total_sulfur_dioxide\": 170,\"density\": 1.001,\"ph\": 3,\"sulphates\": 0.45,\"alcohol\": 8.8}}", e);
            std::process::exit(1);
        }
    };
    match executor::block_on(inference(deserialized)) {
        Ok(result) => println!("{}", result),
        Err(e) => eprintln!("{}", e),
    };
}
