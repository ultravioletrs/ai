use futures::executor;
use winequality_inference::infer::{inference, WineQualityItem};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let deserialized: WineQualityItem = serde_json::from_str(&args[1]).expect("Please provide a valid JSON input for example: {\"fixed_acidity\": 7,\"volatile_acidity\": 0.27,\"citric_acid\": 0.36,\"residual_sugar\": 20.7,\"chlorides\": 0.045,\"free_sulfur_dioxide\": 45,\"total_sulfur_dioxide\": 170,\"density\": 1.001,\"ph\": 3,\"sulphates\": 0.45,\"alcohol\": 8.8}");
    match executor::block_on(inference(deserialized)) {
        Ok(result) => println!("{}", result),
        Err(e) => eprintln!("{}", e),
    };
}
