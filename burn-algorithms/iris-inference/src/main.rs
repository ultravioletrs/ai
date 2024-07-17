use futures::executor;
use iris_inference::infer::{inference, IrisItem};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let deserialized: IrisItem = serde_json::from_str(&args[1]).expect("Please provide a valid JSON input for example: {\"sepal_length\": 5.1, \"sepal_width\": 3.5, \"petal_length\": 1.4, \"petal_width\": 0.2}");
    match executor::block_on(inference(deserialized)) {
        Ok(result) => println!("{}", result),
        Err(e) => eprintln!("{}", e),
    };
}
