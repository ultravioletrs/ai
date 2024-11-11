use crate::model::{build_and_load_model, NDBackend};
use burn::tensor::Tensor;

pub async fn inference(input: &[f32]) -> Result<i64, String> {
    let model = Some(build_and_load_model());

    let model = match model.as_ref() {
        Some(model) => model,
        None => return Err("Failed to load model".to_string()),
    };

    let device = Default::default();

    let input = Tensor::<NDBackend, 1>::from_floats(input, &device).reshape([1, 28, 28]);
    let output = model.forward(input);

    let output = burn::tensor::activation::softmax(output, 1);
    let max_index = output.argmax(1);

    Ok(max_index.into_scalar())
}
