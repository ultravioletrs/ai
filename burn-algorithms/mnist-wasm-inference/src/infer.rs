use crate::model::{build_and_load_model, NDBackend};
use burn::tensor::Tensor;

pub async fn inference(input: &[f32]) -> i64 {
    let model = Some(build_and_load_model());

    let model = model.as_ref().unwrap();

    let device = Default::default();

    let input = Tensor::<NDBackend, 1>::from_floats(input, &device).reshape([1, 28, 28]);
    let output = model.forward(input);

    let output = burn::tensor::activation::softmax(output, 1);
    let max_index = output.argmax(1);

    #[cfg(not(target_family = "wasm"))]
    return max_index.into_scalar();

    #[cfg(target_family = "wasm")]
    return max_index.into_scalar().await;
}
