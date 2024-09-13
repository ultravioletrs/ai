use crate::model::{build_and_load_model, NDBackend};
use burn::tensor::Tensor;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct IrisItem {
    pub sepal_length: f32,
    pub sepal_width: f32,
    pub petal_length: f32,
    pub petal_width: f32,
}

pub async fn inference(iris_item: IrisItem) -> Result<&'static str, String> {
    let model = match build_and_load_model() {
        Ok(model) => Some(model),
        Err(e) => return Err(e),
    };

    let model = match model.as_ref() {
        Some(model) => model,
        None => return Err("Model not found".to_string()),
    };

    let device = Default::default();

    let input: [f32; 4] = [
        iris_item.sepal_length,
        iris_item.sepal_width,
        iris_item.petal_length,
        iris_item.petal_width,
    ];

    let input = Tensor::<NDBackend, 1>::from_floats(input, &device).unsqueeze();
    let output = model.forward(input);

    let output = burn::tensor::activation::softmax(output, 1);
    let max_index = output.argmax(1);

    let class: i64 = max_index.into_scalar();

    Ok(label_to_class(class))
}

fn label_to_class(label: i64) -> &'static str {
    match label {
        0 => "Iris-setosa",
        1 => "Iris-versicolor",
        2 => "Iris-virginica",
        _ => panic!("Invalid class"),
    }
}
