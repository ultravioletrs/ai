use crate::model::{build_and_load_model, NDBackend};
use burn::tensor::Tensor;
use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct WineQualityItem {
    pub fixed_acidity: f64,
    pub volatile_acidity: f64,
    pub citric_acid: f64,
    pub residual_sugar: f64,
    pub chlorides: f64,
    pub free_sulfur_dioxide: f64,
    pub total_sulfur_dioxide: f64,
    pub density: f64,
    pub ph: f64,
    pub sulphates: f64,
    pub alcohol: f64,
}

pub async fn inference(wine_quality_item: WineQualityItem) -> Result<f32, String> {
    let model = match build_and_load_model() {
        Ok(model) => Some(model),
        Err(e) => return Err(e),
    };

    let model = match model.as_ref() {
        Some(model) => model,
        None => return Err("Model not found".to_string()),
    };

    let device = Default::default();

    // The constants used to normalize the input data are the same as the ones used to train the model
    // They are the minimum and maximum values of each feature in the training dataset
    let input: [f32; 11] = [
        (wine_quality_item.fixed_acidity as f32 - 3.8) / (14.2 - 3.8),
        (wine_quality_item.volatile_acidity as f32 - 0.08) / (1.1 - 0.08),
        (wine_quality_item.citric_acid as f32 - 0.0) / (1.66 - 0.0),
        (wine_quality_item.residual_sugar as f32 - 0.6) / (65.8 - 0.6),
        (wine_quality_item.chlorides as f32 - 0.009) / (0.346 - 0.009),
        (wine_quality_item.free_sulfur_dioxide as f32 - 2.0) / (289.0 - 2.0),
        (wine_quality_item.total_sulfur_dioxide as f32 - 9.0) / (440.0 - 9.0),
        (wine_quality_item.density as f32 - 0.98711) / (1.03898 - 0.98711),
        (wine_quality_item.ph as f32 - 2.72) / (3.82 - 2.72),
        (wine_quality_item.sulphates as f32 - 0.22) / (1.08 - 0.22),
        (wine_quality_item.alcohol as f32 - 8.0) / (14.2 - 8.0),
    ];

    let input = Tensor::<NDBackend, 1>::from_floats(input, &device).unsqueeze();
    let output = model.forward(input);

    #[cfg(not(target_family = "wasm"))]
    let result = output.into_scalar();

    #[cfg(target_family = "wasm")]
    let result = output.into_scalar().await;

    Ok(result)
}
