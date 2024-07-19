use crate::data::{WineQualityBatcher, WineQualityDataset};
use crate::model::RegressionModelConfig;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::SgdConfig,
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::store::{Aggregate, Direction, Split},
        metric::LossMetric,
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

static ARTIFACT_DIR: &str = "artifacts/winequality/";

#[derive(Config)]
pub struct ExpConfig {
    pub optimizer: SgdConfig,

    #[config(default = 100)]
    pub num_epochs: usize,
    #[config(default = 5)]
    pub stop_after_n_epochs: usize,
    #[config(default = 128)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 11)]
    pub input_feature_len: usize,
    #[config(default = 5e-3)]
    pub learning_rate: f64,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let optimizer = SgdConfig::new();
    let config = ExpConfig::new(optimizer);
    let model = RegressionModelConfig::new(config.input_feature_len).init(&device);
    B::seed(config.seed);

    let train_dataset = WineQualityDataset::train();
    let test_dataset = WineQualityDataset::test();

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Test Dataset Size: {}", test_dataset.len());

    let batcher_train = WineQualityBatcher::<B>::new(device.clone());

    let batcher_test = WineQualityBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince {
                n_epochs: config.stop_after_n_epochs,
            },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), config.learning_rate);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config
        .save(format!("{ARTIFACT_DIR}/config.json").as_str())
        .expect("Failed to save config");

    model_trained
        .save_file(
            format!("{ARTIFACT_DIR}/model"),
            &NoStdTrainingRecorder::new(),
        )
        .expect("Failed to save trained model");
}
