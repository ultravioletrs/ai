use crate::data::{IrisBatcher, IrisDataset};
use crate::model::ClassificationModelConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::train::metric::AccuracyMetric;
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, NoStdTrainingRecorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::store::{Aggregate, Direction, Split},
        metric::LossMetric,
        LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
    },
};

static ARTIFACT_DIR: &str = "iris/artifacts/";

#[derive(Config)]
pub struct ExpConfig {
    #[config(default = 100)]
    pub num_epochs: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    pub optimizer: AdamConfig,

    #[config(default = 4)]
    pub input_feature_len: usize,

    #[config(default = 128)]
    pub hidden_size: usize,
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let optimizer = AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(5e-5)));
    let config = ExpConfig::new(optimizer);
    let model =
        ClassificationModelConfig::new(config.input_feature_len, config.hidden_size).init(&device);
    B::seed(config.seed);

    let train_dataset = IrisDataset::train();
    let test_dataset = IrisDataset::test();

    println!("Train Dataset Size: {}", train_dataset.len());
    println!("Test Dataset Size: {}", test_dataset.len());

    let batcher_train = IrisBatcher::<B>::new(device.clone());

    let batcher_test = IrisBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(train_dataset.len())
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(test_dataset.len())
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(test_dataset);

    let learner = LearnerBuilder::new(ARTIFACT_DIR)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new::<LossMetric<B>>(
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 10 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, config.optimizer.init(), 5e-3);

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
