use crate::{
    data::{BertCasedTokenizer, ClassificationBatcher, ClassificationDataset, Tokenizer},
    model::ClassificationModelConfig,
};
use burn::{
    data::{dataloader::DataLoaderBuilder, dataset::transform::SamplerDataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

#[derive(Config)]
pub struct ExperimentConfig {
    pub transformer: TransformerEncoderConfig,
    pub optimizer: AdamConfig,
    #[config(default = 512)]
    pub max_seq_length: usize,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 5)]
    pub num_epochs: usize,
}

pub fn train<B: AutodiffBackend, D: ClassificationDataset + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(BertCasedTokenizer::default());

    let batcher_train =
        ClassificationBatcher::<B>::new(tokenizer.clone(), device.clone(), config.max_seq_length);
    let batcher_test = ClassificationBatcher::<B::InnerBackend>::new(
        tokenizer.clone(),
        device.clone(),
        config.max_seq_length,
    );

    let model = ClassificationModelConfig::new(
        config.transformer.clone(),
        D::num_classes(),
        tokenizer.vocab_size(),
        config.max_seq_length,
    )
    .init(&device);

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_train, 50_000));
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(SamplerDataset::new(dataset_test, 5_000));

    let optim = config.optimizer.init();

    let lr_scheduler = NoamLrSchedulerConfig::new(1e-2)
        .with_warmup_steps(1000)
        .with_model_size(config.transformer.d_model)
        .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(model, optim, lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(format!("{artifact_dir}/config.json")).unwrap();
    CompactRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
