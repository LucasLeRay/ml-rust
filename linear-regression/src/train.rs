use ndarray::{Array1, Array2, Axis};

use crate::io;
use crate::model;

const EPOCHS: usize = 1000000;
const LEARNING_RATE: f32 = 0.00001;

fn split_dataset(dataset: Array2<f32>) -> (Array2<f32>, Array1<f32>) {
    let n_features: usize = dataset.len_of(Axis(1)) - 1;

    let features: Array2<f32> = dataset.view().split_at(Axis(1), n_features).0.to_owned();
    let targets: Array1<f32> = dataset.column(n_features).to_owned();

    (features, targets)
}

pub fn train(filepath: String) {
    let dataset: Array2<f32> = io::load_dataset(filepath);
    let (features, targets): (Array2<f32>, Array1<f32>) = split_dataset(dataset);

    let n_features: usize = features.len_of(Axis(1));
    let mut model: model::Model = model::Model::new(n_features);

    for epoch in 0..EPOCHS {
        let predictions: Array1<f32> = model.predict(&features);
        let errors: Array1<f32> = &predictions - &targets;
        
        model.gradient_descent(&errors, &features, LEARNING_RATE);
        
        let mean_error: f32 = errors.mean().unwrap().abs();
        if epoch % 1000 == 0 {
            println!("mean error: {}",mean_error);
            println!("new model: {}", model);
        }

        println!("mean error: {}", mean_error);
    }
    println!("final model: {:?}", model);
}
