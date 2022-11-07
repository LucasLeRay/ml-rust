use std::fmt;

use ndarray::{Array1, Array2, Axis};

fn step_intercept(alpha: f32, cost: &Array1<f32>) -> f32 {
    alpha * cost.mean().unwrap()
}

fn step_coefficient(alpha: f32, cost: &Array1<f32>, feature: &Array1<f32>) -> f32 {
    alpha * (cost * feature).mean().unwrap()
}

#[derive(Debug)]
pub struct Model {
    pub intercept: f32,
    pub coefficients: Array1<f32>,
}

impl Model {
    pub fn predict(&self, features: &Array2<f32>) -> Array1<f32> {
        let mut predictions: Array1<f32> = Array1::from_elem(features.len(), self.intercept);
        for (feature_col, coeff) in features.axis_iter(Axis(1)).zip(self.coefficients.iter()) {
            for (prediction, feature) in predictions.iter_mut().zip(feature_col.iter()) {
                *prediction += feature * coeff;
            }
        }
        predictions
    }

    pub fn gradient_descent(&mut self, errors: &Array1<f32>, features: &Array2<f32>, alpha: f32) {
        self.intercept -= step_intercept(alpha, &errors);

        for (i, coeff) in self.coefficients.iter_mut().enumerate() {
            *coeff -= step_coefficient(alpha, &errors, &features.column(i).to_owned());
        }
    }

    pub fn new(n_features: usize) -> Self {
        Model {
            intercept: 0.0,
            coefficients: Array1::zeros(n_features)
        }
    }
}

impl fmt::Display for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(intercept: {}, coeffs: {})", self.intercept, self.coefficients)
    }
}
