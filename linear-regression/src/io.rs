use csv::Reader;
use std::fs::File;

use ndarray::{Array2, Axis};

fn build_vec(mut reader: Reader<File>) -> Vec<Vec<f32>> {
    let mut vec: Vec<Vec<f32>> = Vec::new();

    for row in reader.records() {
        let res: Vec<f32> = row.unwrap().deserialize(None).unwrap();
        vec.push(res);
    };
    vec
}

fn data_from_vec(vec: Vec<Vec<f32>>) -> Array2<f32> {
    let mut data: Array2<f32> = Array2::<f32>::default((vec.len(), vec[0].len()));

    for (i, mut row) in data.axis_iter_mut(Axis(0)).enumerate() {
        for (j, col) in row.iter_mut().enumerate() {
            *col = vec[i][j];
        }
    };

    data
}

pub fn load_dataset(filepath: String) -> Array2<f32> {
    let reader = csv::Reader::from_path(filepath).unwrap();

    let vec: Vec<Vec<f32>> = build_vec(reader);
    data_from_vec(vec)
}
