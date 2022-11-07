use ndarray::Array2;

mod command;
pub mod io;
pub mod model;
mod train;

fn predict(filepath: String) {
    let dataset: Array2<f32> = io::load_dataset(filepath);
    println!("{}", dataset);
}

fn main() {
    let command: command::Command = command::parse_args();

    match command {
        command::Command::Train(filepath) => train::train(filepath),
        command::Command::Predict(filepath) => predict(filepath)
    }
}
