use std::env;

#[derive(Debug)]
pub enum Command {
    Train(String),
    Predict(String)
}

pub fn parse_args() -> Command {
    let args: Vec<String> = env::args().collect();

    let action_idx: usize = 1;
    let filepath_idx: usize = 2;

    match args[action_idx].as_str() {
        "train" => Command::Train(args[filepath_idx].to_string()),
        "predict" => Command::Predict(args[filepath_idx].to_string()),
        _ => panic!("Invalid action!")
    }
}
