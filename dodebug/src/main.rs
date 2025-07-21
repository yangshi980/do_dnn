mod torch;
use env;
use std::time::{SystemTime, UNIX_EPOCH};
use tch::{IValue, Tensor};
fn main() {
    let args: Vec<String> = env::args().collect();

    let file = std::fs::File::open(&args[1]).unwrap();
    let model = torch::Model::load(file);
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();
    println!("start:{}", time);
    let mut sum: f32 = 0f32;
    for _ in 0..1000 {
        let input = tch::Tensor::randn([1024, 2048], (tch::Kind::Float, tch::Device::Cpu));
        let result: Vec<f32> = model.inference(IValue::Tensor(input));
        let tsum: f32 = result.iter().sum();
        sum += tsum;
    }
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        - time;
    println!("elapse:{}", time);
    println!("sum result:{}", sum);
}
