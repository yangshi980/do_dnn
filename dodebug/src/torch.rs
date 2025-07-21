use tch::{jit::IValue, CModule, Device};
pub struct Model {
    model: CModule,
}
impl Model {
    pub fn load<T: std::io::Read>(mut r: T) -> Model {
        let mut model = CModule::load_data_on_device(&mut r, Device::Cpu).unwrap();
        model.set_eval();
        Model { model }
    }

    pub fn inference(&self, w_iv: IValue) -> Vec<f32> {
        let _guard = tch::no_grad_guard();
        let iv = self.model.forward_is(&[w_iv]).unwrap();

        if let IValue::Tensor(tensor) = iv {
            return Vec::<f32>::try_from(tensor.view(-1)).unwrap();
        }
        Vec::default()
    }
}
