#[cfg(not(feature = "wgpu"))]
mod ndarray {
    use burn::backend::{
        ndarray::{NdArray, NdArrayDevice},
        Autodiff,
    };
    use iris::training;

    pub fn run(csv_file_path: &str) {
        let device = NdArrayDevice::Cpu;
        training::run::<Autodiff<NdArray>>(device, csv_file_path);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::{
        wgpu::{Wgpu, WgpuDevice},
        Autodiff,
    };
    use iris::training;

    pub fn run(csv_file_path: &str) {
        let device = WgpuDevice::default();
        training::run::<Autodiff<Wgpu>>(device, csv_file_path);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: iris <csv-file-path>");
        std::process::exit(1);
    }
    #[cfg(not(feature = "wgpu"))]
    ndarray::run(&args[1]);
    #[cfg(feature = "wgpu")]
    wgpu::run(&args[1]);
}
