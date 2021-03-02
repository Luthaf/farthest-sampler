use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::path::PathBuf;
use ndarray::Array2;
use ndarray_npy::read_npy;

use voronoi_fps::voronoi::select_fps;

fn voronoi_fps_boston(c: &mut Criterion) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("boston.npy");
    let data: Array2<f64> = read_npy(path).unwrap();
    let data = data.t().as_standard_layout().to_owned();

    c.bench_function("Voronoi FPS on Boston dataset", |b| b.iter(|| {
        select_fps(data.view(), black_box(10), black_box(9));
    }));
}

fn voronoi_fps_soap(c: &mut Criterion) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("soap.npy");
    let data: Array2<f64> = read_npy(path).unwrap();
    let data = data.t().as_standard_layout().to_owned();

    c.bench_function("Voronoi FPS on SOAP", |b| b.iter(|| {
        select_fps(data.view(), black_box(100), black_box(0));
    }));
}


criterion_group!(
    name = fps;
    config = Criterion::default().measurement_time(std::time::Duration::from_secs(10));
    targets = voronoi_fps_boston, voronoi_fps_soap
);
criterion_main!(fps);
