#![allow(clippy::needless_return)]

use criterion::{criterion_group, criterion_main};

mod samples {
    use criterion::Criterion;
    use std::path::PathBuf;
    use ndarray::Array2;

    fn load_soap() -> Array2<f64> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("soap.npy");
        return ndarray_npy::read_npy(path).unwrap();
    }

    fn load_boston() -> Array2<f64> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("boston.npy");
        return ndarray_npy::read_npy(path).unwrap();
    }

    pub fn standard_boston(c: &mut Criterion) {
        let data = load_boston();
        c.bench_function("Standard FPS/100 samples/Boston dataset", |b| b.iter(|| {
            farthest_sampler::simple::select_fps(data.view(), 100, 0);
        }));
        c.bench_function("Standard FPS/200 samples/Boston dataset", |b| b.iter(|| {
            farthest_sampler::simple::select_fps(data.view(), 200, 0);
        }));
    }

    pub fn standard_soap(c: &mut Criterion) {
        let data = load_soap();
        c.bench_function("Standard FPS/100 samples/SOAP", |b| b.iter(|| {
            farthest_sampler::simple::select_fps(data.view(), 100, 0);
        }));
        c.bench_function("Standard FPS/200 samples/SOAP", |b| b.iter(|| {
            farthest_sampler::simple::select_fps(data.view(), 200, 0);
        }));
    }

    pub fn voronoi_boston(c: &mut Criterion) {
        let data = load_boston();
        c.bench_function("Voronoi FPS/100 samples/Boston dataset", |b| b.iter(|| {
            farthest_sampler::voronoi::select_fps(data.view(), 100, 0);
        }));
        c.bench_function("Voronoi FPS/200 samples/Boston dataset", |b| b.iter(|| {
            farthest_sampler::voronoi::select_fps(data.view(), 200, 0);
        }));
    }

    pub fn voronoi_soap(c: &mut Criterion) {
        let data = load_soap();
        c.bench_function("Voronoi FPS/100 samples/SOAP", |b| b.iter(|| {
            farthest_sampler::voronoi::select_fps(data.view(), 100, 0);
        }));
        c.bench_function("Voronoi FPS/200 samples/SOAP", |b| b.iter(|| {
            farthest_sampler::voronoi::select_fps(data.view(), 200, 0);
        }));
    }
}

mod features {
    use criterion::Criterion;
    use std::path::PathBuf;
    use ndarray::Array2;

    fn load_soap() -> Array2<f64> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("soap.npy");
        let data: Array2<f64> = ndarray_npy::read_npy(path).unwrap();
        return data.t().as_standard_layout().to_owned();
    }

    fn load_boston() -> Array2<f64> {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("boston.npy");
        let data: Array2<f64> = ndarray_npy::read_npy(path).unwrap();
        return data.t().as_standard_layout().to_owned();
    }

    pub fn standard_boston(c: &mut Criterion) {
        let data = load_boston();
        c.bench_function("Standard FPS/10 features/Boston dataset", |b| b.iter(|| {
            farthest_sampler::simple::select_fps(data.view(), 10, 0);
        }));
    }

    pub fn standard_soap(c: &mut Criterion) {
        let data = load_soap();
        c.bench_function("Standard FPS/100 features/SOAP", |b| b.iter(|| {
            farthest_sampler::simple::select_fps(data.view(), 100, 0);
        }));
        c.bench_function("Standard FPS/200 features/SOAP", |b| b.iter(|| {
            farthest_sampler::simple::select_fps(data.view(), 200, 0);
        }));
    }

    pub fn voronoi_boston(c: &mut Criterion) {
        let data = load_boston();
        c.bench_function("Voronoi FPS/10 features/Boston dataset", |b| b.iter(|| {
            farthest_sampler::voronoi::select_fps(data.view(), 10, 0);
        }));
    }

    pub fn voronoi_soap(c: &mut Criterion) {
        let data = load_soap();
        c.bench_function("Voronoi FPS/100 features/SOAP", |b| b.iter(|| {
            farthest_sampler::voronoi::select_fps(data.view(), 100, 0);
        }));
        c.bench_function("Voronoi FPS/200 features/SOAP", |b| b.iter(|| {
            farthest_sampler::voronoi::select_fps(data.view(), 200, 0);
        }));
    }
}

criterion_group!(
    samples,
    samples::standard_boston, samples::standard_soap, samples::voronoi_boston, samples::voronoi_soap
);

criterion_group!(
    features,
    features::standard_boston, features::standard_soap, features::voronoi_boston, features::voronoi_soap
);

criterion_main!(samples, features);
