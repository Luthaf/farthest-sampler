#![allow(clippy::needless_lifetimes)]

use pyo3::prelude::*;

use std::cell::RefCell;

use ndarray::ArrayView2;
use numpy::{PyArray1, PyArray2};

#[pyclass]
pub struct VoronoiDecomposer {
    decomposer: RefCell<crate::VoronoiDecomposer<'static>>,
}

#[pymethods]
impl VoronoiDecomposer {
    #[new]
    fn new(points: &PyArray2<f64>, initial: usize) -> Self {
        let points = points.readonly();

        let points: ArrayView2<f64> = points.as_array();

        // SAFETY: nothing, but we REALLY don't want to make a copy of this data
        let points: ArrayView2<'static, f64> = unsafe {
            std::mem::transmute(points)
        };

        VoronoiDecomposer {
            decomposer: RefCell::new(crate::VoronoiDecomposer::new(points.into(), initial)),
        }
    }

    fn add_point(&self, new_point: usize) -> f64 {
        let mut decomposer = self.decomposer.borrow_mut();
        decomposer.add_point(new_point);
        return *decomposer.cells().last().unwrap().radius2;
    }

    fn next_point(&self) -> (usize, f64) {
        let decomposer = self.decomposer.borrow();
        return decomposer.next_point();
    }

    fn radius2<'a>(&self, py: Python<'a>) -> &'a PyArray1<f64> {
        let decomposer = self.decomposer.borrow();
        let r2 = decomposer.cells().radius2;

        return PyArray1::from_slice(py, r2);
    }
}

#[pyclass]
pub struct FpsSelector {
    selector: RefCell<crate::FpsSelector<'static>>,
}

#[pymethods]
impl FpsSelector {
    #[new]
    fn new(points: &PyArray2<f64>, initial: usize) -> Self {
        let points = points.readonly();

        let points: ArrayView2<f64> = points.as_array();

        // SAFETY: nothing, but we REALLY don't want to make a copy of this data
        let points: ArrayView2<'static, f64> = unsafe {
            std::mem::transmute(points)
        };

        FpsSelector {
            selector: RefCell::new(crate::FpsSelector::new(points.into(), initial)),
        }
    }

    fn add_point(&self, new_point: usize) {
        let mut selector = self.selector.borrow_mut();
        selector.add_point(new_point);
    }

    fn next_point(&self) -> (usize, f64) {
        let selector = self.selector.borrow();
        return selector.next_point();
    }
}


#[pymodule]
fn farthest_sampler(_: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m, "select_fps_voronoi")]
    fn select_fps_voronoi<'py>(py: Python<'py>, points: &PyArray2<f64>, n_select: usize, initial: usize) -> &'py PyArray1<usize> {
        let points = points.readonly();
        let selected = crate::voronoi::select_fps(points.as_array(), n_select, initial);
        return PyArray1::from_vec(py, selected);
    }

    #[pyfn(m, "select_fps_standard")]
    fn select_fps_standard<'py>(py: Python<'py>, points: &PyArray2<f64>, n_select: usize, initial: usize) -> &'py PyArray1<usize> {
        let points = points.readonly();
        let selected = crate::simple::select_fps(points.as_array(), n_select, initial);
        return PyArray1::from_vec(py, selected);
    }

    time_graph::enable_data_collection(true);

    #[pyfn(m, "clear_profiling")]
    fn clear_profiling<'py>(_py: Python<'py>) {
        time_graph::clear_collected_data();
    }

    #[pyfn(m, "get_profiling")]
    fn get_profiling<'py>(_py: Python<'py>) -> String {
        time_graph::get_full_graph().as_short_table()
    }

    m.add_class::<VoronoiDecomposer>()?;
    m.add_class::<FpsSelector>()?;
    Ok(())
}
