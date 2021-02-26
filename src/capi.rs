//! Very basic C API to allow calling voronoi FPS from Python
#![allow(clippy::missing_safety_doc)]
use std::{os::raw::c_char, sync::Arc};
use parking_lot::Mutex;

use ndarray::ArrayView;

use once_cell::sync::Lazy;

use tracing_timing_graph::{SpanGraph, SpanTimingLayer};
use tracing_subscriber::{prelude::*, registry::Registry};

#[no_mangle]
pub unsafe extern fn select_fps_voronoi(ptr: *const f64, ncols: usize, nrows: usize, n_select: usize, initial: usize, output: *mut usize) {
    let slice = std::slice::from_raw_parts(ptr, ncols * nrows);
    let array = ArrayView::from(slice).into_shape((ncols, nrows)).unwrap();

    let results = crate::voronoi::select_fps(array, n_select, initial);
    let output = std::slice::from_raw_parts_mut(output, n_select);
    output.copy_from_slice(&results);
}

#[no_mangle]
pub unsafe extern fn select_fps_simple(ptr: *const f64, ncols: usize, nrows: usize, n_select: usize, initial: usize, output: *mut usize) {
    let slice = std::slice::from_raw_parts(ptr, ncols * nrows);
    let array = ArrayView::from(slice).into_shape((ncols, nrows)).unwrap();

    let results = crate::simple::select_fps(array, n_select, initial);
    let output = std::slice::from_raw_parts_mut(output, n_select);
    output.copy_from_slice(&results);
}

static TIMER_GRAPH: Lazy<Arc<Mutex<SpanGraph>>> = Lazy::new(|| {
    let span_timer = SpanTimingLayer::new();
    let graph = span_timer.graph();
    let subscriber = Registry::default().with(span_timer);
    tracing::subscriber::set_global_default(subscriber).unwrap();
    return graph;
});

#[no_mangle]
pub unsafe extern fn reset_tracing() {
    let mut graph = TIMER_GRAPH.lock();
    graph.clear();
}

#[no_mangle]
pub unsafe extern fn get_tracing_table(buffer: *mut c_char, len: usize) {
    let buffer = std::slice::from_raw_parts_mut(buffer as *mut u8, len);

    let table = TIMER_GRAPH.lock().as_table();
    let table = table.as_bytes();

    let copy_size = std::cmp::min(len - 1, table.len());
    buffer.fill(0);
    buffer[..copy_size].copy_from_slice(&table[..copy_size]);
    buffer[len - 1] = 0;
}
