#![allow(clippy::needless_return, clippy::redundant_field_names)]

#[cfg(features = "time-graph")]
macro_rules! tracing_span {
    ($name: literal, $code: tt) => {
        {
            let span = tracing::info_span!($name);
            let _guard = span.enter();
            $code
        }
    };
}

#[cfg(not(features = "time-graph"))]
macro_rules! tracing_span {
    ($name: literal, $code: tt) => {
        {
            $code
        }
    };
}

/// Get both the maximal value in `values` and the position of this maximal
/// value
pub fn find_max<'a, I: Iterator<Item=&'a f64>>(values: I) -> (usize, f64) {
    values
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("got NaN value"))
        .map(|(index, value)| (index, *value))
        .expect("got an empty slice")
}

pub mod simple;

pub mod voronoi;
pub use voronoi::VoronoiDecomposer;

#[cfg(feature = "python")]
mod python;
