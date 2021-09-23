use ndarray::{Array1, ArrayView2, Axis, CowArray, Ix2, par_azip, s};

use super::find_max;

#[derive(Debug)]
pub struct FpsSelector<'a> {
    /// Input points
    points: CowArray<'a, f64, Ix2>,
    /// Norm of the vector from origin for each points
    norms: Array1<f64>,
    /// Shortest distance for each point to already selected points
    haussdorf: Array1<f64>,
    /// indexes of selected points
    indexes: Vec<usize>,
    /// temporary array for distances computation
    new_distances: Array1<f64>,
}

impl<'a> FpsSelector<'a> {
    pub fn new(points: CowArray<'a, f64, Ix2>, initial: usize) -> FpsSelector<'a> {
        let norms = points.axis_iter(Axis(0))
            .map(|row| row.dot(&row))
            .collect::<Array1<f64>>();

        let n_points = points.nrows();
        let mut selector = FpsSelector {
            points: points,
            norms: norms,
            haussdorf: Array1::from_elem([n_points], 0.0),
            indexes: vec![initial],
            new_distances: Array1::from_elem([n_points], 0.0),
        };

        selector.compute_haussdorf(initial);
        selector.haussdorf = selector.new_distances.clone();
        return selector;
    }

    /// Allocate capacity for `additional` more selected points
    pub fn reserve(&mut self, additional: usize) {
        self.indexes.reserve(additional);
    }

    /// Add a new selected point as the center of a Voronoï cell
    #[cfg_attr(feature = "time-graph", time_graph::instrument(name = "FpsSelector::add_point"))]
    pub fn add_point(&mut self, new_point: usize) {
        assert!(new_point < self.points.nrows());
        self.indexes.push(new_point);

        self.compute_haussdorf(new_point);

        for (d, &new_d) in self.haussdorf.iter_mut().zip(&self.new_distances) {
            if new_d < *d {
                *d = new_d;
            };
        }
    }

    /// Get the potential next point, i.e. the point with highest Haussdorf distance
    #[cfg_attr(feature = "time-graph", time_graph::instrument(name = "FpsSelector::next_point"))]
    pub fn next_point(&self) -> (usize, f64) {
        return find_max(self.haussdorf.iter());
    }

    #[cfg_attr(feature = "time-graph", time_graph::instrument(name = "FpsSelector::compute_haussdorf"))]
    fn compute_haussdorf(&mut self, current: usize) {
        let point = self.points.slice(s![current, ..]);
        let output = &mut self.new_distances;
        let norms = &self.norms;
        let points = &self.points;

        par_azip!((o in output, norm in norms, other in points.axis_iter(Axis(0))) {
            *o = norm + norms[current] - 2.0 * point.dot(&other);
        })
    }
}

/// Select `n_select` points from `points` using Farthest Points Sampling, and
/// return the indexes of selected points. The first point (already selected) is
/// the point at the `initial` index.
#[cfg_attr(feature = "time-graph", time_graph::instrument)]
pub fn select_fps(points: ArrayView2<'_, f64>, n_select: usize, initial: usize) -> Vec<usize> {
    let n_points = points.nrows();

    if n_select > n_points {
        panic!("can not select more points than what we have")
    }

    let mut fps = FpsSelector::new(points.into(), initial);
    fps.reserve(n_select - 1);

    for _ in 1..n_select {
        let (new_point, _) = fps.next_point();
        fps.add_point(new_point);
    }

    return fps.indexes;
}


#[cfg(test)]
mod test {
    use super::*;

    use ndarray::Array2;

    use ndarray_npy::read_npy;
    use std::path::PathBuf;

    #[test]
    fn check_simple() {
        let data = Array2::from_shape_vec((4, 2), vec![
            0.0, 1.0,
            0.8, 0.5,
            0.0, 0.0,
            1.0, 0.0,
        ]).unwrap();

        let expected = vec![0, 3, 2, 1];
        for n_select in 1..expected.len() {
            let selected = select_fps(data.view(), n_select, expected[0]);
            assert_eq!(selected, expected[..n_select]);
        }
    }

    #[test]
    fn check_boston() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("boston.npy");
        let data: Array2<f64> = read_npy(path).unwrap();

        let expected = vec![9, 3, 11, 6, 1, 10, 8, 0, 12, 2, 5, 7, 4];
        for n_select in 1..13 {
            let selected = select_fps(data.t(), n_select, expected[0]);
            assert_eq!(selected, expected[..n_select]);
        }
    }
}
