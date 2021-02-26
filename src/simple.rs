use ndarray::{Array1, ArrayView2, Axis, s};

use super::find_max;

/// Select `n_select` points from `points` using Farthest Points Sampling, and
/// return the indexes of selected points. The first point (already selected) is
/// the point at the `initial` index.
#[tracing::instrument]
pub fn select_fps(points: ArrayView2<'_, f64>, n_select: usize, initial: usize) -> Vec<usize> {
    let n_points = points.nrows();

    if n_select > n_points {
        panic!("can not select more points than what we have")
    }

    let norms = points.axis_iter(Axis(0))
        .map(|row| row.dot(&row))
        .collect::<Array1<f64>>();

    let mut fps_indexes = Vec::with_capacity(n_select);
    fps_indexes.push(initial);

    let first_point = points.slice(s![initial, ..]);
    let mut haussdorf = &norms + norms[initial] - 2.0 * first_point.dot(&points.t());

    for _ in 1..n_select {
        let (new, _) = find_max(haussdorf.iter());
        fps_indexes.push(new);

        let new_point = points.slice(s![new, ..]);
        let new_distances = &norms + norms[new] - 2.0 * new_point.dot(&points.t());

        for (d, &new_d) in haussdorf.iter_mut().zip(&new_distances) {
            if new_d < *d {
                *d = new_d;
            };
        }
    }

    return fps_indexes;
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
