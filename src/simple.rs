use ndarray::{Array1, ArrayView1, ArrayView2, ArrayViewMut1, Axis, par_azip, s};

use super::find_max;

fn compute_haussdorf(
    points: ArrayView2<'_, f64>,
    norms: ArrayView1<'_, f64>,
    current: usize,
    output: ArrayViewMut1<f64>
) {
    let point = points.slice(s![current, ..]);
    par_azip!((o in output, norm in norms, other in points.axis_iter(Axis(0))) {
        *o = norm + norms[current] - 2.0 * point.dot(&other);
    })
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

    let norms = points.axis_iter(Axis(0))
        .map(|row| row.dot(&row))
        .collect::<Array1<f64>>();

    let mut fps_indexes = Vec::with_capacity(n_select);
    fps_indexes.push(initial);

    let mut haussdorf = Array1::from_elem([n_points], 0.0);
    compute_haussdorf(points, norms.view(), initial, haussdorf.view_mut());
    let (_, initial_max_value) = find_max(haussdorf.iter());
    let mut new_distances = Array1::from_elem([n_points], 0.0);

    for i in 1..n_select {
        let (new, max_value) = find_max(haussdorf.iter());
        if max_value / initial_max_value < 1e-12 {
            panic!(
                "unable to select more than {} points, all remaining \
                points are identical to already selected points",
                i
            );
        }

        fps_indexes.push(new);

        compute_haussdorf(points, norms.view(), new, new_distances.view_mut());

        par_azip!((d in &mut haussdorf, &new_d in &new_distances) {
            if new_d < *d {
                *d = new_d;
            }
        });
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
