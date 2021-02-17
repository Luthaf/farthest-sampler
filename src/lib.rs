#![allow(clippy::needless_return, clippy::redundant_field_names)]

use soa_derive::{StructOfArray, soa_zip};

use ndarray::{ArrayView, Array1, ArrayView2, Axis, s};

/// Single Voronoï cell
#[derive(StructOfArray, Debug)]
#[soa_derive = "Debug"]
pub struct VoronoiCell {
    /// Position of the center of the cell
    center: Array1<f64>,
    /// Index of the cell center among all points
    center_idx: usize,
    /// Index of the point farthest from the center in this cell
    farthest: usize,
    /// Distance (squared) to the farthest point from the center in this cell
    radius2: f64,
}

/// Get both the maximal value in `values` and the position of this maximal
/// value
fn find_max<'a, I: Iterator<Item=&'a f64>>(values: I) -> (usize, f64) {
    values
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).expect("got NaN value"))
        .map(|(index, value)| (index, *value))
        .expect("got an empty slice")
}

#[allow(clippy::missing_safety_doc)]
#[no_mangle]
pub unsafe extern fn select_fps_voronoi(ptr: *const f64, ncols: usize, nrows: usize, n_select: usize, initial: usize, output: *mut usize) {
    let slice = std::slice::from_raw_parts(ptr, ncols * nrows);
    let array = ArrayView::from(slice).into_shape((ncols, nrows)).unwrap();

    let results = select_fps(array, n_select, initial);
    let output = std::slice::from_raw_parts_mut(output, n_select);
    output.copy_from_slice(&results);
}

/// Select `n_select` points from `points` using Farthest Points Sampling, and
/// return the indexes of selected points. The first point (already selected) is
/// the point at the `initial` index.
pub fn select_fps(points: ArrayView2<'_, f64>, n_select: usize, initial: usize) -> Vec<usize> {
    let n_points = points.nrows();

    if n_select > n_points {
        panic!("can not select more points than what we have")
    }

    let center = points.slice(s![initial, ..]);

    let norms = points.axis_iter(Axis(0)).map(|row| {
        row.iter().map(|v| v*v).sum()
    }).collect::<Array1<f64>>();
    let norm_first = norms[initial];

    // Haussdorf distance of all points to the selected points
    let mut haussdorf = &norms + norm_first - 2.0 * center.dot(&points.t());

    // List of already selected Voronoï cells
    let mut cells = VoronoiCellVec::with_capacity(n_select);
    let (farthest, radius2) = find_max(haussdorf.iter());
    cells.push(VoronoiCell {
        center_idx: initial,
        center: center.to_owned(),
        farthest: farthest,
        radius2: radius2,
    });

    // Assign points to cell, start with everyone in the initial cell (cell 0)
    let mut cell_for_point = vec![0; points.nrows()];

    let mut distance2_to_new_point = Vec::with_capacity(n_select);
    for i in 1..n_select {
        distance2_to_new_point.clear();

        // Find the maximum minimum (maxmin) distance and the corresponding
        // point. The maxmin point must be one of the farthest points from the
        // Voronoï decomposition, so we only have to look at the list of
        // existing cells to find it.

        let (cell, _) = find_max(cells.radius2.iter());
        let new_idx = cells.farthest[cell];

        let new_cell = VoronoiCell {
            center: points.slice(s![new_idx, ..]).to_owned(),
            center_idx: new_idx,
            radius2: 0.0, // this will be updated below,
            farthest: new_idx,
        };

        // now we find the "active" Voronoi cells, i.e. those that might change
        // due to the new selection.
        let mut active_cells = Vec::new();

        // we must compute distance of the new point to all the previous FPS.
        for (&center_idx, center) in soa_zip!(&cells, [center_idx, center]) {
            let d2 = norms[new_idx] + norms[center_idx] - 2.0 * new_cell.center.dot(&center.view());
            distance2_to_new_point.push(d2);
        }

        for (cell_idx, &radius2) in cells.radius2.iter().enumerate() {
            // triangle inequality (r > d / 2), squared
            if 0.25 * distance2_to_new_point[cell_idx] < radius2 {
                active_cells.push(cell_idx);
            }
        }

        for &cell_idx in &active_cells {
            cells.radius2[cell_idx] = 0.0;
            cells.farthest[cell_idx] = cells.center_idx[cell_idx];
        }

        cells.push(new_cell);

        // update the voronoï decomposition with the new cell
        for j in 0..n_points {
            let mut cell_idx = cell_for_point[j];

            if active_cells.contains(&cell_idx) {
                // Check if we can skip this check for point j. This is a
                // tighter bound on the distance, since ||x_j - x_new|| <
                // new_radius
                if 0.25 * distance2_to_new_point[cell_idx] < haussdorf[j] {
                    let d2_j = norms[new_idx] + norms[j] - 2.0 * cells.center[i].dot(&points.slice(s![j, ..]));

                    if d2_j < haussdorf[j] {
                        // We have to reassign point j to the new cell.
                        haussdorf[j] = d2_j;

                        cell_idx = i;
                        cell_for_point[j] = cell_idx;
                    }
                }

                // also update the voronoi radius/farthest point for the cell
                // containing the current point
                if haussdorf[j] > cells.radius2[cell_idx] {
                    cells.radius2[cell_idx] = haussdorf[j];
                    cells.farthest[cell_idx] = j;
                }
            }
        }
    }

    return cells.center_idx;
}

#[cfg(test)]
mod test {
    use super::*;

    use ndarray::Array2;

    use ndarray_npy::read_npy;
    use std::path::PathBuf;

    #[test]
    fn check() {

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
    fn check_on_boston() {
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
