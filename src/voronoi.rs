use std::cell::Cell;

use rayon::prelude::*;
use thread_local::ThreadLocal;

use soa_derive::{StructOfArray, soa_zip};
use ndarray::{Array1, ArrayView2, Axis, s};

use super::find_max;

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

/// Point farthest away from a the center of a cell
#[derive(Clone, Copy, Debug, PartialEq, Default)]
struct FarthestPoint {
    pub distance2: f64,
    pub index: usize,
}

/// allocation cache for `VoronoiDecomposer` when adding a new point
#[derive(Debug)]
struct WorkArrays {
    /// Distance between the new point and the center of all cells
    distance_to_new_point: Vec<f64>,
    /// List of active cells that might need to change
    active_cells: Vec<usize>,
    /// New radius2 for all active cells. This needs to be thread local to
    /// prevent data races on radius access in the loop below.
    farthest_points: ThreadLocal<Vec<Cell<FarthestPoint>>>,
}

impl WorkArrays {
    fn new() -> WorkArrays {
        WorkArrays {
            distance_to_new_point: Vec::new(),
            active_cells: Vec::new(),
            farthest_points: ThreadLocal::new(),
        }
    }

    fn clear(&mut self) {
        self.distance_to_new_point.clear();
        self.active_cells.clear();
        self.farthest_points = ThreadLocal::new();
    }

    fn reserve(&mut self, additional: usize) {
        self.distance_to_new_point.reserve(additional);
        self.active_cells.reserve(additional);
    }
}

#[derive(Debug)]
pub struct VoronoiDecomposer<'a> {
    /// Input points
    points: ArrayView2<'a, f64>,
    /// Current list of cells
    cells: VoronoiCellVec,
    /// For each point, index of the cell in `cells` containing the point
    cell_for_point: Vec<usize>,
    /// Norm of the vectors
    norms: Vec<f64>,
    /// Shortest distance for each point to already selected points
    haussdorf: Vec<f64>,
    /// Cached allocations when adding new points
    work: WorkArrays,
}

impl<'a> VoronoiDecomposer<'a> {
    #[tracing::instrument(name = "initialize voronoi")]
    pub fn new(points: ArrayView2<'a, f64>, initial: usize) -> VoronoiDecomposer<'a> {
        let norms = points.axis_iter(Axis(0))
            .map(|row| row.dot(&row))
            .collect::<Array1<f64>>();
        let center = points.slice(s![initial, ..]);
        let haussdorf = &norms + norms[initial] - 2.0 * center.dot(&points.t());

        let mut cells = VoronoiCellVec::new();
        let (farthest, radius2) = find_max(haussdorf.iter());
        cells.push(VoronoiCell {
            center_idx: initial,
            center: center.to_owned(),
            farthest: farthest,
            radius2: radius2,
        });

        VoronoiDecomposer {
            points: points,
            cells: cells,
            // start with all points in the initial cell (cell 0)
            cell_for_point: vec![0; points.nrows()],
            norms: norms.to_vec(),
            haussdorf: haussdorf.to_vec(),
            work: WorkArrays::new(),
        }
    }

    /// Allocate capacity for `additional` more cells/selected points
    pub fn reserve(&mut self, additional: usize) {
        self.cells.reserve(additional);
        self.cell_for_point.reserve(additional);
        self.work.reserve(additional);
    }

    /// Add a new selected point as the center of a Voronoï cell
    #[tracing::instrument(name = "add new voronoi cell")]
    pub fn add_point(&mut self, new_point: usize) {
        let new_center = self.points.slice(s![new_point, ..]);
        let new_cell_idx = self.cells.len();

        self.work.clear();

        tracing_span!("find active cells", {
            // now we find the "active" Voronoi cells, i.e. those that might change
            // due to the new selection. We must compute distance of the new point
            // to all the previous FPS.
            for (&center_idx, center) in soa_zip!(&self.cells, [center_idx, center]) {
                let d2 = self.norms[new_point] + self.norms[center_idx] - 2.0 * new_center.dot(&center.view());
                self.work.distance_to_new_point.push(d2);
            }

            for (cell_idx, &radius2) in self.cells.radius2.iter().enumerate() {
                // triangle inequality (r > d / 2), squared
                if 0.25 * self.work.distance_to_new_point[cell_idx] < radius2 {
                    self.work.active_cells.push(cell_idx);
                }
            }

            // the new cell is always active
            self.work.active_cells.push(new_cell_idx);
            // ensure the cells are sorted to be able to use binary_search
            self.work.active_cells.sort_unstable();

            self.cells.push(VoronoiCell {
                center: new_center.to_owned(),
                center_idx: new_point,
                radius2: 0.0, // this will be updated below,
                farthest: new_point,
            });

            for &cell_idx in &self.work.active_cells {
                self.cells.radius2[cell_idx] = 0.0;
                self.cells.farthest[cell_idx] = self.cells.center_idx[cell_idx];
            }
        });

        let points = &self.points;
        let norms = &self.norms;
        let work = &mut self.work;
        let n_active_cells = work.active_cells.len();

        tracing_span!("update decomposition", {
            // update the voronoï decomposition with the new cell
            self.cell_for_point.par_iter_mut()
            .zip_eq(&mut self.haussdorf)
            .enumerate()
            .map(|(j, (cell_idx, haussdorf))| (j, cell_idx, haussdorf))
            // process 1000 points at the time to reduce threading overhead
            .chunks(1000)
            .for_each(|chunk| {
                for (j, cell_idx, haussdorf) in chunk {
                    let active_cells_idx = work.active_cells.binary_search(cell_idx);

                    if let Ok(mut active_cells_idx) = active_cells_idx {
                        // Check if we can skip this check for point j. This is a
                        // tighter bound on the distance, since ||x_j - x_new|| <
                        // new_radius
                        if 0.25 * work.distance_to_new_point[*cell_idx] < *haussdorf {
                            let d2_j = norms[new_point] + norms[j] - 2.0 * new_center.dot(&points.slice(s![j, ..]));
                            if d2_j < *haussdorf {
                                // We have to reassign point j to the new cell.
                                *haussdorf = d2_j;
                                *cell_idx = new_cell_idx;
                                active_cells_idx = n_active_cells - 1;
                            }
                        }

                        let farthest = work.farthest_points.get_or(|| {
                            vec![Cell::new(Default::default()); n_active_cells]
                        });

                        // also update the voronoi radius/farthest point for the cell
                        // containing the current point
                        if *haussdorf > farthest[active_cells_idx].get().distance2 {
                            farthest[active_cells_idx].set(FarthestPoint {
                                distance2: *haussdorf,
                                index: j,
                            });
                        }
                    }
                }
            });

            // merge the thread locals and update everything
            for farthests in work.farthest_points.iter_mut() {
                for (farthest, &cell_idx) in farthests.iter().zip(&work.active_cells) {
                    let farthest = farthest.get();
                    if farthest.distance2 > self.cells.radius2[cell_idx] {
                        self.cells.radius2[cell_idx] = farthest.distance2;
                        self.cells.farthest[cell_idx] = farthest.index;
                    }
                }
            }
        });
    }

    /// Access the current list of cells
    pub fn cells(&self) -> VoronoiCellSlice {
        self.cells.as_slice()
    }
}

/// Select `n_select` points from `points` using Farthest Points Sampling, and
/// return the indexes of selected points. The first point (already selected) is
/// the point at the `initial` index.
#[tracing::instrument]
pub fn select_fps(points: ArrayView2<'_, f64>, n_select: usize, initial: usize) -> Vec<usize> {
    let n_points = points.nrows();

    if n_select > n_points {
        panic!("can not select more points than what we have")
    }

    let mut voronoi = VoronoiDecomposer::new(points, initial);
    voronoi.reserve(n_select - 1);

    for _ in 1..n_select {
        // Find the maximum minimum (maxmin) distance and the corresponding
        // point. The maxmin point must be one of the farthest points from the
        // Voronoï decomposition, so we only have to look at the list of
        // existing cells to find it.
        let new_point = {
            let cells = voronoi.cells();
            let (max_radius_cell, _) = find_max(cells.radius2.iter());
            cells.farthest[max_radius_cell]
        };
        voronoi.add_point(new_point);
    }

    return voronoi.cells().center_idx.to_owned();
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
