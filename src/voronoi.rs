use std::cell::Cell;
use std::collections::HashSet;

use rayon::prelude::*;
use thread_local::ThreadLocal;

use soa_derive::{StructOfArray, soa_zip};
use ndarray::{Array1, ArrayView2, Axis, CowArray, Ix2, s};

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
    /// Indexes of the points in this cell
    points: Vec<usize>,
}

/// Point farthest away from a the center of a cell
#[derive(Clone, Debug, PartialEq, Default)]
struct FarthestPoint {
    pub distance2: Cell<f64>,
    pub index: Cell<usize>,
}

/// allocation cache for `VoronoiDecomposer` when adding a new point
#[derive(Debug)]
struct WorkArrays {
    /// Distance between the new point and the center of all cells
    distance_to_new_point: Vec<f64>,
    /// List of active cells that might need to change
    active_cells: HashSet<usize>,
}

impl WorkArrays {
    fn new() -> WorkArrays {
        WorkArrays {
            distance_to_new_point: Vec::new(),
            active_cells: HashSet::new(),
        }
    }

    fn clear(&mut self) {
        self.distance_to_new_point.clear();
        self.active_cells.clear();
    }

    fn reserve(&mut self, additional: usize) {
        self.distance_to_new_point.reserve(additional);
        self.active_cells.reserve(additional);
    }
}

#[derive(Debug)]
pub struct VoronoiDecomposer<'a> {
    /// Input points
    points: CowArray<'a, f64, Ix2>,
    /// Current list of cells
    cells: VoronoiCellVec,
    /// Norm of the vector from origin for each points
    norms: Vec<f64>,
    /// Shortest distance for each point to already selected points
    haussdorf: Vec<f64>,
    /// haussdorf distance of the first point
    initial_max_haussdorf: f64,
    /// Cached allocations when adding new points
    work: WorkArrays,
}

impl<'a> VoronoiDecomposer<'a> {
    #[cfg_attr(feature = "time-graph", time_graph::instrument(name = "initialize voronoi"))]
    pub fn new(points: CowArray<'a, f64, Ix2>, initial: usize) -> VoronoiDecomposer<'a> {
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
            points: (0..points.shape()[0]).collect()
        });

        VoronoiDecomposer {
            points: points,
            cells: cells,
            norms: norms.to_vec(),
            haussdorf: haussdorf.to_vec(),
            initial_max_haussdorf: radius2,
            work: WorkArrays::new(),
        }
    }

    /// Allocate capacity for `additional` more cells/selected points
    pub fn reserve(&mut self, additional: usize) {
        self.cells.reserve(additional);
        self.work.reserve(additional);
    }

    /// Add a new selected point as the center of a Voronoï cell
    #[cfg_attr(feature = "time-graph", time_graph::instrument(name = "add new voronoi cell"))]
    pub fn add_point(&mut self, new_point: usize) {
        self.work.clear();

        let new_center = self.points.slice(s![new_point, ..]);
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
                    self.work.active_cells.insert(cell_idx);
                }
            }

            for &cell_idx in &self.work.active_cells {
                self.cells.radius2[cell_idx] = 0.0;
                self.cells.farthest[cell_idx] = self.cells.center_idx[cell_idx];
            }
        });

        let mut new_cell = VoronoiCell {
            center: new_center.to_owned(),
            center_idx: new_point,
            // these will be updated below,
            radius2: 0.0,
            farthest: new_point,
            points: Vec::new(),
        };

        // use a channel to communicate the points that need to be added to the
        // new cell.
        let (new_cell_points_sender, new_cell_points_receiver) = std::sync::mpsc::channel();
        let new_farthest_point = ThreadLocal::new();

        let points = &self.points;
        let norms = &self.norms;
        let work = &self.work;
        let all_haussdorf = &self.haussdorf;

        tracing_span!("update decomposition", {
            self.cells.points
                .par_iter_mut()
                .zip_eq(&mut self.cells.radius2)
                .zip_eq(&mut self.cells.farthest)
                .enumerate()
                .filter_map(|(cell_idx, ((points_idx, radius2), farthest))| {
                    if work.active_cells.contains(&cell_idx) {
                        Some((cell_idx, points_idx, radius2, farthest))
                    } else {
                        None
                    }
                })
                .for_each_with(new_cell_points_sender, |sender, (cell_idx, points_idx, radius2, farthest)| {
                    let mut cell_updated_points = Vec::new();
                    // farthest point found on this thread
                    let farthest_point = new_farthest_point.get_or(FarthestPoint::default);

                    for &point in &*points_idx {
                        let haussdorf = all_haussdorf[point];

                        // Check if we can skip this check for this point. This is a
                        // tighter bound on the distance, since ||x_j - x_new|| <
                        // new_radius
                        if 0.25 * work.distance_to_new_point[cell_idx] < haussdorf {
                            let d2 = norms[new_point] + norms[point] - 2.0 * new_center.dot(&points.slice(s![point, ..]));
                            if haussdorf > d2 {
                                // We assign this point to the new cell
                                sender.send((point, d2)).expect("failed to send new point");

                                if d2 > farthest_point.distance2.get() {
                                    farthest_point.distance2.set(d2);
                                    farthest_point.index.set(point);
                                }

                                continue;
                            }
                        }

                        // the point is still in the same cell, make sure to
                        // update the cell radius/farthest point if needed
                        cell_updated_points.push(point);
                        if haussdorf > *radius2 {
                            *radius2 = haussdorf;
                            *farthest = point;
                        }
                    }

                    std::mem::swap(points_idx, &mut cell_updated_points);
                });

                for (point, haussdorf) in new_cell_points_receiver.iter() {
                    new_cell.points.push(point);
                    self.haussdorf[point] = haussdorf;
                }
        });

        for farthest in new_farthest_point.into_iter() {
            if farthest.distance2.get() > new_cell.radius2 {
                new_cell.radius2 = farthest.distance2.get();
                new_cell.farthest = farthest.index.get();
            }
        }

        self.cells.push(new_cell);

        // sanity check that all points are in the right place
        for cell in &self.cells {
            debug_assert!(!cell.points.is_empty());
            if cell.points.len() == 1 {
                debug_assert_eq!(cell.points[0], *cell.center_idx);
            }
        }
    }

    /// Access the current list of cells
    pub fn cells(&self) -> VoronoiCellSlice {
        self.cells.as_slice()
    }

    /// Get the potential next point, i.e. the point with highest Haussdorf distance
    pub fn next_point(&self) -> (usize, f64) {
        let (max_radius_cell, radius2) = find_max(self.cells.radius2.iter());

        if radius2 / self.initial_max_haussdorf < 1e-12 {
            panic!(
                "unable to select more than {} points, all remaining \
                points are identical to already selected points",
                self.cells.len()
            );
        }

        return (self.cells.farthest[max_radius_cell], radius2);
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

    let mut voronoi = VoronoiDecomposer::new(points.into(), initial);
    voronoi.reserve(n_select - 1);

    for _ in 1..n_select {
        // Find the maximum minimum (maxmin) distance and the corresponding
        // point. The maxmin point must be one of the farthest points from the
        // Voronoï decomposition, so we only have to look at the list of
        // existing cells to find it.
        let new_point = {
            let cells = voronoi.cells();
            let (max_radius_cell, radius2) = find_max(cells.radius2.iter());

            if radius2 / voronoi.initial_max_haussdorf < 1e-12 {
                panic!(
                    "unable to select more than {} points, all remaining \
                    points are identical to already selected points",
                    voronoi.cells.len()
                );
            }

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
