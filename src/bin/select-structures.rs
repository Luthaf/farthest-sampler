#![allow(clippy::needless_return)]

use clap::{Arg, App};

use ndarray::{Array1, Array2, concatenate, Axis};
use ndarray_npy::{read_npy, write_npy};

use voronoi_fps::voronoi::VoronoiDecomposer;
use voronoi_fps::find_max;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = App::new("select-structures")
        .author("Guillaume Fraux <guillaume.fraux@epfl.ch>")
        .about(
"Select training points from a dataset using a Voronoï realization of FPS.

This tool automatically select adds all environments from a structure when any
environment in this structure is selected."
        )
        .arg(Arg::with_name("structures")
            .long("structures")
            .value_name("structures.npy")
            .help("array of structure indexes")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("n_structures")
            .short("n")
            .help("how many structures to select")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("output")
            .short("o")
            .long("output")
            .value_name("output.npy")
            .help("where to output selected structures indexes")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("output-radius")
            .long("radius")
            .value_name("radius.npy")
            .help("where to output Voronoi radii of selected points")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("selected")
            .long("already-selected")
            .value_name("selected.npy")
            .help("set of points already selected")
            .takes_value(true)
            .required(true))
        .arg(Arg::with_name("points")
            .long("points")
            .value_name("points.npy")
            .help("points to select")
            .takes_value(true)
            .required(true))
        .get_matches();

    let points: Array2<f64> = read_npy(matches.value_of("points").unwrap())?;
    let selected: Array2<f64> = read_npy(matches.value_of("selected").unwrap())?;
    let structures: Array1<i32> = read_npy(matches.value_of("structures").unwrap())?;
    let n_select: usize = matches.value_of("n_structures").unwrap().parse()?;

    let n_already_selected = selected.nrows();
    let selected_structures_id = vec![-1; n_already_selected];
    let points = concatenate!(Axis(0), selected, points);
    let structures = concatenate!(Axis(0), selected_structures_id, structures);

    let initial = 0;
    let mut voronoi = VoronoiDecomposer::new(points.view(), initial);
    let mut radius_when_selected = Vec::new();
    radius_when_selected.push(*voronoi.cells().last().unwrap().radius2);

    for i in 1..n_already_selected {
        voronoi.add_point(i);
        radius_when_selected.push(*voronoi.cells().last().unwrap().radius2);
    }
    let mut selected_structures = Vec::new();
    if n_already_selected == 0 {
        selected_structures.push(structures[initial]);
        // add all environments from the first structure
        for point in structures.iter()
        .enumerate()
        .filter_map(|(i, &s)| {
            if s == structures[initial] && i != initial {
                Some(i)
            } else {
                None
            }
        }) {
                voronoi.add_point(point);
                radius_when_selected.push(*voronoi.cells().last().unwrap().radius2);
            }
    }

    for _ in 1..n_select {
        let selected_point = {
            let cells = voronoi.cells();
            let (max_radius_cell, radius) = find_max(cells.radius2.iter());
            radius_when_selected.push(radius);

            cells.farthest[max_radius_cell]
        };

        voronoi.add_point(selected_point);

        let selected = structures[selected_point];
        selected_structures.push(selected);
        for point in structures.iter()
            .enumerate()
            .filter_map(|(i, &s)| {
                if s == selected && i != selected_point {
                    Some(i)
                } else {
                    None
                }
            }) {
                voronoi.add_point(point);
                radius_when_selected.push(*voronoi.cells().last().unwrap().radius2);
            }
    }

    let selected_structures = Array1::from(selected_structures);
    write_npy(matches.value_of("output").unwrap(), &selected_structures)?;

    let radius_when_selected = Array1::from(radius_when_selected);
    write_npy(matches.value_of("output-radius").unwrap(), &radius_when_selected)?;

    return Ok(());
}
