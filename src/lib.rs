use std::path::{Path, PathBuf};
use std::error::Error;
use geo::Geometry;

pub mod partitioner;

pub enum InputFormat {
    GeoJSON,
}

pub fn process_file(
    path: &Path, 
    format: InputFormat,
    _target_crs: Option<&str>
) -> Result<Vec<Geometry<f64>>, Box<dyn Error>> {
    match format {
        InputFormat::GeoJSON => {
            partitioner::load_geometries(path)
        }
    }
}

pub fn process_files(
    files: Vec<PathBuf>,
    num_partitions: Option<usize>,
    distinct: bool
) -> Result<(), Box<dyn Error>> {
    partitioner::process_gis_files(files, num_partitions, distinct)
}