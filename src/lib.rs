use std::path::{Path, PathBuf};
use std::error::Error;
use geo::Geometry;

pub mod partitioner;

pub enum InputFormat {
    GeoJSON,
    GML,
}

pub fn process_file(
    path: &Path, 
    format: InputFormat,
    _target_crs: Option<&str>
) -> Result<Vec<Geometry<f64>>, Box<dyn Error>> {
    match format {
        InputFormat::GeoJSON => {
            partitioner::load_geometries(&path.to_path_buf())
        },
        InputFormat::GML => {
            let mut reader = partitioner::StreamingGmlReader::new(path)?;
            let mut geometries = Vec::new();
            while let Some(geometry) = reader.next_geometry()? {
                geometries.push(geometry);
            }
            Ok(geometries)
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