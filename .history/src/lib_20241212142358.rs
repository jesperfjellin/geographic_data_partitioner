use std::path::Path;
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
    target_crs: Option<&str>
) -> Result<Vec<Geometry<f64>>, Box<dyn Error>> {
    match format {
        InputFormat::GeoJSON => {
            partitioner::load_geometries(&path.to_path_buf())
        },
        InputFormat::GML => {
            let mut reader = partitioner::GmlReader::new();
            reader.read_geometries(path)
        }
    }
}