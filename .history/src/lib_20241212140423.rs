pub mod partitioner;

pub use partitioner::*;

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
            load_geometries(&path.to_path_buf())
        },
        InputFormat::GML => {
            let mut reader = GmlReader::new(path)?;
            if let Some(crs) = target_crs {
                reader.set_target_crs(crs);
            }
            reader.read_geometries()
        }
    }
}