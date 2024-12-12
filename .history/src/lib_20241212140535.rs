pub mod partitioner;

pub use partitioner::*;

pub enum InputFormat {
    GeoJSON,  // Keep GeoJSON support
    // Remove GML for now since it's GDAL-dependent
}

pub fn process_file(
    path: &Path, 
    format: InputFormat,
    target_crs: Option<&str>
) -> Result<Vec<Geometry<f64>>, Box<dyn Error>> {
    match format {
        InputFormat::GeoJSON => {
            // If you need CRS transformation, you can add it here using proj
            load_geometries(&path.to_path_buf())
        }
    }
}