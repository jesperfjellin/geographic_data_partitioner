use std::io::BufReader;
use geo::{Geometry, Polygon, LineString, Point};
use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::contains::Contains;
use geo::algorithm::intersects::Intersects;
use geojson::{Value as GeoJsonValue};
use std::path::PathBuf;
use std::fs::File;
use std::fs::create_dir_all;
use std::error::Error;
use std::path::Path;
use geojson::{Feature, FeatureCollection, GeoJson, Geometry as GeoJsonGeometry};
use quick_xml::Reader;
use quick_xml::events::Event;

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateSystem {
    pub srid: String,
    pub source: String, // file name where this CRS was found
}

pub struct Partition {
    pub boundary: Polygon<f64>,
}

// Add this new struct to store clipped geometries for each partition
struct PartitionOutput {
    id: usize,
    geometries: Vec<Geometry<f64>>,
}


// Add this struct alongside your existing ones
pub struct GmlReader {
    // Remove proj-related fields
}

// Add these implementations after your existing code
impl GmlReader {
    pub fn new() -> Self {
        GmlReader {}
    }

    pub fn read_geometries(&mut self, path: &Path) -> Result<Vec<Geometry<f64>>, Box<dyn Error>> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut reader = Reader::from_reader(buf_reader);
        reader.trim_text(true);

        let mut geometries = Vec::new();
        let mut buf = Vec::new();
        let mut coords_text = String::new();
        let mut in_coordinates = false;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"gml:coordinates" | b"coordinates" => {
                            in_coordinates = true;
                            coords_text.clear();
                        },
                        _ => ()
                    }
                },
                Ok(Event::Text(e)) if in_coordinates => {
                    coords_text.push_str(&e.unescape().unwrap_or_default());
                },
                Ok(Event::End(ref e)) => {
                    match e.name().as_ref() {
                        b"gml:coordinates" | b"coordinates" => {
                            in_coordinates = false;
                            if let Some(geom) = self.parse_coordinates(&coords_text)? {
                                geometries.push(geom);
                            }
                        },
                        _ => ()
                    }
                },
                Ok(Event::Eof) => break,
                Err(e) => return Err(Box::new(e)),
                _ => ()
            }
            buf.clear();
        }

        Ok(geometries)
    }

    fn parse_coordinates(&self, coords_text: &str) -> Result<Option<Geometry<f64>>, Box<dyn Error>> {
        let mut points = Vec::new();
        
        for coord_pair in coords_text.split_whitespace() {
            let coords: Vec<&str> = coord_pair.split(',').collect();
            if coords.len() >= 2 {
                let x = coords[0].parse::<f64>()?;
                let y = coords[1].parse::<f64>()?;
                points.push((x, y).into()); // Add .into() here
            }
        }
    
        if points.len() >= 2 {
            if points.first() == points.last() {
                Ok(Some(Geometry::Polygon(Polygon::new(
                    LineString::new(points),
                    vec![]
                ))))
            } else {
                Ok(Some(Geometry::LineString(LineString::new(points))))
            }
        } else {
            Ok(None)
        }
    }

    // Add the new CRS detection method
    pub fn detect_crs(&self, path: &Path) -> Result<CoordinateSystem, Box<dyn Error>> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let mut reader = Reader::from_reader(buf_reader);
        reader.trim_text(true);

        let mut buf = Vec::new();
        let mut srid = None;

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"gml:srsName" | b"srsName" => {
                            if let Ok(Event::Text(text)) = reader.read_event_into(&mut buf) {
                                let srs_text = text.unescape().unwrap_or_default();
                                if let Some(epsg_code) = extract_epsg_code(&srs_text) {
                                    srid = Some(epsg_code);
                                    break;
                                }
                            }
                        },
                        _ => ()
                    }
                },
                Ok(Event::Eof) => break,
                Err(e) => return Err(Box::new(e)),
                _ => ()
            }
            buf.clear();
        }

        match srid {
            Some(code) => Ok(CoordinateSystem {
                srid: code,
                source: path.display().to_string(),
            }),
            None => Err("No CRS information found in GML file".into())
        }
    }
}

fn extract_epsg_code(srs: &str) -> Option<String> {
    if srs.contains("EPSG") {
        // Handle URN format
        if let Some(code) = srs.split(':').last() {
            return Some(format!("EPSG:{}", code));
        }
        // Handle URL format
        if let Some(code) = srs.split('#').last() {
            return Some(format!("EPSG:{}", code));
        }
    }
    None
}

// Add this function near other public functions
pub fn validate_crs(files: &[PathBuf]) -> Result<Option<CoordinateSystem>, Box<dyn Error>> {
    let mut detected_crs: Option<CoordinateSystem> = None;

    for file in files {
        let crs = match file.extension().and_then(|ext| ext.to_str()) {
            Some("gml") | Some("xml") => {
                let reader = GmlReader::new();
                Some(reader.detect_crs(file)?)
            },
            Some("geojson") | Some("json") => {
                // GeoJSON files are assumed to be WGS84 unless specified otherwise
                Some(CoordinateSystem {
                    srid: "EPSG:4326".to_string(),
                    source: file.display().to_string(),
                })
            },
            _ => None
        };

        if let Some(file_crs) = crs {
            match &detected_crs {
                None => detected_crs = Some(file_crs),
                Some(existing_crs) => {
                    if existing_crs.srid != file_crs.srid {
                        return Err(format!(
                            "CRS mismatch: {} uses {}, but {} uses {}",
                            existing_crs.source, existing_crs.srid,
                            file_crs.source, file_crs.srid
                        ).into());
                    }
                }
            }
        }
    }

    Ok(detected_crs)
}

pub fn load_geometries(file_path: &PathBuf) -> Result<Vec<Geometry<f64>>, Box<dyn Error>> {
    println!("Loading file: {}", file_path.display());
    
    match file_path.extension().and_then(|ext| ext.to_str()) {
        Some("geojson") | Some("json") => load_geojson(file_path),
        Some("gml") | Some("xml") => {
            let mut reader = GmlReader::new();
            reader.read_geometries(file_path)
        },
        _ => Err("Unsupported file format".into())
    }
}

fn load_geojson(file_path: &PathBuf) -> Result<Vec<Geometry<f64>>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let geojson = GeoJson::from_reader(reader)?;
    let mut geometries = Vec::new();
    
    if let GeoJson::FeatureCollection(fc) = geojson {
        let total_features = fc.features.len();
        println!("Found {} features in file", total_features);
        
        for (feature_count, feature) in fc.features.into_iter().enumerate() {
            if (feature_count + 1) % 1000 == 0 {
                println!("Processing feature {}/{}", feature_count + 1, total_features);
            }
            if let Some(geometry) = feature.geometry {
                match geometry.value {
                    GeoJsonValue::LineString(coords) => {
                        let points: Vec<_> = coords.iter()
                            .map(|coord| (coord[0], coord[1]).into())
                            .collect();
                        geometries.push(Geometry::LineString(LineString::new(points)));
                    },
                    GeoJsonValue::Polygon(coords) => {
                        let exterior: Vec<_> = coords[0].iter()
                            .map(|coord| (coord[0], coord[1]).into())
                            .collect();
                        let holes: Vec<LineString<f64>> = coords.iter().skip(1)
                            .map(|ring| {
                                LineString::new(
                                    ring.iter()
                                        .map(|coord| (coord[0], coord[1]).into())
                                        .collect()
                                )
                            })
                            .collect();
                        geometries.push(Geometry::Polygon(
                            Polygon::new(LineString::new(exterior), holes)
                        ));
                    },
                    _ => continue,
                }
            }
        }
        println!("Loaded {} geometries from file", geometries.len());
    }
    
    Ok(geometries)
}

pub fn calculate_common_extent(files: &[PathBuf]) -> Result<Polygon<f64>, Box<dyn Error>> { 
    println!("Calculating extent across {} files...", files.len());
    // Initialize min/max coordinates with first geometry's bounds
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    // Iterate through all files to find overall bounds
    for file in files {
        let geometries = load_geometries(file)?;
        for geometry in geometries {
            if let Some(bbox) = geometry.bounding_rect() {
                min_x = min_x.min(bbox.min().x);
                min_y = min_y.min(bbox.min().y);
                max_x = max_x.max(bbox.max().x);
                max_y = max_y.max(bbox.max().y);
            }
        }
    }

    // Create polygon from bounding box coordinates
    let exterior = LineString::new(vec![
        (min_x, min_y).into(),
        (max_x, min_y).into(),
        (max_x, max_y).into(),
        (min_x, max_y).into(),
        (min_x, min_y).into(), // Close the polygon
    ]);

    Ok(Polygon::new(exterior, vec![]))
}

// Function to suggest number of partitions based on file size
pub fn suggest_partitions(files: &[PathBuf]) -> usize {
    // Simple heuristic: 1 partition per 10MB of data
    let total_size: u64 = files.iter()
        .map(|f| f.metadata().map(|m| m.len()).unwrap_or(0))
        .sum();
    
    let suggested = (total_size / (10 * 1024 * 1024)).max(1) as usize;
    println!("Suggesting {} partitions based on total file size of {} bytes", 
             suggested, total_size);
    suggested
}

// Function to create partitions based on user input or suggestion
pub fn create_partitions(extent: &Polygon<f64>, num_partitions: usize) -> Vec<Partition> {
    let bbox = extent.bounding_rect().unwrap();
    let width = bbox.max().x - bbox.min().x;
    let height = bbox.max().y - bbox.min().y;

    // Calculate grid dimensions
    let ratio = width / height;
    let num_cols = (num_partitions as f64 * ratio).sqrt().ceil() as usize;
    let num_rows = ((num_partitions as f64) / ratio).sqrt().ceil() as usize;

    println!("Creating {}x{} grid of partitions", num_rows, num_cols);
    println!("Total number of partitions: {}", num_rows * num_cols);
    println!("Extent dimensions: {:.2} x {:.2}", width, height);
    println!("Partition dimensions: {:.2} x {:.2}", width / num_cols as f64, height / num_rows as f64);
    println!("Extent bounds: ({:.2}, {:.2}) to ({:.2}, {:.2})", 
        bbox.min().x, bbox.min().y, bbox.max().x, bbox.max().y);

    let cell_width = width / num_cols as f64;
    let cell_height = height / num_rows as f64;

    let mut partitions = Vec::with_capacity(num_rows * num_cols);

    // Create grid of partitions
    for row in 0..num_rows {
        for col in 0..num_cols {
            let min_x = bbox.min().x + (col as f64 * cell_width);
            let min_y = bbox.min().y + (row as f64 * cell_height);
            let max_x = min_x + cell_width;
            let max_y = min_y + cell_height;

            println!("Partition {},{} bounds: ({:.2}, {:.2}) to ({:.2}, {:.2})", 
                row, col, min_x, min_y, max_x, max_y);

            // Create partition polygon
            let exterior = LineString::new(vec![
                (min_x, min_y).into(),
                (max_x, min_y).into(),
                (max_x, max_y).into(),
                (min_x, max_y).into(),
                (min_x, min_y).into(), // Close the polygon
            ]);

            partitions.push(Partition {
                boundary: Polygon::new(exterior, vec![])
            });
        }
    }

    partitions
}

pub fn clip_linestring(line: &LineString<f64>, partition: &Partition) -> Option<LineString<f64>> {
    if let (Some(line_bbox), Some(partition_bbox)) = (line.bounding_rect(), partition.boundary.bounding_rect()) {
        // Quick rejection test
        if !line_bbox.intersects(&partition_bbox) {
            return None;
        }

        let points: Vec<_> = line.points().collect();
        let mut clipped_points = Vec::new();
        
        for i in 0..points.len() - 1 {
            let p1 = points[i];
            let p2 = points[i + 1];
            
            let p1_inside = partition.boundary.contains(&Point::new(p1.x(), p1.y()));
            let p2_inside = partition.boundary.contains(&Point::new(p2.x(), p2.y()));

            match (p1_inside, p2_inside) {
                (true, true) => {
                    if clipped_points.is_empty() {
                        clipped_points.push(p1);
                    }
                    clipped_points.push(p2);
                },
                (true, false) | (false, true) => {
                    if let Some(intersection) = find_edge_intersection(
                        (p1.x(), p1.y()),
                        (p2.x(), p2.y()),
                        &partition.boundary
                    ) {
                        let int_point = Point::new(intersection.0, intersection.1);
                        
                        if p1_inside {
                            if clipped_points.is_empty() {
                                clipped_points.push(p1);
                            }
                            clipped_points.push(int_point);
                        } else if p2_inside {
                            if clipped_points.is_empty() {
                                clipped_points.push(int_point);
                            }
                            clipped_points.push(p2);
                        }
                    }
                },
                (false, false) => ()
            }
        }

        if clipped_points.len() >= 2 {
            return Some(LineString::new(
                clipped_points.into_iter()
                    .map(|p| (p.x(), p.y()).into())
                    .collect()
            ));
        }
    }
    None
}

pub fn clip_polygon(polygon: &Polygon<f64>, partition: &Partition) -> Option<Polygon<f64>> {
    if let (Some(poly_bbox), Some(partition_bbox)) = (polygon.bounding_rect(), partition.boundary.bounding_rect()) {
        // Quick rejection test
        if !poly_bbox.intersects(&partition_bbox) {
            return None;
        }

        // Check for complete containment cases
        let partition_corners: Vec<Point<f64>> = partition.boundary.exterior()
            .points()
            .take(4)
            .collect();
        
        let all_corners_inside = partition_corners.iter()
            .all(|point| polygon.contains(point));

        if all_corners_inside {
            return Some(partition.boundary.clone());
        }

        if partition.boundary.contains(polygon) {
            return Some(polygon.clone());
        }

        let exterior_points: Vec<_> = polygon.exterior().points().collect();
        let mut new_exterior_points = Vec::new();
        
        // Define add_point helper function at the function level
        let add_point = |points: &mut Vec<Point<f64>>, point: Point<f64>| {
            const EPSILON: f64 = 1e-10;
            if points.is_empty() || 
               points.last().map(|last| {
                   (last.x() - point.x()).abs() > EPSILON || 
                   (last.y() - point.y()).abs() > EPSILON
               }).unwrap_or(true) {
                points.push(point);
            }
        };
        
        // Process each edge of the polygon
        for i in 0..exterior_points.len() - 1 {
            let p1 = exterior_points[i];
            let p2 = exterior_points[i + 1];
            
            let p1_inside = partition.boundary.contains(&Point::new(p1.x(), p1.y()));
            let p2_inside = partition.boundary.contains(&Point::new(p2.x(), p2.y()));

            match (p1_inside, p2_inside) {
                (true, true) => {
                    add_point(&mut new_exterior_points, p1);
                    add_point(&mut new_exterior_points, p2);
                },
                (true, false) | (false, true) => {
                    let intersections = find_all_edge_intersections(
                        (p1.x(), p1.y()),
                        (p2.x(), p2.y()),
                        &partition.boundary
                    );

                    // Sort intersections by distance from p1
                    let mut sorted_intersections: Vec<_> = intersections.into_iter()
                        .map(|(x, y)| {
                            let dx = x - p1.x();
                            let dy = y - p1.y();
                            let dist = (dx * dx + dy * dy).sqrt();
                            (dist, Point::new(x, y))
                        })
                        .collect();
                    sorted_intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                    if p1_inside {
                        add_point(&mut new_exterior_points, p1);
                        if let Some((_, int_point)) = sorted_intersections.first() {
                            add_point(&mut new_exterior_points, int_point.clone());
                        }
                    } else {
                        if let Some((_, int_point)) = sorted_intersections.last() {
                            add_point(&mut new_exterior_points, int_point.clone());
                            add_point(&mut new_exterior_points, p2);
                        }
                    }
                },
                (false, false) => {
                    let intersections = find_all_edge_intersections(
                        (p1.x(), p1.y()),
                        (p2.x(), p2.y()),
                        &partition.boundary
                    );
                    
                    if intersections.len() >= 2 {
                        // Sort intersections by distance from p1
                        let mut sorted_intersections: Vec<_> = intersections.into_iter()
                            .map(|(x, y)| {
                                let dx = x - p1.x();
                                let dy = y - p1.y();
                                let dist = (dx * dx + dy * dy).sqrt();
                                (dist, Point::new(x, y))
                            })
                            .collect();
                        sorted_intersections.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                        // Add only the entry and exit points
                        if let Some((_, entry)) = sorted_intersections.first() {
                            add_point(&mut new_exterior_points, entry.clone());
                        }
                        if let Some((_, exit)) = sorted_intersections.last() {
                            add_point(&mut new_exterior_points, exit.clone());
                        }
                    }
                }
            }
        }

        // Close the polygon if needed
        if new_exterior_points.len() >= 3 {
            if new_exterior_points[0] != new_exterior_points[new_exterior_points.len() - 1] {
                // Clone the first point before using it
                let first_point = new_exterior_points[0].clone();
                add_point(&mut new_exterior_points, first_point);
            }

            let new_exterior = LineString::new(
                new_exterior_points.into_iter()
                    .map(|p| (p.x(), p.y()).into())
                    .collect()
            );

            let new_holes: Vec<LineString<f64>> = polygon.interiors()
                .iter()
                .filter_map(|hole| clip_ring(hole, partition))
                .collect();

            return Some(Polygon::new(new_exterior, new_holes));
        }
    }
    None
}

// New helper function to find intersection with partition boundary
fn find_edge_intersection(
    p1: (f64, f64),
    p2: (f64, f64),
    partition: &Polygon<f64>
) -> Option<(f64, f64)> {
    let partition_points: Vec<_> = partition.exterior().points().collect();
    
    for i in 0..partition_points.len() - 1 {
        let b1 = partition_points[i];
        let b2 = partition_points[i + 1];
        
        if let Some(intersection) = line_intersection(
            p1,
            p2,
            (b1.x(), b1.y()),
            (b2.x(), b2.y())
        ) {
            return Some(intersection);
        }
    }
    None
}

// New helper function to find all intersections along an edge
fn find_all_edge_intersections(
    p1: (f64, f64),
    p2: (f64, f64),
    partition: &Polygon<f64>
) -> Vec<(f64, f64)> {
    let partition_points: Vec<_> = partition.exterior().points().collect();
    let mut intersections = Vec::new();
    
    for i in 0..partition_points.len() - 1 {
        let b1 = partition_points[i];
        let b2 = partition_points[i + 1];
        
        if let Some(intersection) = line_intersection(
            p1,
            p2,
            (b1.x(), b1.y()),
            (b2.x(), b2.y())
        ) {
            intersections.push(intersection);
        }
    }
    
    intersections
}


pub fn process_distinct_files(files: Vec<PathBuf>, num_partitions: Option<usize>) -> Result<(), Box<dyn Error>> {
    println!("\n=== Starting Distinct GIS Processing ===");
    
    if let Some(crs) = validate_crs(&files)? {
        println!("Detected Coordinate Reference System: {}", crs.srid);
    } else {
        println!("Warning: Could not detect CRS information. Assuming WGS84 (EPSG:4326)");
    }
    
    // Create output directory
    let output_dir = Path::new("output");
    create_dir_all(output_dir)?;
    println!("Created output directory: {}", output_dir.display());

    // Calculate common extent from all files
    println!("\nCalculating common extent...");
    let common_extent = calculate_common_extent(&files)?;
    let partitions_count = num_partitions.unwrap_or_else(|| suggest_partitions(&files));
    let partitions = create_partitions(&common_extent, partitions_count);

    // Process each file separately but use the same partitions
    for (file_index, file) in files.iter().enumerate() {
        println!("\nProcessing file {}/{}: {}", file_index + 1, files.len(), file.display());
        let geometries = load_geometries(file)?;
        
        // Create partition outputs for this file
        let mut partition_outputs: Vec<PartitionOutput> = partitions.iter().enumerate()
            .map(|(id, _)| PartitionOutput { id, geometries: Vec::new() })
            .collect();

        println!("Processing geometries across {} partitions", partitions.len());
        let total_geometries = geometries.len();
        let mut processed_count = 0;
        let progress_interval = (total_geometries / 20).max(1);

        // Process geometries for this file
        for geometry in geometries {
            processed_count += 1;
            if processed_count % progress_interval == 0 {
                println!("Progress: {:.1}% ({}/{})", 
                    (processed_count as f64 / total_geometries as f64) * 100.0,
                    processed_count,
                    total_geometries
                );
            }

            for (partition_idx, partition) in partitions.iter().enumerate() {
                match &geometry {
                    Geometry::LineString(ref line) => {
                        if let Some(clipped_line) = clip_linestring(line, partition) {
                            partition_outputs[partition_idx].geometries.push(
                                Geometry::LineString(clipped_line)
                            );
                        }
                    },
                    Geometry::Polygon(ref polygon) => {
                        if let Some(clipped_polygon) = clip_polygon(polygon, partition) {
                            partition_outputs[partition_idx].geometries.push(
                                Geometry::Polygon(clipped_polygon)
                            );
                        }
                    },
                    _ => {}
                }
            }
        }

        // Write partitions for this file
        println!("\nWriting partitioned data for file {}...", file.display());
        let file_stem = file.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        for partition_output in &partition_outputs {
            if !partition_output.geometries.is_empty() {
                let output_path = output_dir.join(
                    format!("{}_partition_{}.geojson", file_stem, partition_output.id)
                );
                write_partition_to_file(partition_output, &output_path)?;
                println!("Written partition {} with {} geometries", 
                    partition_output.id,
                    partition_output.geometries.len()
                );
            }
        }
    }

    println!("\n=== Processing Complete ===");
    println!("Output files can be found in: {}", output_dir.display());
    Ok(())
}

fn clip_ring(ring: &LineString<f64>, partition: &Partition) -> Option<LineString<f64>> {
    if let (Some(ring_bbox), Some(partition_bbox)) = (ring.bounding_rect(), partition.boundary.bounding_rect()) {
        // Quick rejection test
        if !ring_bbox.intersects(&partition_bbox) {
            return None;
        }

        // If ring is completely inside partition, return as-is
        if partition.boundary.contains(ring) {
            return Some(ring.clone());
        }

        let points: Vec<_> = ring.points().collect();
        let mut clipped_points = Vec::new();
        
        // Process each edge of the ring
        for i in 0..points.len() - 1 {
            let p1 = points[i];
            let p2 = points[i + 1];
            
            let p1_inside = partition.boundary.contains(&Point::new(p1.x(), p1.y()));
            let p2_inside = partition.boundary.contains(&Point::new(p2.x(), p2.y()));

            match (p1_inside, p2_inside) {
                (true, true) => {
                    // Both points inside - add first point
                    if clipped_points.is_empty() {
                        clipped_points.push(p1);
                    }
                    clipped_points.push(p2);
                },
                (true, false) | (false, true) => {
                    // Edge crosses partition boundary - find intersection
                    if let Some(intersection) = find_edge_intersection(
                        (p1.x(), p1.y()),
                        (p2.x(), p2.y()),
                        &partition.boundary
                    ) {
                        let int_point = Point::new(intersection.0, intersection.1);
                        
                        if p1_inside {
                            if clipped_points.is_empty() {
                                clipped_points.push(p1);
                            }
                            clipped_points.push(int_point);
                        } else if p2_inside {
                            if clipped_points.is_empty() {
                                clipped_points.push(int_point);
                            }
                            clipped_points.push(p2);
                        }
                    }
                },
                (false, false) => () // Skip points outside partition
            }
        }

        // Close the ring if needed
        if !clipped_points.is_empty() && 
           clipped_points[0] != clipped_points[clipped_points.len() - 1] {
            clipped_points.push(clipped_points[0].clone());
        }

        // Create new ring if we have enough points
        if clipped_points.len() >= 4 { // Minimum points for a valid ring (3 + closing point)
            return Some(LineString::new(
                clipped_points.into_iter()
                    .map(|p| (p.x(), p.y()).into())
                    .collect()
            ));
        }
    }
    None
}


// Helper function to calculate exact line intersection point
fn line_intersection(
    p1: (f64, f64),
    p2: (f64, f64),
    p3: (f64, f64),
    p4: (f64, f64)
) -> Option<(f64, f64)> {
    let x1 = p1.0; let y1 = p1.1;
    let x2 = p2.0; let y2 = p2.1;
    let x3 = p3.0; let y3 = p3.1;
    let x4 = p4.0; let y4 = p4.1;

    let denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if denominator == 0.0 {
        return None; // Lines are parallel
    }

    let t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator;
    if t < 0.0 || t > 1.0 {
        return None; // Intersection point lies outside the first line segment
    }

    let u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator;
    if u < 0.0 || u > 1.0 {
        return None; // Intersection point lies outside the second line segment
    }

    let x = x1 + t * (x2 - x1);
    let y = y1 + t * (y2 - y1);
    
    Some((x, y))
}


fn write_partition_to_file(
    partition: &PartitionOutput,
    output_path: &Path,
) -> Result<(), Box<dyn Error>> {
    let features: Vec<Feature> = partition.geometries.iter().map(|geom| {
        let geojson_geom = match geom {
            Geometry::LineString(line) => {
                let coords: Vec<Vec<f64>> = line.points()
                    .map(|p| vec![p.x(), p.y()])
                    .collect();
                GeoJsonGeometry::new(geojson::Value::LineString(coords))
            },
            Geometry::Polygon(polygon) => {
                let exterior: Vec<Vec<f64>> = polygon.exterior()
                    .points()
                    .map(|p| vec![p.x(), p.y()])
                    .collect();
                let holes: Vec<Vec<Vec<f64>>> = polygon.interiors()
                    .iter()
                    .map(|ring| {
                        ring.points()
                            .map(|p| vec![p.x(), p.y()])
                            .collect()
                    })
                    .collect();
                let mut rings = vec![exterior];
                rings.extend(holes);
                GeoJsonGeometry::new(geojson::Value::Polygon(rings))
            },
            _ => panic!("Unsupported geometry type"),
        };
        
        Feature {
            bbox: None,
            geometry: Some(geojson_geom),
            id: None,
            properties: Some(serde_json::Map::new()),
            foreign_members: None,
        }
    }).collect();

    let feature_collection = FeatureCollection {
        bbox: None,
        features,
        foreign_members: None,
    };

    let file = File::create(output_path)?;
    serde_json::to_writer_pretty(file, &feature_collection)?;
    
    Ok(())
}

pub fn process_gis_files(files: Vec<PathBuf>, num_partitions: Option<usize>) -> Result<(), Box<dyn Error>> {
    println!("\n=== Starting GIS Processing ===");
    
    if let Some(crs) = validate_crs(&files)? {
        println!("Detected Coordinate Reference System: {}", crs.srid);
    } else {
        println!("Warning: Could not detect CRS information. Assuming WGS84 (EPSG:4326)");
    }
    
    // Create output directory
    let output_dir = Path::new("output");
    create_dir_all(output_dir)?;
    println!("Created output directory: {}", output_dir.display());

    // Load all geometries from all files first
    println!("\nLoading geometries from all input files...");
    let mut all_geometries = Vec::new();
    for file in &files {
        println!("Loading file: {}", file.display());
        let geometries = load_geometries(file)?;
        println!("Loaded {} geometries from {}", geometries.len(), file.display());
        all_geometries.extend(geometries);
    }
    println!("Total geometries loaded: {}", all_geometries.len());

    // Calculate common extent from combined geometries
    let common_extent = calculate_extent_from_geometries(&all_geometries)?;
    let partitions_count = num_partitions.unwrap_or_else(|| suggest_partitions(&files));
    let partitions = create_partitions(&common_extent, partitions_count);

    // Create a vector to store outputs for each partition
    let mut partition_outputs: Vec<PartitionOutput> = partitions.iter().enumerate()
        .map(|(id, _)| PartitionOutput { id, geometries: Vec::new() })
        .collect();

    // Process all geometries
    println!("\nProcessing geometries across {} partitions", partitions.len());
    let total_geometries = all_geometries.len();
    let mut processed_count = 0;
    let progress_interval = (total_geometries / 20).max(1);

    for geometry in all_geometries {
        processed_count += 1;
        if processed_count % progress_interval == 0 {
            println!("Progress: {:.1}% ({}/{})", 
                (processed_count as f64 / total_geometries as f64) * 100.0,
                processed_count,
                total_geometries
            );
        }

        for (partition_idx, partition) in partitions.iter().enumerate() {
            match &geometry {
                Geometry::LineString(ref line) => {
                    if let Some(clipped_line) = clip_linestring(line, partition) {
                        partition_outputs[partition_idx].geometries.push(
                            Geometry::LineString(clipped_line)
                        );
                    }
                },
                Geometry::Polygon(ref polygon) => {
                    if let Some(clipped_polygon) = clip_polygon(polygon, partition) {
                        partition_outputs[partition_idx].geometries.push(
                            Geometry::Polygon(clipped_polygon)
                        );
                    }
                },
                _ => {}
            }
        }
    }

    // Write one file per partition containing geometries from all input files
    println!("\nWriting partitioned data to files...");
    for partition_output in &partition_outputs {
        if !partition_output.geometries.is_empty() {
            let output_path = output_dir.join(format!("partition_{}.geojson", partition_output.id));
            write_partition_to_file(partition_output, &output_path)?;
            println!("Written partition {} with {} geometries", 
                partition_output.id,
                partition_output.geometries.len()
            );
        }
    }

    println!("\n=== Processing Complete ===");
    println!("Output files can be found in: {}", output_dir.display());
    Ok(())
}

// New helper function to calculate extent from geometries
fn calculate_extent_from_geometries(geometries: &[Geometry<f64>]) -> Result<Polygon<f64>, Box<dyn Error>> {
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for geometry in geometries {
        if let Some(bbox) = geometry.bounding_rect() {
            min_x = min_x.min(bbox.min().x);
            min_y = min_y.min(bbox.min().y);
            max_x = max_x.max(bbox.max().x);
            max_y = max_y.max(bbox.max().y);
        }
    }

    let exterior = LineString::new(vec![
        (min_x, min_y).into(),
        (max_x, min_y).into(),
        (max_x, max_y).into(),
        (min_x, max_y).into(),
        (min_x, min_y).into(),
    ]);

    Ok(Polygon::new(exterior, vec![]))
}