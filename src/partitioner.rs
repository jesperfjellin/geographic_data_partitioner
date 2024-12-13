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
use std::collections::HashMap;
use std::io::{Write, BufWriter};
use std::fs::OpenOptions;
use serde_json;

#[derive(Debug, Clone, PartialEq)]
pub struct CoordinateSystem {
    pub srid: String,
    pub source: String, // file name where this CRS was found
}

#[derive(Clone, Debug)]
pub struct FeatureAttributes {
    pub feature_type: String,
    pub properties: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Partition {
    pub boundary: Polygon<f64>,
}

struct PartitionOutput {
    id: usize,
    features: Vec<(Geometry<f64>, FeatureAttributes)>,
}


pub struct StreamingGmlReader {
    reader: Reader<BufReader<File>>,
    buf: Vec<u8>,
    coords_text: String,
    file_path: PathBuf,  // Store the file path for CRS detection
}

impl StreamingGmlReader {
    pub fn next_feature(&mut self) -> Result<Option<(Geometry<f64>, FeatureAttributes)>, Box<dyn Error>> {
        let mut feature_type = String::new();
        let mut properties = HashMap::new();
        let mut in_geometry = false;
        let mut current_property = String::new();
        let mut coords_text = String::new();
        let mut debug_depth = 0;
        let mut in_feature = false;
        
        loop {
            match self.reader.read_event_into(&mut self.buf)? {
                Event::Start(e) => {
                    debug_depth += 1;
                    let name = e.name();
                    let name_str = std::str::from_utf8(name.as_ref())?;
                    println!("{:indent$}Start: {}", "", name_str, indent = debug_depth * 2);
                    
                    // Detect main feature (like Bufferavgrensning)
                    if !name_str.contains("gml:") && !name_str.contains("featureMember") && !in_feature {
                        println!("{:indent$}Found main feature: {}", "", name_str, indent = debug_depth * 2);
                        feature_type = name_str.to_string();
                        in_feature = true;
                        
                        // Get attributes from feature element
                        for attr in e.attributes() {
                            if let Ok(attr) = attr {
                                let key = std::str::from_utf8(attr.key.as_ref())?;
                                let value = attr.unescape_value()?;
                                println!("{:indent$}Found attribute: {} = {}", "", key, value, indent = debug_depth * 2);
                                properties.insert(key.to_string(), value.to_string());
                            }
                        }
                    }
                    
                    match name_str {
                        "geometri" => {
                            println!("{:indent$}Found geometri element", "", indent = debug_depth * 2);
                        },
                        "gml:coordinates" | "gml:posList" => {
                            println!("{:indent$}Entering geometry section", "", indent = debug_depth * 2);
                            in_geometry = true;
                            coords_text.clear();
                        },
                        _ => {
                            if in_feature {
                                current_property = name_str.to_string();
                            }
                        }
                    }
                },
                Event::Text(e) => {
                    let text = e.unescape().unwrap_or_default();
                    if in_geometry {
                        println!("{:indent$}Found geometry coordinates: {}", "", text, indent = debug_depth * 2);
                        coords_text.push_str(&text);
                    } else if in_feature && !current_property.contains("gml:") {
                        println!("{:indent$}Found property value: {} = {}", "", current_property, text, indent = debug_depth * 2);
                        properties.insert(current_property.clone(), text.to_string());
                    }
                },
                Event::End(e) => {
                    let name = e.name();
                    let name_str = std::str::from_utf8(name.as_ref())?;
                    println!("{:indent$}End: {}", "", name_str, indent = debug_depth * 2);
                    
                    match name_str {
                        "gml:coordinates" | "gml:posList" => {
                            println!("{:indent$}Parsing coordinates: {}", "", coords_text, indent = debug_depth * 2);
                            in_geometry = false;
                            let line_string = parse_coordinates(&coords_text)?;
                            println!("{:indent$}Successfully parsed geometry", "", indent = debug_depth * 2);
                            return Ok(Some((
                                Geometry::LineString(line_string),
                                FeatureAttributes {
                                    feature_type: feature_type.clone(),
                                    properties: properties.clone(),
                                }
                            )));
                        },
                        "geometri" => {
                            println!("{:indent$}End of geometri element", "", indent = debug_depth * 2);
                        },
                        _ if name_str == feature_type => {
                            // Only return None at the end of the main feature if no geometry was found
                            if in_feature {
                                println!("{:indent$}End of main feature {} with geometry found: {}", 
                                    "", name_str, !coords_text.is_empty(), indent = debug_depth * 2);
                                if coords_text.is_empty() {
                                    return Ok(None);
                                }
                            }
                        },
                        _ => ()
                    }
                    debug_depth -= 1;
                },
                Event::Eof => {
                    println!("Reached end of file");
                    return Ok(None);
                },
                _ => (),
            }
            self.buf.clear();
        }
    }

    // Existing methods remain unchanged:
    pub fn new(path: &Path) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let buf_reader = BufReader::new(file);
        let reader = Reader::from_reader(buf_reader);
        
        Ok(StreamingGmlReader {
            reader,
            buf: Vec::new(),
            coords_text: String::new(),
            file_path: path.to_path_buf(),
        })
    }

    pub fn detect_crs(&mut self) -> Result<CoordinateSystem, Box<dyn Error>> {
        let file = File::open(&self.file_path)?;
        let buf_reader = BufReader::new(file);
        let mut reader = Reader::from_reader(buf_reader);
        let mut buf = Vec::new();
        let mut srid = None;
        
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name().as_ref() == b"gml:Envelope" {
                        // Look for srsName attribute
                        for attr in e.attributes() {
                            if let Ok(attr) = attr {
                                if attr.key.as_ref() == b"srsName" {
                                    if let Ok(value) = attr.unescape_value() {
                                        if let Some(epsg_code) = extract_epsg_code(&value) {
                                            srid = Some(epsg_code);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
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
                source: self.file_path.display().to_string(),
            }),
            None => Err(format!("No CRS information found in GML file: {}.",
                                self.file_path.display()).into())
        }
    }

    pub fn next_geometry(&mut self) -> Result<Option<Geometry<f64>>, Box<dyn Error>> {
        let mut in_coordinates = false;
        
        loop {
            match self.reader.read_event_into(&mut self.buf) {
                Ok(Event::Start(ref e)) => {
                    // Create a longer-lived binding for the name
                    let name = e.name();
                    let name_str = std::str::from_utf8(name.as_ref())?;
                    match name_str {
                        "gml:coordinates" | "coordinates" | "gml:posList" | "posList" => {
                            in_coordinates = true;
                            self.coords_text.clear();
                        },
                        _ => ()
                    }
                },
                Ok(Event::Text(e)) if in_coordinates => {
                    let text = e.unescape().unwrap_or_default();
                    self.coords_text.push_str(&text);
                },
                Ok(Event::End(ref e)) => {
                    // Create a longer-lived binding for the name
                    let name = e.name();
                    let name_str = std::str::from_utf8(name.as_ref())?;
                    match name_str {
                        "gml:coordinates" | "coordinates" | "gml:posList" | "posList" => {
                            let line_string = parse_coordinates(&self.coords_text)?;
                            return Ok(Some(Geometry::LineString(line_string)));
                        },
                        _ => ()
                    }
                },
                Ok(Event::Eof) => return Ok(None),
                Err(e) => return Err(Box::new(e)),
                _ => ()
            }
            self.buf.clear();
        }
    }
}


fn extract_epsg_code(srs: &str) -> Option<String> {
    // Handle URL format (http://www.opengis.net/def/crs/EPSG/0/XXXXX)
    if let Some(code) = srs.split('/').last() {
        let code = code.parse::<i32>().ok()?;
        // If it's EPSG:6173 (ETRS89 / UTM zone 33N + NN2000 height),
        // treat it as EPSG:25833 (ETRS89 / UTM zone 33N)
        if code == 6173 {
            return Some("EPSG:25833".to_string());
        }
        return Some(format!("EPSG:{}", code));
    }
    None
}

fn parse_coordinates(coords_text: &str) -> Result<LineString<f64>, Box<dyn Error>> {
    let mut points = Vec::new();
    let numbers: Vec<&str> = coords_text.split_whitespace().collect();

    let mut i = 0;
    while i < numbers.len() {
        if i + 1 < numbers.len() {
            let x_res = numbers[i].parse::<f64>();
            let y_res = numbers[i + 1].parse::<f64>();

            match (x_res, y_res) {
                (Ok(x), Ok(y)) => {
                    points.push((x, y).into());
                    // Check if there's a 'NaN' after the Y coordinate (some GML has 3D with NaN Z)
                    if i + 2 < numbers.len() && numbers[i + 2] == "NaN" {
                        i += 3; // Skip the NaN
                    } else {
                        i += 2;
                    }
                }
                _ => {
                    eprintln!("Warning: Failed to parse coordinate pair: {}, {}",
                        numbers[i], numbers[i + 1]);
                    i += 2; // Skip these malformed coordinates
                }
            }
        } else {
            // Odd number of coordinates, can't form a pair, just break
            break;
        }
    }

    Ok(LineString::new(points))
}

pub fn validate_crs(files: &[PathBuf]) -> Result<Option<CoordinateSystem>, Box<dyn Error>> {
    let mut detected_crs: Option<CoordinateSystem> = None;

    for file in files {
        let crs = match file.extension().and_then(|ext| ext.to_str()) {
            Some("gml") | Some("xml") => {
                let mut reader = StreamingGmlReader::new(file)?;
                Some(reader.detect_crs()?)
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
            let mut reader = StreamingGmlReader::new(file_path)?;
            let mut geometries = Vec::new();
            while let Some(geometry) = reader.next_geometry()? {
                geometries.push(geometry);
            }
            Ok(geometries)
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

pub fn calculate_common_extent_streaming(
    files: &[PathBuf],
    crs: &CoordinateSystem
) -> Result<Polygon<f64>, Box<dyn Error>> {
    println!("Calculating extent across {} files using {}", files.len(), crs.srid);
    println!("Calculating extent across {} files...", files.len());
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;

    for file in files {
        match file.extension().and_then(|ext| ext.to_str()) {
            Some("gml") | Some("xml") => {
                let mut reader = StreamingGmlReader::new(file)?;
                while let Some(geometry) = reader.next_geometry()? {
                    if let Some(bbox) = geometry.bounding_rect() {
                        min_x = min_x.min(bbox.min().x);
                        min_y = min_y.min(bbox.min().y);
                        max_x = max_x.max(bbox.max().x);
                        max_y = max_y.max(bbox.max().y);
                    }
                }
            },
            Some("geojson") | Some("json") => {
                // For GeoJSON, we'll still use the existing method as it's already efficient
                let geometries = load_geojson(file)?;
                for geometry in geometries {
                    if let Some(bbox) = geometry.bounding_rect() {
                        min_x = min_x.min(bbox.min().x);
                        min_y = min_y.min(bbox.min().y);
                        max_x = max_x.max(bbox.max().x);
                        max_y = max_y.max(bbox.max().y);
                    }
                }
            },
            _ => return Err("Unsupported file format".into())
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
pub fn create_partitions(
    extent: &Polygon<f64>,
    num_partitions: usize,
    crs: &CoordinateSystem
) -> Vec<Partition> {
    println!("Creating partitions using {}", crs.srid);
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


fn append_geometries_to_file(
    features: &[(Geometry<f64>, FeatureAttributes)],
    output_path: &Path,
    source_file: &Path,
) -> Result<(), Box<dyn Error>> {
    match source_file.extension().and_then(|ext| ext.to_str()) {
        Some("gml") | Some("xml") => {
            let file = if output_path.exists() {
                OpenOptions::new()
                    .write(true)
                    .append(true)
                    .open(output_path)?
            } else {
                let file = File::create(output_path)?;
                let mut writer = BufWriter::new(file);
                // Write GML header with namespaces
                writeln!(writer, r#"<?xml version="1.0" encoding="UTF-8"?>"#)?;
                writeln!(writer, r#"<gml:FeatureCollection"#)?;
                writeln!(writer, r#"    xmlns:gml="http://www.opengis.net/gml""#)?;
                writeln!(writer, r#"    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance""#)?;
                writeln!(writer, r#"    xsi:schemaLocation="http://www.opengis.net/gml http://schemas.opengis.net/gml/3.1.1/base/gml.xsd">"#)?;
                writer.flush()?;
                OpenOptions::new()
                    .write(true)
                    .append(true)
                    .open(output_path)?
            };

            let mut writer = BufWriter::new(file);

            for (geom, attrs) in features {
                // Start feature member
                writeln!(writer, "  <gml:featureMember>")?;
                writeln!(writer, "    <{}>", attrs.feature_type)?;

                // Write all properties
                for (key, value) in &attrs.properties {
                    writeln!(writer, "      <{}>{}</{}>", key, value, key)?;
                }

                // Write geometry based on type
                match geom {
                    Geometry::LineString(line) => {
                        writeln!(writer, "      <gml:LineString>")?;
                        writeln!(writer, "        <gml:coordinates>")?;
                        // Format coordinates with proper precision
                        let coords: String = line.points()
                            .map(|p| format!("{:.3},{:.3}", p.x(), p.y()))
                            .collect::<Vec<_>>()
                            .join(" ");
                        writeln!(writer, "          {}", coords)?;
                        writeln!(writer, "        </gml:coordinates>")?;
                        writeln!(writer, "      </gml:LineString>")?;
                    },
                    Geometry::Polygon(polygon) => {
                        writeln!(writer, "      <gml:Polygon>")?;
                        // Write exterior ring
                        writeln!(writer, "        <gml:exterior>")?;
                        writeln!(writer, "          <gml:LinearRing>")?;
                        writeln!(writer, "            <gml:coordinates>")?;
                        let coords: String = polygon.exterior().points()
                            .map(|p| format!("{:.3},{:.3}", p.x(), p.y()))
                            .collect::<Vec<_>>()
                            .join(" ");
                        writeln!(writer, "              {}", coords)?;
                        writeln!(writer, "            </gml:coordinates>")?;
                        writeln!(writer, "          </gml:LinearRing>")?;
                        writeln!(writer, "        </gml:exterior>")?;

                        // Write interior rings (holes)
                        for interior in polygon.interiors() {
                            writeln!(writer, "        <gml:interior>")?;
                            writeln!(writer, "          <gml:LinearRing>")?;
                            writeln!(writer, "            <gml:coordinates>")?;
                            let coords: String = interior.points()
                                .map(|p| format!("{:.3},{:.3}", p.x(), p.y()))
                                .collect::<Vec<_>>()
                                .join(" ");
                            writeln!(writer, "              {}", coords)?;
                            writeln!(writer, "            </gml:coordinates>")?;
                            writeln!(writer, "          </gml:LinearRing>")?;
                            writeln!(writer, "        </gml:interior>")?;
                        }
                        writeln!(writer, "      </gml:Polygon>")?;
                    },
                    _ => return Err("Unsupported geometry type".into()),
                }

                // Close feature tags
                writeln!(writer, "    </{}>", attrs.feature_type)?;
                writeln!(writer, "  </gml:featureMember>")?;
            }
            writer.flush()?;
            Ok(())
        },
        Some("geojson") | Some("json") => {
            let features: Vec<Feature> = features.iter().map(|(geom, attrs)| {
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
                            .map(|ring| ring.points().map(|p| vec![p.x(), p.y()]).collect())
                            .collect();
                        let mut rings = vec![exterior];
                        rings.extend(holes);
                        GeoJsonGeometry::new(geojson::Value::Polygon(rings))
                    },
                    _ => panic!("Unsupported geometry type"),
                };

                // Convert attributes to GeoJSON properties
                let mut properties = serde_json::Map::new();
                properties.insert(
                    "feature_type".to_string(),
                    serde_json::Value::String(attrs.feature_type.clone())
                );
                for (key, value) in &attrs.properties {
                    properties.insert(
                        key.clone(),
                        serde_json::Value::String(value.clone())
                    );
                }
                
                Feature {
                    bbox: None,
                    geometry: Some(geojson_geom),
                    id: None,
                    properties: Some(properties),
                    foreign_members: None,
                }
            }).collect();

            let mut existing_collection = if output_path.exists() {
                let file = File::open(output_path)?;
                let reader = BufReader::new(file);
                match GeoJson::from_reader(reader)? {
                    GeoJson::FeatureCollection(fc) => fc,
                    _ => FeatureCollection {
                        bbox: None,
                        features: Vec::new(),
                        foreign_members: None,
                    },
                }
            } else {
                FeatureCollection {
                    bbox: None,
                    features: Vec::new(),
                    foreign_members: None,
                }
            };

            existing_collection.features.extend(features);

            let file = File::create(output_path)?;
            serde_json::to_writer_pretty(file, &existing_collection)?;
            
            Ok(())
        },
        _ => Err("Unsupported file format".into()),
    }
}

fn flush_partitions_to_disk(
    partition_outputs: &mut Vec<PartitionOutput>,
    source_file: &Path,
    output_dir: &Path,
) -> Result<(), Box<dyn Error>> {
    let file_stem = source_file.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Get the same extension as the source file
    let extension = source_file.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("gml");

    for partition_output in partition_outputs.iter_mut() {
        if !partition_output.features.is_empty() {
            let output_path = output_dir.join(
                format!("{}_partition_{}.{}", file_stem, partition_output.id, extension)
            );
            
            append_geometries_to_file(&partition_output.features, &output_path, source_file)?;
            
            partition_output.features.clear();
        }
    }

    // If this is the last flush for a GML file, append the closing tag
    if extension == "gml" || extension == "xml" {
        for partition_output in partition_outputs.iter() {
            let output_path = output_dir.join(
                format!("{}_partition_{}.{}", file_stem, partition_output.id, extension)
            );
            if output_path.exists() {
                let file = OpenOptions::new()
                    .write(true)
                    .append(true)
                    .open(output_path)?;
                let mut writer = BufWriter::new(file);
                writeln!(writer, "</gml:FeatureCollection>")?;
                writer.flush()?;
            }
        }
    }
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


pub fn process_gis_files(
    files: Vec<PathBuf>, 
    num_partitions: Option<usize>,
    distinct: bool
) -> Result<(), Box<dyn Error>> {
    println!("\n=== Starting GIS Processing ===");
    
    // Validate CRS across all files
    let crs = validate_crs(&files)?
        .ok_or_else(|| "No CRS information found in input files".to_string())?;
    println!("Detected Coordinate Reference System: {}", crs.srid);
    
    // Create output directory
    let output_dir = Path::new("output");
    create_dir_all(output_dir)?;
    println!("Created output directory: {}", output_dir.display());

    let batch_size = 1000;

    // Calculate partitions based on distinct flag
    let partitions = if !distinct {
        // Calculate common extent and partitions for all files
        let common_extent = calculate_common_extent_streaming(&files, &crs)?;
        let partitions_count = num_partitions.unwrap_or_else(|| suggest_partitions(&files));
        create_partitions(&common_extent, partitions_count, &crs)
    } else {
        Vec::new() // Empty vec for distinct mode
    };

    // Process each file
    for (file_index, file) in files.iter().enumerate() {
        println!("\nProcessing file {}/{}: {}", file_index + 1, files.len(), file.display());
        
        // Get partitions for this file
        let file_partitions = if distinct {
            // Create new partitions for this file
            let file_extent = calculate_file_extent(file, &crs)?;
            let partitions_count = num_partitions.unwrap_or_else(|| suggest_partitions(&[file.clone()]));
            create_partitions(&file_extent, partitions_count, &crs)
        } else {
            // Use the common partitions
            partitions.clone()
        };

        // Initialize partition outputs
        let mut partition_outputs: Vec<PartitionOutput> = file_partitions.iter().enumerate()
            .map(|(id, _)| PartitionOutput { 
                id, 
                features: Vec::with_capacity(batch_size)
            })
            .collect();

        // Create new reader for feature processing
        let mut reader = StreamingGmlReader::new(file)?;
        let mut processed_count = 0;
        
        println!("Reading features in batches of {}", batch_size);

        // Process features in batches
        loop {
            let mut batch_features = Vec::with_capacity(batch_size);
            
            // Fill batch
            while batch_features.len() < batch_size {
                match reader.next_feature()? {
                    Some((geometry, attributes)) => {
                        batch_features.push((geometry, attributes));
                    },
                    None => break,
                }
            }

            if batch_features.is_empty() {
                break;
            }

            // Process batch
            for (geometry, attributes) in batch_features {
                processed_count += 1;
                if let Some(geom_bbox) = geometry.bounding_rect() {
                    for (partition_idx, partition) in file_partitions.iter().enumerate() {
                        if !partition.boundary.intersects(&geom_bbox) {
                            continue;
                        }
                        
                        match geometry {
                            Geometry::LineString(ref line) => {
                                if let Some(clipped_line) = clip_linestring(line, partition) {
                                    partition_outputs[partition_idx].features.push((
                                        Geometry::LineString(clipped_line),
                                        attributes.clone()
                                    ));
                                }
                            },
                            Geometry::Polygon(ref polygon) => {
                                if let Some(clipped_polygon) = clip_polygon(polygon, partition) {
                                    partition_outputs[partition_idx].features.push((
                                        Geometry::Polygon(clipped_polygon),
                                        attributes.clone()
                                    ));
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }

            // Flush full batches to disk
            flush_partitions_to_disk(&mut partition_outputs, file, output_dir)?;

            if processed_count % 1000 == 0 {
                println!("Processed {} features", processed_count);
            }
        }

        // Final flush for any remaining features
        flush_partitions_to_disk(&mut partition_outputs, file, output_dir)?;
        println!("Completed processing {} features from file", processed_count);
    }

    println!("\n=== Processing Complete ===");
    println!("Output files can be found in: {}", output_dir.display());
    Ok(())
}

pub fn calculate_file_extent(
    file: &Path,
    crs: &CoordinateSystem
) -> Result<Polygon<f64>, Box<dyn Error>> {
    println!("Calculating extent for file {} using {}", file.display(), crs.srid);
    let mut min_x = f64::MAX;
    let mut min_y = f64::MAX;
    let mut max_x = f64::MIN;
    let mut max_y = f64::MIN;
    let mut reader = StreamingGmlReader::new(file)?;
    println!("Created reader for file: {}", file.display());

    println!("Calculating extent for file: {}", file.display());
    let mut geometry_count = 0;

    while let Some(geometry) = reader.next_geometry()? {
        geometry_count += 1;
        if geometry_count % 100000 == 0 {
            println!("Processed {} geometries for extent calculation", geometry_count);
        }

        if let Some(bbox) = geometry.bounding_rect() {
            min_x = min_x.min(bbox.min().x);
            min_y = min_y.min(bbox.min().y);
            max_x = max_x.max(bbox.max().x);
            max_y = max_y.max(bbox.max().y);
        }
    }

    println!("Extent calculation complete. Processed {} geometries", geometry_count);
    println!("Extent bounds: ({:.2}, {:.2}) to ({:.2}, {:.2})", 
             min_x, min_y, max_x, max_y);

    Ok(Polygon::new(
        LineString::new(vec![
            (min_x, min_y).into(),
            (max_x, min_y).into(),
            (max_x, max_y).into(),
            (min_x, max_y).into(),
            (min_x, min_y).into(),
        ]),
        vec![]
    ))
}