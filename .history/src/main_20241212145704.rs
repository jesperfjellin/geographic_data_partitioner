use clap::{Command, Arg};
use std::path::PathBuf;
mod partitioner;

fn main() {
    let matches = Command::new("GIS Partitioner")
        .version("1.0")
        .author("Jesper Fjellin")
        .about("Partitions GIS files into smaller chunks while maintaining geometry validity")
        .arg(
            Arg::new("files")
                .short('f')
                .long("files")
                .num_args(1..)
                .required(true)
                .help("Input GIS files to partition"),
        )
        .arg(
            Arg::new("partitions")
                .short('p')
                .long("partitions")
                .num_args(1)
                .help("Number of partitions (optional, will be calculated if not provided)"),
        )
        .get_matches();

    // Get input files
    let files: Vec<PathBuf> = matches
        .get_many::<String>("files")
        .unwrap()
        .map(|s| PathBuf::from(s))
        .collect();

    // Get number of partitions if provided
    let num_partitions = matches
        .get_one::<String>("partitions")
        .map(|p| p.parse::<usize>().expect("Invalid number of partitions"));

    // Validate that input files exist
    for file in &files {
        if !file.exists() {
            eprintln!("Error: File not found: {}", file.display());
            std::process::exit(1);
        }
    }

    // Process the files
    match partitioner::process_gis_files(files, num_partitions) {
        Ok(_) => println!("Processing completed successfully"),
        Err(e) => {
            eprintln!("Error processing files: {}", e);
            if e.to_string().contains("CRS mismatch") {
                eprintln!("Please ensure all input files use the same coordinate reference system.");
            }
            std::process::exit(1);
        }
    }
}