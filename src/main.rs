use clap::{Command, Arg};
use std::path::PathBuf;
use gis_partitioner::process_files;  // Update this import

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
        .arg(
            Arg::new("distinct")
                .long("distinct")
                .action(clap::ArgAction::SetTrue)
                .help("Process files separately but with the same partition lines"),
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

    // Check if distinct flag is set
    let distinct = matches.contains_id("distinct");

    // Validate that input files exist
    for file in &files {
        if !file.exists() {
            eprintln!("Error: File not found: {}", file.display());
            std::process::exit(1);
        }
    }

    // Process the files using the new combined function
    let result = process_files(files, num_partitions, distinct);

    match result {
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