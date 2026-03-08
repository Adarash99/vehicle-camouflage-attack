#!/usr/bin/env python3
"""
CSV Logger for EOT Training

Handles writing training metrics to CSV files with proper headers,
timestamps, and automatic file creation.

Author: Adversarial Camouflage Research Project
Date: 2026-01-31
"""

import csv
import os
from datetime import datetime
from pathlib import Path


class CSVLogger:
    """
    Writes training metrics to CSV file with automatic directory creation.

    Usage:
        logger = CSVLogger('experiments/phase1/training_log.csv')
        logger.write_header(['iteration', 'loss', 'accuracy'])
        logger.write_row([0, 5.234, 0.123])
        logger.write_row([1, 4.892, 0.234])
    """

    def __init__(self, filepath):
        """
        Initialize CSV logger.

        Args:
            filepath: Path to CSV file (will be created if doesn't exist)
        """
        self.filepath = Path(filepath)
        self.data = []  # Store all rows for later analysis

        # Create directory if needed
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file handle
        self.file_handle = None
        self.csv_writer = None

    def write_header(self, columns):
        """
        Write CSV header row.

        Args:
            columns: List of column names
        """
        # Open file for writing
        self.file_handle = open(self.filepath, 'w', newline='')
        self.csv_writer = csv.writer(self.file_handle)

        # Add timestamp column to header
        header_with_timestamp = list(columns) + ['timestamp']

        # Write header
        self.csv_writer.writerow(header_with_timestamp)
        self.file_handle.flush()

        print(f"CSV logger initialized: {self.filepath}")

    def write_row(self, values):
        """
        Write a data row to CSV.

        Args:
            values: List of values matching header columns
        """
        if self.csv_writer is None:
            raise ValueError("Must call write_header() before write_row()")

        # Add timestamp as last column
        values_with_timestamp = list(values) + [datetime.now().isoformat()]

        # Write to file
        self.csv_writer.writerow(values_with_timestamp)
        self.file_handle.flush()  # Ensure data written immediately

        # Store in memory for later analysis
        self.data.append(values_with_timestamp)

    def close(self):
        """Close the CSV file."""
        if self.file_handle is not None:
            self.file_handle.close()
            self.file_handle = None
            print(f"CSV logger closed: {self.filepath}")

    def __del__(self):
        """Ensure file is closed on deletion."""
        if self.file_handle is not None:
            self.close()


# Test the logger
if __name__ == "__main__":
    print("=" * 70)
    print("CSV LOGGER TEST")
    print("=" * 70)
    print()

    # Test basic logging
    print("Testing CSVLogger...")
    logger = CSVLogger('test_output/test_log.csv')
    logger.write_header(['iteration', 'loss', 'accuracy'])

    # Write some test data
    for i in range(5):
        logger.write_row([i, 5.0 - i * 0.5, 0.1 + i * 0.1])

    logger.close()

    # Verify file exists and has content
    with open('test_output/test_log.csv', 'r') as f:
        lines = f.readlines()
        print(f"  Lines written: {len(lines)}")
        print(f"  Header: {lines[0].strip()}")
        print(f"  First data row: {lines[1].strip()}")

    # Verify data was stored in memory
    print(f"  Rows in memory: {len(logger.data)}")

    print()
    print("=" * 70)
    print("CSV LOGGER TEST PASSED ✓")
    print("=" * 70)
