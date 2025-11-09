"""
Extract run numbers and step counts from a specified log file.

Usage:
    python extract_runs.py <log_file_path>

Arguments:
    log_file_path (str): The path to the log file to process.

Returns
-------
    List[str]: A list of strings, each containing the run number and step count.
"""

import re
import sys
from pathlib import Path


def extract_runs_and_steps(log_file_path: str) -> list[str]:
    """
    Extract run numbers and step counts from a log file.

    Args:
        log_file_path (str): The path to the log file to process.

    Returns
    -------
        List[str]: A list of strings, each containing the run number and step count.
    """
    with Path(log_file_path).open() as file:
        lines = file.readlines()

    pattern = re.compile(r"Run (\d+): (\d+) steps")
    results = []

    for line in lines:
        match = pattern.search(line)
        if match:
            run, steps = match.groups()
            results.append(f"Run {run}: {steps} steps")

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_runs.py <log_file_path>")
        sys.exit(1)

    log_file_path = sys.argv[1]
    extracted_data = extract_runs_and_steps(log_file_path)
    for entry in extracted_data:
        print(entry)
