"""Utilities for T5 run processing and lookup tables."""

from __future__ import annotations

import csv
from pathlib import Path


def get_t5_to_tank_time_offset(run_number: int) -> float:
    """Return the T5-to-tank offset in ns for a given run number.

    The lookup file stores rows as:
        run,offset_ns,error_ns
    and the offset value is the second column in the row matching the run number.
    """
    lookup_path = Path(__file__).resolve().parent / "data" / "t5_tank_offset_lookup.csv"

    with lookup_path.open("r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Lookup file is empty: {lookup_path}")

        for row in reader:
            if not row:
                continue
            if int(row[0]) == run_number:
                return float(row[1])

    raise ValueError(f"No T5-to-tank offset found for run number {run_number} in {lookup_path}")


if __name__ == "__main__":
    mean = get_t5_to_tank_time_offset(1610)
    print(f"T5-to-tank time offset for run 1610: {mean} ns")
    