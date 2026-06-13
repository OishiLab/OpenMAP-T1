import os

import numpy as np
import pandas as pd

# Total labeled brain volume below this threshold (mm³) is treated as a clear failure.
DEFAULT_MIN_BRAIN_VOLUME_MM3 = 10_000


def total_brain_volume_mm3(parcellation: np.ndarray) -> int:
    """Return the total labeled brain volume in mm³ on the 1 mm isotropic grid."""
    return int(np.count_nonzero(parcellation))


class ProcessingLog:
    """Collect failed, skipped, and suspiciously low-volume cases for batch QC."""

    def __init__(self, min_brain_volume_mm3: int = DEFAULT_MIN_BRAIN_VOLUME_MM3):
        self.min_brain_volume_mm3 = min_brain_volume_mm3
        self.records = []

    def add_failed(self, input_path: str, case_id: str, error_message: str) -> None:
        self.records.append(
            {
                "case_id": case_id,
                "input_path": input_path,
                "status": "failed",
                "reason": error_message,
                "total_brain_volume_mm3": "",
            }
        )

    def add_skipped(self, input_path: str, case_id: str, skip_reason: str) -> None:
        self.records.append(
            {
                "case_id": case_id,
                "input_path": input_path,
                "status": "skipped",
                "reason": skip_reason,
                "total_brain_volume_mm3": "",
            }
        )

    def add_low_volume(self, input_path: str, case_id: str, total_volume: int) -> None:
        self.records.append(
            {
                "case_id": case_id,
                "input_path": input_path,
                "status": "low_volume",
                "reason": (
                    f"Total brain volume ({total_volume} mm³) is below "
                    f"threshold ({self.min_brain_volume_mm3} mm³)."
                ),
                "total_brain_volume_mm3": total_volume,
            }
        )

    def check_low_volume(self, input_path: str, case_id: str, parcellation: np.ndarray) -> bool:
        total_volume = total_brain_volume_mm3(parcellation)
        if total_volume < self.min_brain_volume_mm3:
            self.add_low_volume(input_path, case_id, total_volume)
            return True
        return False

    def save(self, output_folder: str, filename: str = "failed_cases.csv") -> None:
        if not self.records:
            print("No failed, skipped, or low-volume cases to report.")
            return

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        pd.DataFrame(self.records).to_csv(output_path, index=False)
        print(f"Wrote {len(self.records)} issue(s) to {output_path}")
