import os
import subprocess
from pathlib import Path

# DICOM root directory (your local DICOM folder)
DICOM_ROOT = Path(r"C:\Users\AV75950\Documents\ADNI")

# Output directory (where converted NIfTI files will be saved)
OUTPUT_ROOT = Path(r"C:\Users\AV75950\Documents\ADNI_NIfTI")

# Path to the dcm2niix executable
DCM2NIIX_PATH = r"C:\Users\AV75950\Downloads\dcm2niix_win\dcm2niix.exe"

# Iterate over each DICOM series folder
for subject_folder in DICOM_ROOT.glob("*/MPRAGE/*"):
    if subject_folder.is_dir():
        subject_id = subject_folder.parts[-3]  # e.g., 002_S_0729
        scan_date = subject_folder.parts[-1]   # e.g., 2006-08-02_07_02_00.0

        output_dir = OUTPUT_ROOT / subject_id
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            DCM2NIIX_PATH,
            "-z", "y",         # gzip compression
            "-f", f"{subject_id}_{scan_date}",
            "-o", str(output_dir),
            str(subject_folder)
        ]

        print("Running:", " ".join(command))
        subprocess.run(command)
