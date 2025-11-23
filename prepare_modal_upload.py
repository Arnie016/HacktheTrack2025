#!/usr/bin/env python3
"""Prepare code for Modal upload by copying only needed files, excluding cache."""
import shutil
import os
from pathlib import Path

def prepare_upload():
    """Copy source files to a temp directory, excluding __pycache__ and other cache files."""
    base_dir = Path(__file__).parent
    temp_dir = base_dir / ".modal_upload"
    
    # Clean temp directory
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Directories to copy
    dirs_to_copy = ["src", "notebooks", "scripts", "templates"]
    
    # Files to copy
    files_to_copy = [
        "requirements.txt",
        "app.py",
        "grcup_modal.py",
    ]
    
    # Copy directories, excluding __pycache__
    for dir_name in dirs_to_copy:
        src_dir = base_dir / dir_name
        if not src_dir.exists():
            print(f"⚠ Warning: {dir_name} doesn't exist")
            continue
        
        dst_dir = temp_dir / dir_name
        shutil.copytree(
            src_dir,
            dst_dir,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "*.pyc"),
            dirs_exist_ok=False,
        )
        print(f"✓ Copied {dir_name}")
    
    # Copy individual files
    for file_name in files_to_copy:
        src_file = base_dir / file_name
        if src_file.exists():
            shutil.copy2(src_file, temp_dir / file_name)
            print(f"✓ Copied {file_name}")
        else:
            print(f"⚠ Warning: {file_name} doesn't exist")
    
    # Copy Race directories (just the needed files)
    for race_dir in ["Race 1", "Race 2"]:
        src_race = base_dir / race_dir
        if not src_race.exists():
            continue
        
        dst_race = temp_dir / race_dir
        dst_race.mkdir()
        
        # Copy only the files we need
        race_files = [
            "03_Provisional Results_Race 1_Anonymized.CSV",
            "03_Results GR Cup Race 2 Official_Anonymized.CSV",
            "03_Provisional Results_Race 2_Anonymized.CSV",
            "23_AnalysisEnduranceWithSections_Race 1_Anonymized.CSV",
            "23_AnalysisEnduranceWithSections_Race 2_Anonymized.CSV",
            "26_Weather_Race 1_Anonymized.CSV",
            "26_Weather_Race 2_Anonymized.CSV",
            "vir_lap_end_R1.csv",
            "vir_lap_start_R1.csv",
            "vir_lap_time_R1.csv",
            "vir_lap_end_R2.csv",
            "vir_lap_start_R2.csv",
            "vir_lap_time_R2.csv",
            "R1_telemetry_features.csv",
            "R2_telemetry_features.csv",
        ]
        
        for race_file in race_files:
            src_file = src_race / race_file
            if src_file.exists():
                shutil.copy2(src_file, dst_race / race_file)
                print(f"✓ Copied {race_dir}/{race_file}")
    
    print(f"\n✓ Upload preparation complete: {temp_dir}")
    print("Now update grcup_modal.py to use .modal_upload/ instead of current dir")

if __name__ == "__main__":
    prepare_upload()

