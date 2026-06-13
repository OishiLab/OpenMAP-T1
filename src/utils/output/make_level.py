import os

import numpy as np
import pandas as pd

from utils.output.output_space import save_label_volume

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEVEL_DIR = os.path.join(PROJECT_ROOT, "level")


def create_parcellated_images(output, output_dir, basename, odata, data, output_ext=".nii.gz", output_space="native"):
    """
    Creates parcellated segmentation images for each specified level based on a mapping
    read from CSV files. The mapping is recalculated for each level using the original image labels.

    Parameters:
      output (numpy.ndarray): The image data array after calling .get_fdata() (contains Type1_Level5 labels).
      output_dir (str): The output directory path.
      basename (str): The base name for the output files.
      affine (numpy.ndarray, optional): The affine matrix for the image. If None, np.eye(4) is used.

    The CSV files are read from:
      ../level/Level_ROI_No.csv and ../level/Level_ROI_Name.csv
    The mapping is created from df_no where:
      - Keys: values in the 'Type1_Level5' column (original labels)
      - Values: values in the column corresponding to the current level (e.g., 'Type1_Level1', etc.)

    The output NIfTI files are saved as:
      os.path.join(output_dir, f"parcellated/{basename}_{level}{output_ext}")
    """

    # CSVファイルのパスを LEVEL_DIR を基準に作成
    df_no = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_No.csv"))
    df_name = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_Name.csv"))

    # List of target levels (exclude "Type1_Level5" since it is the input label type)
    all_level = [
        "Type1_Level1",
        "Type1_Level2",
        "Type1_Level3",
        "Type1_Level4",
        "Type2_Level1",
        "Type2_Level2",
        "Type2_Level3",
        "Type2_Level4",
        "Type2_Level5",
    ]

    # Process each target level
    for level in all_level:
        # Create mapping: original labels (from Type1_Level5) to new labels for the current level
        mapping = dict(zip(df_no["Type1_Level5"], df_no[level]))

        # Create a copy of the original image data to avoid in-place modification
        label = np.copy(output)

        # Apply mapping to the entire image data
        for old, new in mapping.items():
            label[label == old] = new

        save_label_volume(
            output_dir,
            "parcellated",
            basename,
            level,
            label,
            odata,
            data,
            output_ext,
            output_space,
        )
