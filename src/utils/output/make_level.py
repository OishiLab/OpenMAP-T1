from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils.output.make_csv import get_level_dir
from utils.output.output_space import save_label_volume


def create_parcellated_images(
    output,
    output_dir,
    basename,
    odata,
    data,
    output_ext: str = ".nii.gz",
    output_space: str = "native",
    mode: str = "t1",
):
    """Create parcellated segmentation images for each hierarchical level."""
    level_dir = get_level_dir(mode)
    df_no = pd.read_csv(os.path.join(level_dir, "Level_ROI_No.csv"))

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

    for level in all_level:
        mapping = dict(zip(df_no["Type1_Level5"], df_no[level]))
        label = np.copy(output)
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
