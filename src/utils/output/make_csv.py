from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

LEVEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "level"))

# Frontal, central, and parietal sulci (external/internal), excluding the Sylvian fissure (used in OpenMAP-T1c).
SULCUS_LABELS = (250, 251, 252, 253, 254, 255, 256, 257, 260, 261, 262, 263)
SYLVIAN_LABELS = (258, 259)


def get_level_dir(mode: str = "t1") -> str:
    subdir = "OpenMAP-T1c" if mode == "t1c" else "OpenMAP-T1"
    candidate = os.path.join(LEVEL_ROOT, subdir)
    if os.path.isdir(candidate):
        return candidate
    return LEVEL_ROOT


def change_level(df, level="Type1_Level1", sulcus=True, mode: str = "t1"):
    """Aggregate Type1_Level5 region columns into a coarser level."""
    level_dir = get_level_dir(mode)
    ROI_number = pd.read_csv(os.path.join(level_dir, "Level_ROI_No.csv"))
    ROI_name = pd.read_csv(os.path.join(level_dir, "Level_ROI_Name.csv"))

    if sulcus is False:
        tmp = ROI_number["Type1_Level2"]
        ROI_number = ROI_number[tmp != 18]
        ROI_number = ROI_number[tmp != 19]
        ROI_name = ROI_name[tmp != 18]
        ROI_name = ROI_name[tmp != 19]
    data = dict(zip(ROI_number["ROI"], ROI_number[level]))
    level_dict = defaultdict(list)
    for key, value in data.items():
        level_dict[str(value)].append(key)

    change_df_list = []
    for i, (_key, value) in enumerate(level_dict.items()):
        name = ROI_name[level].unique()[i]
        change_df_list.append(df[value].sum(axis=1).rename(name))

    return pd.concat(change_df_list, axis=1)


def _write_sylvian_ratio(parcellation: np.ndarray, output_dir: str, basename: str) -> None:
    sum_sulcus = int(sum(np.count_nonzero(parcellation == label) for label in SULCUS_LABELS))
    sum_syl = int(sum(np.count_nonzero(parcellation == label) for label in SYLVIAN_LABELS))
    ratio = sum_syl / sum_sulcus if sum_sulcus else None
    table = pd.DataFrame(
        {
            "SylvianRatio": [ratio],
            "SylvianFissure_L+R": [sum_syl],
            "Sulcus_L+R": [sum_sulcus],
            "SylvianRatio_Caluclation_Formula": ["(Sylvian Fissure L+R)/(Sulcus L+R)"],
            "Sulcus_L+R_Caluclation_Formula": ["(Frontal Sulcus LR)+(Central Sulcus LR)+(Parietal Sulcus LR)"],
        }
    )
    table.to_csv(os.path.join(output_dir, "csv", f"{basename}_SylvianRatio.csv"), index=False)


def make_csv(parcellation, output_dir, basename, mode: str = "t1"):
    """Generate CSV files containing regional volumes (mm³ on 1 mm grid) across all hierarchy levels."""
    level_dir = get_level_dir(mode)
    csv_path = os.path.join(level_dir, "Type1Level5.txt")
    if not os.path.isfile(csv_path):
        csv_path = os.path.join(level_dir, "Level5.txt")

    df_Type1_level5 = (
        pd.read_table(csv_path, names=["number", "region"]).astype("str").set_index("number")
    )
    for i in range(1, 281):
        volume = np.count_nonzero(parcellation == i)
        df_Type1_level5.loc[str(i), basename] = volume

    df_Type1_level5 = df_Type1_level5.set_index("region").T.reset_index(drop=True)
    levels = {
        "Type1_Level5": df_Type1_level5,
        "Type1_Level4": change_level(df_Type1_level5, level="Type1_Level4", mode=mode),
        "Type1_Level3": change_level(df_Type1_level5, level="Type1_Level3", mode=mode),
        "Type1_Level2": change_level(df_Type1_level5, level="Type1_Level2", mode=mode),
        "Type1_Level1": change_level(df_Type1_level5, level="Type1_Level1", mode=mode),
        "Type2_Level5": change_level(df_Type1_level5, level="Type2_Level5", mode=mode),
        "Type2_Level4": change_level(df_Type1_level5, level="Type2_Level4", mode=mode),
        "Type2_Level3": change_level(df_Type1_level5, level="Type2_Level3", mode=mode),
        "Type2_Level2": change_level(df_Type1_level5, level="Type2_Level2", mode=mode),
        "Type2_Level1": change_level(df_Type1_level5, level="Type2_Level1", mode=mode),
    }

    os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
    for level_name, frame in levels.items():
        frame.to_csv(os.path.join(output_dir, "csv", f"{basename}_{level_name}.csv"), index=False)

    if mode == "t1c":
        _write_sylvian_ratio(parcellation, output_dir, basename)

    return df_Type1_level5
