import numpy as np
import pandas as pd
from collections import defaultdict

def change_level(df, level="Type1_Level1", sulcus=True):
    ROI_number = pd.read_csv("level/Level_ROI_No.csv")
    ROI_name = pd.read_csv("level/Level_ROI_Name.csv")
   
    if sulcus == False:
        tmp = ROI_number["Type1_Level2"]
        ROI_number = ROI_number[tmp!=18]
        ROI_number = ROI_number[tmp!=19]
        ROI_name = ROI_name[tmp!=18]
        ROI_name = ROI_name[tmp!=19]
    data = dict(zip(ROI_number["ROI"], ROI_number[level]))
    level_dict = defaultdict(list)
    for key, value in data.items():
        level_dict[str(value)].append(key)
    
    change_df_list = []
    for i, (key, value) in enumerate(level_dict.items()):
        name = ROI_name[level].unique()[i]
        change_df_list.append(df[value].sum(axis=1).rename(name))
    
    change_df = pd.concat(change_df_list, axis=1)
    return change_df

def mark_cell(x, region_stats, col):
    if x < region_stats[col]['min'] or x > region_stats[col]['max']:
        return f"#{x}"
    elif x < region_stats[col]['mean'] - 3 * region_stats[col]['std'] or x > region_stats[col]['mean'] + 3 * region_stats[col]['std']:
        return f"*{x}"
    else:
        return x

def make_csv(parcellation, save, pixdim):
    csv_path = "level/Level5.txt"
    df = (
        pd.read_table(csv_path, names=["number", "region"])
        .astype("str")
        .set_index("number")
    )
    for i in range(280):
        i += 1
        volume = np.count_nonzero(parcellation == i)
        df.loc[str(i), save] = volume
        
    lev5_df = df.set_index("region").T.reset_index().rename(columns={"index": "uid"})
    lev4_df = change_level(lev5_df, level="Type1_Level4")
    lev3_df = change_level(lev5_df, level="Type1_Level3")
    lev2_df = change_level(lev5_df, level="Type1_Level2")
    lev1_df = change_level(lev5_df, level="Type1_Level1")
    
    region_stats = {
        'Telencephalon_L': {'min': 321688.750000, 'max': 694587.707520, 'mean': 460443.2969543, 'std': 51764.523737},
        'Telencephalon_R': {'min': 326145.073700, 'max': 706307.939926, 'mean': 465247.798455, 'std': 52265.325944},
        'Diencephalon_L': {'min': 5287.283532, 'max': 11819.827568, 'mean': 8085.498710, 'std': 813.392248},
        'Diencephalon_R': {'min': 5635.978500, 'max': 11917.817848, 'mean': 8209.691222, 'std': 826.833165},
        'Mesencephalon': {'min': 6337.000000, 'max': 13937.617500, 'mean': 9549.460494, 'std': 1094.829481},
        'Metencephalon': {'min': 93660.261081, 'max': 195092.666076, 'mean': 140629.501438, 'std': 14361.567933},
        'Myelencephalon': {'min': 3114.067841, 'max': 6801.250000, 'mean': 4835.313220, 'std': 513.705261},
        'CSF': {'min': 31516.571045, 'max': 334491.264954, 'mean': 156960.761099, 'std': 40687.350710}
    }
    
    qc_df = lev1_df.copy()
    for col in qc_df.columns:
        qc_df[col] = qc_df[col].apply(mark_cell, args=(region_stats, col,))
        
    message = "In the csv file, values outside the range of the maximum and minimum in the trained dataset are marked with '#', while values outside the range of the mean plus or minus three standard deviations are marked with '*'."
    num_columns = len(qc_df.columns)
    message = [message] + [""] * (num_columns - 1) 

    qc_df.loc[len(qc_df)] = pd.Series(message, index=qc_df.columns)
    lev1_df["pixdim"] = [pixdim]
    
    return lev5_df, lev4_df, lev3_df, lev2_df, lev1_df, qc_df
