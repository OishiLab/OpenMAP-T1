import os
from collections import defaultdict

import numpy as np
import pandas as pd

# このファイルのあるディレクトリの絶対パスを取得し、そこから level ディレクトリへの絶対パスを作成
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LEVEL_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "..", "..", "level"))


def change_level(df, level="Type1_Level1", sulcus=True):
    """
    Change the level of the given DataFrame based on specified ROI levels.

    Parameters:
    df (pd.DataFrame): The input DataFrame to be modified.
    level (str): The level to which the DataFrame should be changed. Default is "Type1_Level1".
    sulcus (bool): A flag indicating whether to include sulcus regions. Default is True.

    Returns:
    pd.DataFrame: The modified DataFrame with the specified level changes applied.
    """
    # LEVEL_DIR を基準に CSV ファイルの絶対パスを作成
    ROI_number = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_No.csv"))
    ROI_name = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_Name.csv"))

    if sulcus == False:
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
    for i, (key, value) in enumerate(level_dict.items()):
        name = ROI_name[level].unique()[i]
        change_df_list.append(df[value].sum(axis=1).rename(name))

    change_df = pd.concat(change_df_list, axis=1)
    return change_df


def make_csv(parcellation, output_dir, basename):
    """
    Generates multiple CSV files containing volume data for different levels of parcellation.

    Parameters:
    parcellation (numpy.ndarray): The parcellation data array where each unique integer represents a different region.
    output_dir (str): The directory where the output CSV files will be saved.
    basename (str): The base name for the output CSV files.

    Returns:
    pandas.DataFrame: The DataFrame containing volume data for Type1_Level5.
    """
    # LEVEL_DIR を基準にテキストファイルの絶対パスを作成
    csv_path = os.path.join(LEVEL_DIR, "Level5.txt")
    df_Type1_level5 = (
        pd.read_table(csv_path, names=["number", "region"]).astype("str").set_index("number")
    )
    for i in range(1, 275):
        volume = np.count_nonzero(parcellation == i)
        df_Type1_level5.loc[str(i), basename] = volume

    df_Type1_level5 = df_Type1_level5.set_index("region").T.reset_index(drop=True)
    df_Type1_level4 = change_level(df_Type1_level5, level="Type1_Level4")
    df_Type1_level3 = change_level(df_Type1_level5, level="Type1_Level3")
    df_Type1_level2 = change_level(df_Type1_level5, level="Type1_Level2")
    df_Type1_level1 = change_level(df_Type1_level5, level="Type1_Level1")

    df_Type2_level5 = change_level(df_Type1_level5, level="Type2_Level5")
    df_Type2_level4 = change_level(df_Type1_level5, level="Type2_Level4")
    df_Type2_level3 = change_level(df_Type1_level5, level="Type2_Level3")
    df_Type2_level2 = change_level(df_Type1_level5, level="Type2_Level2")
    df_Type2_level1 = change_level(df_Type1_level5, level="Type2_Level1")

    os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
    df_Type1_level5.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type1_Level5.csv"), index=False
    )
    df_Type1_level4.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type1_Level4.csv"), index=False
    )
    df_Type1_level3.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type1_Level3.csv"), index=False
    )
    df_Type1_level2.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type1_Level2.csv"), index=False
    )
    df_Type1_level1.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type1_Level1.csv"), index=False
    )

    df_Type2_level5.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type2_Level5.csv"), index=False
    )
    df_Type2_level4.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type2_Level4.csv"), index=False
    )
    df_Type2_level3.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type2_Level3.csv"), index=False
    )
    df_Type2_level2.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type2_Level2.csv"), index=False
    )
    df_Type2_level1.to_csv(
        os.path.join(output_dir, f"csv/{basename}_Type2_Level1.csv"), index=False
    )

    df = pd.read_csv(os.path.join(output_dir, f"csv/{basename}_Type1_Level5.csv")).iloc[:,1:]
    sulcus = [249, 250, 251, 252, 255, 256]
    sylvianFissure = [253, 254]
    sum_sulcus = df.iloc[0, sulcus].sum()
    sum_syl = df.iloc[0, sylvianFissure].sum()
    syl_ratio = sum_syl / sum_sulcus if sum_sulcus != 0 else None
    formula = ["(Sylvian Fissure L+R)/(Sulcus L+R)", "(Frontal Sulcus LR)+(Central Sulcus LR)+(Parietal Sulcus LR)"]

    sylvianRatio = pd.DataFrame({
        "SylvianRatio": [syl_ratio],
        "SylvianFissure_L+R": [sum_syl],
        "Sulcus_L+R": [sum_sulcus],
        "SylvianRatio_Caluclation_Formula":[formula[0]],
        "Sulcus_L+R_Caluclation_Formula":[formula[1]],
    })
    sylvianRatio.to_csv(os.path.join(output_dir, f"csv/{basename}_SylvianRatio.csv"), index=False)

    return df_Type1_level5
