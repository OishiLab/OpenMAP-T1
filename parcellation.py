import argparse
import glob
import os
import time
from functools import partial

import nibabel as nib
import numpy as np
import torch
from nibabel import processing
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


from utils.cropping import cropping
from utils.hemisphere import hemisphere
from utils.load_model import load_model
from utils.make_csv import make_csv
from utils.make_reface import make_reface
from utils.parcellation import parcellation
from utils.postprocessing import postprocessing
from utils.preprocessing import preprocessing
from utils.stripping import stripping


def create_parser():
    parser = argparse.ArgumentParser(
        description="Use this to run inference with OpenMAP-T1."
    )
    parser.add_argument(
        "-i",
        required=True,
        help="Input folder. Specifies the folder containing the input brain MRI images.",
    )
    parser.add_argument(
        "-o",
        required=True,
        help="Output folder. Difines the output folder where the results will be saved. If the specified folder does not exist, it will be automatically created.",
    )
    parser.add_argument(
        "-m",
        required=True,
        help="Folder of pretrained models. Indicates the location of the pretrained models to be used for processing.",
    )
    return parser.parse_args()


def main():
    print(
        "\n#######################################################################\n"
        "Please cite the following paper when using OpenMAP-T1:\n"
        "Kei Nishimaki, Kengo Onda, Kumpei Ikuta, Jill Chotiyanonta, Yuto Uchida, Hitoshi Iyatomi, Kenichi Oishi (2024).\n"
        "OpenMAP-T1: A Rapid Deep Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain.\n"
        "paper: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70063.\n"
        "Submitted for publication in the Human Brain Mapping.\n"
        "#######################################################################\n"
    )
    opt = create_parser()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cnet, ssnet, pnet_c, pnet_s, pnet_a, hnet_c, hnet_a = load_model(opt, device)

    print("load complete !!")
    pathes = sorted(glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True))

    for path in tqdm(pathes):
        save = os.path.splitext(os.path.basename(path))[0]
        output_dir = f"{opt.o}/{save}"
        os.makedirs(output_dir, exist_ok=True)

        odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))
        idata = processing.conform(
            odata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1
        )
        bdata = nib.squeeze_image(nib.as_closest_canonical(nib.load("meanbrain.nii")))
        bdata = processing.conform(
            bdata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1
        )
        vdata = nib.squeeze_image(nib.as_closest_canonical(nib.load("meanhead.nii")))
        vdata = processing.conform(
            vdata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1
        )

        odata, data, pixdim = preprocessing(path, save)
        cropped = cropping(data, cnet, device)
        stripped, shift, out_e = stripping(cropped, data, ssnet, device)
        parcellated = parcellation(stripped, pnet_c, pnet_s, pnet_a, device)
        separated = hemisphere(stripped, hnet_c, hnet_a, device)
        output = postprocessing(parcellated, separated, shift, device)

        lev5_df, lev4_df, lev3_df, lev2_df, lev1_df, qc_df = make_csv(output, save, pixdim)
        
        lev5_df.to_csv(os.path.join(output_dir, f"{save}_level5volume.csv"), index=False)
        lev4_df.to_csv(os.path.join(output_dir, f"{save}_level4volume.csv"), index=False)
        lev3_df.to_csv(os.path.join(output_dir, f"{save}_level3volume.csv"), index=False)
        lev2_df.to_csv(os.path.join(output_dir, f"{save}_level2volume.csv"), index=False)
        lev1_df.to_csv(os.path.join(output_dir, f"{save}_level1volume.csv"), index=False)
        qc_df.to_csv(os.path.join(output_dir, f"{save}_qc.csv"), index=False)
        
        ldata = nib.Nifti1Image(out_e.astype(np.uint16), affine=data.affine)
        ldata = processing.conform(
            ldata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1
        )
        header = odata.header
        reface = make_reface(idata, bdata, vdata, ldata)
        nii = nib.Nifti1Image(reface.astype(np.float32), affine=data.affine)
        nii = processing.conform(
            nii,
            out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]),
            voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]),
            order=1,
        )
        nib.save(nii, os.path.join(output_dir, f"{save}.nii"))

        nii = nib.Nifti1Image(output.astype(np.uint16), affine=data.affine)
        nii = processing.conform(
            nii,
            out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]),
            voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]),
            order=0,
        )
        nib.save(nii, os.path.join(output_dir, f"{save}_280.nii"))
        del odata, data
        os.remove(f"N4/{save}.nii")
    return


if __name__ == "__main__":
    main()
