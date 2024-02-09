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
from utils.make_csv import make_csv
from utils.network import UNet
from utils.parcellation import parcellation
from utils.postprocessing import postprocessing
from utils.preprocessing import preprocessing
from utils.stripping import stripping
from utils.load_model import load_model

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", help="input folder")
    parser.add_argument("-o", help="output folder")
    parser.add_argument("-m", help="model path")
    return parser.parse_args()

def main():
    print(
        "\n#######################################################################\n"
        "Please cite the following paper when using OpenMAP-T1:\n"
        "Kei Nishimaki, Kengo Onda, Kumpei Ikuta, Yuto Uchida, Hitoshi Iyatomi, Kenichi Oishi (2024).\n"
        "OpenMAP-T1: A Rapid Deep Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain.\n"
        "paper: https://www.medrxiv.org/content/10.1101/2024.01.18.24301494v1.\n"
        "#######################################################################\n"
        )
    opt = create_parser()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    cnet, ssnet, pnet_c, pnet_s, pnet_a, hnet = load_model(opt, device)

    print("load complete !!")
    pathes = sorted(glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True))

    for path in tqdm(pathes):
        save = os.path.splitext(os.path.basename(path))[0]
        output_dir = f"{opt.o}/{save}"
        os.makedirs(output_dir, exist_ok=True)

        odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))
        nii = nib.Nifti1Image(odata.get_fdata().astype(np.float32), affine=odata.affine)
        nib.save(nii, os.path.join(output_dir, f"{save}.nii"))

        odata, data = preprocessing(path, save)
        cropped = cropping(data, cnet, device)
        stripped = stripping(cropped, data, ssnet, device)

        parcellated = parcellation(stripped, pnet_c, pnet_s, pnet_a, device)
        separated = hemisphere(stripped, hnet, device)
        output = postprocessing(parcellated, separated, device)
        
        df = make_csv(output, save)
        df.to_csv(os.path.join(output_dir, f"{save}_volume.csv"), index=False)
        
        nii = nib.Nifti1Image(output.astype(np.uint16), affine=data.affine)
        header = odata.header
        nii = processing.conform(nii, out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]), voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]), order=0)
        nib.save(nii, os.path.join(output_dir, f"{save}_280.nii"))
        os.remove(f"N4/{save}.nii")
    return




if __name__ == "__main__":
    main()