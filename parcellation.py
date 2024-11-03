import argparse
import glob
import os
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
from utils.parcellation import parcellation
from utils.postprocessing import postprocessing
from utils.preprocessing import preprocessing
from utils.stripping import stripping


def create_parser():
    """
    Creates and returns the argument parser for the script.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Use this to run inference with OpenMAP-T1.")
    parser.add_argument(
        "-i",
        required=True,
        help="Input folder. Specifies the folder containing the input brain MRI images.",
    )
    parser.add_argument(
        "-o",
        required=True,
        help="Output folder. Defines the output folder where the results will be saved. If the specified folder does not exist, it will be automatically created.",
    )
    parser.add_argument(
        "-m",
        required=True,
        help="Folder of pretrained models. Indicates the location of the pretrained models to be used for processing.",
    )
    return parser.parse_args()


def main():
    """
    Main function to execute the OpenMAP-T1 parcellation process.
    This function performs the following steps:
    1. Prints a citation message for the OpenMAP-T1 paper.
    2. Parses command-line arguments.
    3. Determines the device to use (CUDA, MPS, or CPU).
    4. Loads the pretrained models.
    5. Retrieves the list of input NIfTI files.
    6. Iterates over each input file and performs the following operations:
        a. Extracts the base name of the file.
        b. Creates the output directory for the current file.
        c. Loads the input image, converts it to canonical form, and squeezes it.
        d. Creates a new NIfTI image with the data converted to float32.
        e. Saves the new NIfTI image to the output directory.
        f. Preprocesses the input image.
        g. Crops the image using the cropping network.
        h. Strips the image using the stripping network.
        i. Parcellates the stripped image using the parcellation networks.
        j. Separates the hemispheres using the hemisphere networks.
        k. Postprocesses the parcellated and separated image.
        l. Generates a CSV file with volume information and saves it.
        m. Creates a new NIfTI image with the processed output and saves it.
        n. Cleans up temporary files.
    Returns:
        None
    """

    print(
        "\n#######################################################################\n"
        "Please cite the following paper when using OpenMAP-T1:\n"
        "Kei Nishimaki, Kengo Onda, Kumpei Ikuta, Jill Chotiyanonta, Yuto Uchida, Hitoshi Iyatomi, Kenichi Oishi (2024).\n"
        "OpenMAP-T1: A Rapid Deep Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain.\n"
        "paper: https://www.medrxiv.org/content/10.1101/2024.01.18.24301494v1.\n"
        "#######################################################################\n"
    )
    # Parse command-line arguments
    opt = create_parser()

    # Determine the device to use (CUDA, MPS, or CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load the pretrained models
    cnet, ssnet, pnet_c, pnet_s, pnet_a, hnet_c, hnet_a = load_model(opt, device)
    print("load complete !!")

    # Get the list of input files
    pathes = sorted(glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True)) + sorted(
        glob.glob(os.path.join(opt.i, "**/*.nii.gz"), recursive=True)
    )

    for path in tqdm(pathes):
        # Extract the base name of the file (without extension)
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename.endswith(".nii"):
            basename = os.path.splitext(basename)[0]

        # Create the output directory for the current file
        output_dir = os.path.join(opt.o, basename)
        os.makedirs(output_dir, exist_ok=True)

        # Load the input image, convert it to canonical form, and squeeze it
        odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))

        # Create a new NIfTI image with the data converted to float32
        nii = nib.Nifti1Image(odata.get_fdata().astype(np.float32), affine=odata.affine)

        # Save the new NIfTI image to the output directory
        nib.save(nii, os.path.join(output_dir, f"{basename}.nii"))

        # Preprocess the input image
        odata, data = preprocessing(path, output_dir, basename)

        # Crop the image using the cropping network
        cropped = cropping(data, cnet, device)

        # Strip the image using the stripping network
        stripped, shift = stripping(cropped, data, ssnet, device)

        # Parcellate the stripped image using the parcellation networks
        parcellated = parcellation(stripped, pnet_c, pnet_s, pnet_a, device)

        # Separate the hemispheres using the hemisphere networks
        separated = hemisphere(stripped, hnet_c, hnet_a, device)

        # Postprocess the parcellated and separated image
        output = postprocessing(parcellated, separated, shift, device)

        # Generate a CSV file with volume information and save it
        df = make_csv(output, output_dir, basename)

        # Create a new NIfTI image with the processed output and save it
        nii = nib.Nifti1Image(output.astype(np.uint16), affine=data.affine)
        header = odata.header
        nii = processing.conform(
            nii,
            out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]),
            voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]),
            order=0,
        )
        nib.save(nii, os.path.join(output_dir, f"{basename}_280.nii"))

        # Clean up temporary files
        del odata, data
    return


if __name__ == "__main__":
    main()
