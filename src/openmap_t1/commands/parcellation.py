import gc
import os
import pathlib
from dataclasses import dataclass, field

import nibabel as nib
import numpy as np
import torch
from loguru import logger
from nibabel import processing
from tqdm.auto import tqdm

from openmap_t1.utils import (
    UNetModels,
    cropping,
    get_device,
    hemisphere,
    load_models,
    make_csv,
    parcellation,
    postprocessing,
    preprocessing,
    stripping,
)


@dataclass
class ParcellationArgs(object):
    input_folder: pathlib.Path = field(
        metadata={
            "help": "Input folder. Specifies the folder containing the input brain MRI images."
        }
    )
    output_folder: pathlib.Path = field(
        metadata={
            "help": "Output folder. Defines the output folder where the results will be saved. If the specified folder does not exist, it will be automatically created."
        }
    )


def run_parcellation(
    nii_path: pathlib.Path,
    models: UNetModels,
    device: torch.device,
    output_folder: pathlib.Path,
):
    # Extract the base name of the file (without extension)
    basename = nii_path.stem
    if basename.endswith(".nii"):
        basename, _ = os.path.splitext(basename)

    # Create the output directory for the current file
    output_dir = output_folder / basename
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the input image, convert it to canonical form, and squeeze it
    odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(nii_path)))

    # Create a new NIfTI image with the data converted to float32
    nii = nib.Nifti1Image(odata.get_fdata().astype(np.float32), affine=odata.affine)

    # Save the new NIfTI image to the output directory
    nib.save(nii, output_dir / f"{basename}.nii")

    # Preprocess the input image
    odata, data = preprocessing(nii_path, output_dir, basename)

    # Crop the image using the cropping network
    cropped = cropping(data, models.cnet, device)

    # Strip the image using the stripping network
    stripped, shift = stripping(cropped, data, models.ssnet, device)

    # Parcellate the stripped image using the parcellation networks
    parcellated = parcellation(
        stripped, models.pnet_c, models.pnet_s, models.pnet_a, device
    )

    # Separate the hemispheres using the hemisphere networks
    separated = hemisphere(stripped, models.hnet_c, models.hnet_a, device)

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
    gc.collect()


def run_parcellations(args: ParcellationArgs):
    logger.info(
        "\n\n"
        "#######################################################################\n"
        "Please cite the following paper when using OpenMAP-T1:\n"
        "Kei Nishimaki, Kengo Onda, Kumpei Ikuta, Jill Chotiyanonta, Yuto Uchida, Hitoshi Iyatomi, Kenichi Oishi (2024).\n"
        "OpenMAP-T1: A Rapid Deep Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain.\n"
        "paper: https://www.medrxiv.org/content/10.1101/2024.01.18.24301494v1.\n"
        "#######################################################################\n"
    )
    # Determine the device to use (CUDA, MPS, or CPU)
    device = get_device()
    logger.debug(f"Using device: {device}")

    # Load the pretrained models
    models = load_models(device)

    # Get the list of input files
    nii_pathes = sorted(list(args.input_folder.glob("**/*.nii"))) + sorted(
        list(args.input_folder.glob("**/*.nii.gz"))
    )

    # Run the parcellation on each input file
    for nii_path in tqdm(nii_pathes):
        run_parcellation(
            nii_path=nii_path,
            models=models,
            device=device,
            output_folder=args.output_folder,
        )
