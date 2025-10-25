import argparse
import glob
import os
from functools import partial

import torch
from nibabel import processing
from tqdm import tqdm as std_tqdm

# tqdm wrapper with dynamic terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)

import nibabel as nib
import numpy as np

# Project-local utilities
from utils.cropping import cropping
from utils.hemisphere import hemisphere
from utils.load_model import load_model
from utils.make_csv import make_csv
from utils.make_level import create_parcellated_images
from utils.parcellation import parcellation
from utils.postprocessing import postprocessing
from utils.preprocessing import preprocessing
from utils.stripping import stripping


def create_parser():
    """
    Build and return the CLI argument parser.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference with OpenMAP-T1 on T1-weighted brain MRI.")
    parser.add_argument(
        "-i",
        required=True,
        help="Input folder containing one or more NIfTI files (.nii or .nii.gz).",
    )
    parser.add_argument(
        "-o",
        required=True,
        help=("Output folder where results will be written. " "The folder is created automatically if it does not exist."),
    )
    parser.add_argument(
        "-m",
        required=True,
        help="Folder containing pretrained model weights required by OpenMAP-T1.",
    )

    # Mutually exclusive short-circuit modes: run only a subset of the full pipeline.
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--only-face-cropping",
        action="store_true",
        help=("Run only the face-cropping step and exit early. " "All subsequent steps are skipped."),
    )
    group.add_argument(
        "--only-skull-stripping",
        action="store_true",
        help=("Run up to and including skull stripping and exit early. " "Parcellation and later steps are skipped."),
    )

    args = parser.parse_args()
    print("Parsed arguments:", args)
    return args


def main():
    """
    Execute the OpenMAP-T1 parcellation pipeline.

    Processing outline per input NIfTI:
      1) Read image and convert to canonical orientation; persist a float32 copy.
      2) Preprocess and standardize image for downstream networks.
      3) Face cropping (cnet) → optional early exit.
      4) Skull stripping (ssnet) → optional early exit.
      5) Whole-brain parcellation (pnet).
      6) Hemisphere separation (hnet).
      7) Post-processing to reconcile labels and geometry.
      8) Volume quantification and CSV export.
      9) Conform output back to original geometry and save Level 5 labels.
     10) Generate auxiliary parcellated images for visualization.
    """
    # Citation block printed at runtime for proper attribution.
    print(
        "\n#######################################################################\n"
        "Please cite the following paper when using OpenMAP-T1:\n"
        "Kei Nishimaki, Kengo Onda, Kumpei Ikuta, Jill Chotiyanonta, Yuto Uchida, Hitoshi Iyatomi, Kenichi Oishi (2024).\n"
        "OpenMAP-T1: A Rapid Deep Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain.\n"
        "paper: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70063.\n"
        "Submitted for publication in the Human Brain Mapping.\n"
        "#######################################################################\n"
    )

    # Parse command-line arguments.
    opt = create_parser()

    # Device selection order: CUDA → Apple MPS → CPU.
    # Note: MPS is available on Apple Silicon with recent PyTorch builds.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load pretrained models required by the pipeline components.
    try:
        cnet, ssnet, pnet, hnet = load_model(opt, device)
        print("Load complete !!")
    except Exception as e:
        # Continue to allow the script to report the error and exit gracefully later.
        print("Error during model loading:", e)

    # Basic sanity check on input directory.
    if not os.path.exists(opt.i):
        print(f"Error: Input directory {opt.i} does not exist.")
    else:
        print(f"Input directory {opt.i} exists.")

    # Enumerate NIfTI inputs recursively, supporting both .nii and .nii.gz.
    # Note: Double 'sorted' is redundant but harmless; ensures deterministic order.
    pathes = sorted(sorted(glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True)) + sorted(glob.glob(os.path.join(opt.i, "**/*.nii.gz"), recursive=True)))
    print(f"Found {len(pathes)} NIfTI files in {opt.i}")

    # Process each input image independently.
    for path in tqdm(pathes):
        try:
            # Derive a clean base name without any extension.
            basename = os.path.splitext(os.path.basename(path))[0]
            if basename.endswith(".nii"):
                # Handles the .nii.gz case where os.path.splitext removes only .gz.
                basename = os.path.splitext(basename)[0]

            # Create a per-case output subdirectory.
            output_dir = os.path.join(opt.o, basename)
            os.makedirs(output_dir, exist_ok=True)

            # Load image, reorient to RAS+ canonical, and drop degenerate dimensions.
            odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))

            # Persist a canonicalized float32 copy for provenance.
            nii = nib.Nifti1Image(odata.get_fdata().astype(np.float32), affine=odata.affine)
            os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)
            nib.save(nii, os.path.join(output_dir, f"original/{basename}.nii"))

            # Preprocessing: intensity normalization, spacing/orientation harmonization, etc.
            # Returns (original-like) 'odata' and a standardized 'data' used by the networks.
            odata, data = preprocessing(path, output_dir, basename)

            # Face cropping using the cropping network (returns cropped volume + spatial shift).
            cropped, shift = cropping(output_dir, basename, odata, data, cnet, device)

            # Early exit if the user requested cropping only.
            if opt.only_face_cropping:
                continue

            # Skull stripping (brain extraction).
            stripped = stripping(output_dir, basename, cropped, odata, data, ssnet, shift, device)

            # Early exit if the user requested up to skull stripping only.
            if opt.only_skull_stripping:
                continue

            # Parcellation into anatomical labels.
            parcellated = parcellation(stripped, pnet, device)

            # Hemisphere mask/labels to distinguish left/right brain.
            separated = hemisphere(stripped, hnet, device)

            # Post-processing to fuse parcellation with hemisphere info and to restore shifts.
            output = postprocessing(parcellated, separated, shift, device)

            # Quantify regional volumes and export to CSV.
            df = make_csv(output, output_dir, basename)

            # Conform output label image back to the original image geometry.
            # Use nearest-neighbor resampling (order=0) to preserve integer labels.
            nii = nib.Nifti1Image(output.astype(np.uint16), affine=data.affine)
            header = odata.header
            nii = processing.conform(
                nii,
                out_shape=(header["dim"][1], header["dim"][2], header["dim"][3]),
                voxel_size=(header["pixdim"][1], header["pixdim"][2], header["pixdim"][3]),
                order=0,
            )

            # Save standardized Level-5 parcellation.
            os.makedirs(os.path.join(output_dir, "parcellated"), exist_ok=True)
            nib.save(nii, os.path.join(output_dir, f"parcellated/{basename}_Type1_Level5.nii"))

            # Generate auxiliary visualizations / per-level parcellated volumes.
            create_parcellated_images(output, output_dir, basename, odata, data)

            # Explicit cleanup of large arrays to ease memory pressure in long batches.
            del odata, data

        except Exception as e:
            # Robust per-file error isolation: proceed to the next case on failure.
            print(f"Error processing {path}: {e}")
            continue
    return


if __name__ == "__main__":
    main()
