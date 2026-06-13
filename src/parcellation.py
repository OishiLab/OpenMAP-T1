import argparse
import glob
import os
import sys
from functools import partial

import torch
from tqdm import tqdm as std_tqdm

# tqdm wrapper with dynamic terminal width
tqdm = partial(std_tqdm, dynamic_ncols=True)

import nibabel as nib
import numpy as np

# Project-local utilities
from utils.pipeline.cropping import cropping
from utils.pipeline.hemisphere import hemisphere
from utils.models.load_model import load_model
from utils.output.make_csv import make_csv
from utils.output.make_level import create_parcellated_images
from utils.output.output_space import OUTPUT_SPACE_CHOICES, save_label_volume
from utils.pipeline.parcellate import parcellation
from utils.pipeline.postprocessing import postprocessing
from utils.pipeline.preprocessing import preprocessing
from utils.output.processing_log import ProcessingLog
from utils.pipeline.stripping import stripping


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
    parser.add_argument(
        "--output-ext",
        default=".nii.gz",
        choices=[".nii.gz", ".nii"],
        help="Output NIfTI extension for saved images (default: .nii.gz).",
    )
    parser.add_argument(
        "--output-space",
        default="native",
        choices=list(OUTPUT_SPACE_CHOICES),
        help=(
            "Geometry for saved NIfTI outputs. "
            "'native': resample to the N4-corrected canonical input grid (default). "
            "'conform': keep the 1 mm isotropic 256³ processing grid. "
            "'both': write both versions (1 mm outputs under conform/)."
        ),
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
        "Kei Nishimaki, Kengo Onda, Kumpei Ikuta, Jill Chotiyanonta, Yuto Uchida, Susumu Mori, "
        "Hitoshi Iyatomi, Kenichi Oishi, Alzheimer's Disease Neuroimaging Initiative, and "
        "Australian Imaging Biomarkers and Lifestyle Flagship Study of Ageing (2024).\n"
        "OpenMAP-T1: A Rapid Deep-Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain.\n"
        "paper: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70063.\n"
        "Published in Human Brain Mapping.\n"
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
        print(f"Error during model loading: {e}")
        sys.exit(1)

    # Basic sanity check on input directory.
    if not os.path.isdir(opt.i):
        print(f"Error: Input directory {opt.i} does not exist.")
        sys.exit(1)
    print(f"Input directory {opt.i} exists.")

    # Enumerate NIfTI inputs recursively, supporting both .nii and .nii.gz.
    # Note: Double 'sorted' is redundant but harmless; ensures deterministic order.
    pathes = sorted(sorted(glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True)) + sorted(glob.glob(os.path.join(opt.i, "**/*.nii.gz"), recursive=True)))
    print(f"Found {len(pathes)} NIfTI files in {opt.i}")

    processing_log = ProcessingLog()
    os.makedirs(opt.o, exist_ok=True)

    # Process each input image independently.
    for path in tqdm(pathes):
        basename = None
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
            nib.save(nii, os.path.join(output_dir, f"original/{basename}{opt.output_ext}"))

            # Preprocessing: intensity normalization, spacing/orientation harmonization, etc.
            # Returns (original-like) 'odata' and a standardized 'data' used by the networks.
            odata, data = preprocessing(path, output_dir, basename, opt.output_ext)

            # Face cropping using the cropping network (returns cropped volume + spatial shift).
            cropped, shift = cropping(
                output_dir, basename, odata, data, cnet, device, opt.output_ext, opt.output_space
            )

            # Early exit if the user requested cropping only.
            if opt.only_face_cropping:
                processing_log.add_skipped(
                    path,
                    basename,
                    "Pipeline stopped early (--only-face-cropping).",
                )
                continue

            # Skull stripping (brain extraction).
            stripped = stripping(
                output_dir,
                basename,
                cropped,
                odata,
                data,
                ssnet,
                shift,
                device,
                opt.output_ext,
                opt.output_space,
            )

            # Early exit if the user requested up to skull stripping only.
            if opt.only_skull_stripping:
                processing_log.add_skipped(
                    path,
                    basename,
                    "Pipeline stopped early (--only-skull-stripping).",
                )
                continue

            # Parcellation into anatomical labels.
            parcellated = parcellation(stripped, pnet, device)

            # Hemisphere mask/labels to distinguish left/right brain.
            separated = hemisphere(stripped, hnet, device)

            # Post-processing to fuse parcellation with hemisphere info and to restore shifts.
            output = postprocessing(parcellated, separated, shift, device)

            processing_log.check_low_volume(path, basename, output)

            # Quantify regional volumes and export to CSV.
            df = make_csv(output, output_dir, basename)

            save_label_volume(
                output_dir,
                "parcellated",
                basename,
                "Type1_Level5",
                output,
                odata,
                data,
                opt.output_ext,
                opt.output_space,
            )

            create_parcellated_images(
                output, output_dir, basename, odata, data, opt.output_ext, opt.output_space
            )

            # Explicit cleanup of large arrays to ease memory pressure in long batches.
            del odata, data

        except Exception as e:
            # Robust per-file error isolation: proceed to the next case on failure.
            print(f"Error processing {path}: {e}")
            processing_log.add_failed(path, basename or os.path.basename(path), str(e))
            continue

    processing_log.save(opt.o)
    return


if __name__ == "__main__":
    main()
