import os

import nibabel as nib
import SimpleITK as sitk
from nibabel import processing
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform


def N4_Bias_Field_Correction(input_path, output_path):
    """
    Perform N4 Bias Field Correction on an input image and save the corrected image to the specified output path.

    Args:
        input_path (str): Path to the input image file.
        output_path (str): Path to save the corrected image file.

    Returns:
        None
    """
    raw_img_sitk = sitk.ReadImage(input_path, sitk.sitkFloat32)
    transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)
    transformed = sitk.LiThreshold(transformed, 0, 1)
    head_mask = transformed
    shrinkFactor = 4
    inputImage = sitk.Shrink(raw_img_sitk, [shrinkFactor] * raw_img_sitk.GetDimension())
    maskImage = sitk.Shrink(head_mask, [shrinkFactor] * raw_img_sitk.GetDimension())
    bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected = bias_corrector.Execute(inputImage, maskImage)
    log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
    corrected_image_full_resolution = raw_img_sitk / sitk.Exp(log_bias_field)
    sitk.WriteImage(corrected_image_full_resolution, output_path)
    return


def preprocessing(ipath, output_dir, basename):
    """
    Preprocesses a medical image by performing N4 bias field correction and conforming the image to a specified shape and voxel size.

    Args:
        ipath (str): The input file path of the medical image to be processed.
        output_dir (str): The directory where the processed image will be saved.
        basename (str): The base name for the output file.

    Returns:
        tuple: A tuple containing:
            - odata (nibabel.Nifti1Image): The N4 bias field corrected image.
            - data (nibabel.Nifti1Image): The conformed image with specified shape and voxel size.
    """
    opath = os.path.join(output_dir, f"{basename}_N4.nii")
    N4_Bias_Field_Correction(ipath, opath)
    odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(opath)))
    data = processing.conform(odata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1)
    return odata, data
