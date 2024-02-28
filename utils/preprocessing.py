import os

import nibabel as nib
import SimpleITK as sitk
from nibabel import processing
from nibabel.orientations import aff2axcodes, axcodes2ornt, ornt_transform


def N4_Bias_Field_Correction(input_path, output_path):
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


def preprocessing(ipath, save):
    opath = f"N4/{save}.nii"
    os.makedirs("N4", exist_ok=True)
    N4_Bias_Field_Correction(ipath, opath)
    odata = nib.squeeze_image(nib.as_closest_canonical(nib.load(opath)))
    data = processing.conform(
        odata, out_shape=(256, 256, 256), voxel_size=(1.0, 1.0, 1.0), order=1
    )
    return odata, data
