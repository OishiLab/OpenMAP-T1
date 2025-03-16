![Explain](media/Explain.png)

# OpenMAP-T1
![Figure3](media/Representative.png)

[![](http://img.shields.io/badge/medRxiv-10.1101/2024.01.18.24301494-B31B1B.svg)](https://www.medrxiv.org/content/10.1101/2024.01.18.24301494v1)
[![IEEE Xplore](https://img.shields.io/badge/Accepted-Human%20Brain%20Mapping-%2300629B%09)](https://onlinelibrary.wiley.com/journal/10970193)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fmfkxxZjChExnl5cHITYkNYgTu3MZ7Ql#scrollTo=xwZxyL5ewVNF)
![Python 3.8](https://img.shields.io/badge/OpenMAP-T1-brightgreen.svg)

**OpenMAP-T1: A Rapid Deep-Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain**<br>
**Author**: Kei Nishimaki, [Kengo Onda](https://researchmap.jp/kengoonda?lang=en), Kumpei Ikuta, Jill Chotiyanonta, [Yuto Uchida](https://researchmap.jp/uchidayuto), [Susumu Mori](https://www.hopkinsmedicine.org/profiles/details/susumu-mori), [Hitoshi Iyatomi](https://iyatomi-lab.info/english-top), [Kenichi Oishi](https://www.hopkinsmedicine.org/profiles/details/kenichi-oishi)<br>

The Russell H. Morgan Department of Radiology and Radiological Science, The Johns Hopkins University School of Medicine, Baltimore, MD, USA <br>
Department of Applied Informatics, Graduate School of Science and Engineering, Hosei University, Tokyo, Japan <br>
The Richman Family Precision Medicine Center of Excellence in Alzheimer's Disease, Johns Hopkins University School of Medicine, Baltimore, MD, USA<br>

**Abstract**: *This study introduces OpenMAP-T1, a deep learning-based method for rapid and accurate whole brain parcellation in T1-weighted brain MRI, aiming to overcome the limitations of conventional normalization-to-atlas-based approaches and multi-atlas label-fusion (MALF) techniques. Brain image parcellation is a fundamental process in neuroscientific and clinical research, enabling detailed analysis of specific cerebral regions. Normalization-to-atlas-based methods have been employed for this task, but they face limitations due to variations in brain morphology, especially in pathological conditions. The MALF teqhniques improved the accuracy of the image parcellation and robustness to variations in brain morphology but at the cost of high computational demand that requires lengthy processing time. OpenMAP-T1 integrates several convolutional neural network models across six phases: preprocessing, cropping, skull stripping, parcellation, hemisphere segmentation, and final merging. This process involves standardizing MRI images, isolating the brain tissue, and parcellating it into 280 anatomical structures that cover the whole brain, including detailed gray and white matter structures, while simplifying the parcellation processes and incorporating robust training to handle various scan types and conditions. The OpenMAP-T1 was tested on eight available open resources, including real-world clinical images, demonstrating robustness across different datasets with variations in scanner types, magnetic field strengths, and image processing techniques like defacing. Compared to existing methods, OpenMAP-T1 significantly reduced the processing time per image from several hours to less than 90 seconds without compromising accuracy. It was particularly effective in handling images with intensity inhomogeneity and varying head positions, conditions commonly seen in clinical settings. The adaptability of OpenMAP-T1 to a wide range of MRI datasets and robustness to various scan conditions highlight its potential as a versatile tool in neuroimaging.*

Paper: https://onlinelibrary.wiley.com/doi/full/10.1002/hbm.70063<br>
Submitted for publication in the **Human Brain Mapping**<br>

## Installation Instruction
**OpenMAP-T1-V2 parcellates the whole brain into 280 anatomical regions based on JHU-atlas in 50 (sec/case).**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fmfkxxZjChExnl5cHITYkNYgTu3MZ7Ql#scrollTo=xwZxyL5ewVNF)

0. install python and make virtual environment<br>
Python 3.9 or later is recommended.

1. Clone this repository, and go into the repository:
```
git clone https://github.com/OishiLab/OpenMAP-T1.git
cd OpenMAP-T1
```
3. Please install PyTorch compatible with your environment.<br>
https://pytorch.org/

Once you select your environment, the required commands will be displayed.

![image](media/PyTorch.png)

If you want to install an older Pytorch environment, you can download it from the link below.<br>
https://pytorch.org/get-started/previous-versions/

4.  Install libraries other than PyTorch:
```
pip install -r requirements.txt
```
5. Please apply and download the pre-trained model from the link below and upload it to your server.

6. You can run OpenMAP-T1 !!

## How to download the pretrained model.
You can get the pretrained model from this link.
[Link of pretrained model](https://forms.office.com/Pages/ResponsePage.aspx?id=OPSkn-axO0eAP4b4rt8N7Iz6VabmlEBIhG4j3FiMk75UQUxBMkVPTzlIQTQ1UEZJSFY1NURDNzRERC4u)

![image](media/Download_pretrained.png)

## How to use it
Using OpenMAP-T1 is straightforward. You can use it in any terminal on your linux system. We provide CPU as well as GPU support. Running on GPU is a lot faster though and should always be preferred. Here is a minimalistic example of how you can use OpenMAP-T1.

### Basic Usage
Run the script from your terminal using:
```
python3 parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```
* **-i INPUT_FOLDER**: Specifies the folder containing the input brain MRI images.
* **-o OUTPUT_FOLDER**: Defines the folder where the results will be saved. This folder will be created automatically if it does not exist.
* **-m MODEL_FOLDER**: Indicates the folder containing the pretrained models for processing.

### Using Spesific GPU
If you want to run the script on a specific GPU (for example, GPU 1), prepend the command with the ```CUDA_VISIBLE_DEVICES=N```.
```
CUDA_VISIBLE_DEVICES=1 python3 parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```
If the error occurs for Windows users, please change ```Python3``` to ```Python```.

### Optional Processing Steps
OpenMAP-T1 now allows you to perform only specific processing steps using the following mutually exclusive flags:
* **Only Face Cropping**: If you only want to perform face cropping and skip the rest of the processing steps, use:
```
python3 parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-face-cropping
```
* Only Skull Stripping: If you want to perform only skull stripping and skip all other processing steps, use:
```
python3 parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-skull-stripping
```

## Folder
All images you input must be in NifTi format and have a .nii extension.
```
INPUR_FOLDER/
   ├ A.nii
   ├ B.nii
   ├ *.nii

OUTPUT_FOLDER/
   ├── A
   │   ├── cropped
   │   │   ├── A_cropped_mask.nii
   │   │   └── A_cropped.nii
   │   ├── csv
   │   │   ├── A_Type1_Level1.csv
   │   │   ├── A_Type1_Level2.csv
   │   │   ├── A_Type1_Level3.csv
   │   │   ├── A_Type1_Level4.csv
   │   │   ├── A_Type1_Level5.csv
   │   │   ├── A_Type2_Level1.csv
   │   │   ├── A_Type2_Level2.csv
   │   │   ├── A_Type2_Level3.csv
   │   │   ├── A_Type2_Level4.csv
   │   │   └── A_Type2_Level5.csv
   │   ├── original
   │   │   ├── A_N4.nii
   │   │   └── A.nii
   │   ├── parcellated
   │   │   ├── A_Type1_Level1.nii
   │   │   ├── A_Type1_Level2.nii
   │   │   ├── A_Type1_Level3.nii
   │   │   ├── A_Type1_Level4.nii
   │   │   ├── A_Type1_Level5.nii
   │   │   ├── A_Type2_Level1.nii
   │   │   ├── A_Type2_Level2.nii
   │   │   ├── A_Type2_Level3.nii
   │   │   ├── A_Type2_Level4.nii
   │   │   └── A_Type2_Level5.nii
   │   └── stripped
   │       ├── A_stripped_mask.nii
   │       └── A_stripped.nii
   ├── ...

MODEL_FOLDER/
   ├ SSNet/SSNet.pth
   ├ PNet
   |   ├ coronal.pth
   |   ├ sagittal.pth
   |   └ axial.pth
   └ HNet/
      ├ coronal.pth
      └ axial.pth
```

## Supplementary information
![supplementary_level](media/Multilevel.png)
The OpenMAP-T1 parcellates the entire brain into five hierarchical structural levels, with the coarsest level comprising eight structures and the finest level comprising 280 structures.

* For additional visualization and detailed analysis, you can also utilize [3D Slicer](https://www.slicer.org/). 3D Slicer is a free, open-source platform for medical image computing that provides robust tools for segmentation, registration, and 3D visualization, making it an excellent choice for exploring the parcellation maps generated by OpenMAP-T1.

*  For additional visualization and detailed analysis, [ROIEditor](https://www.mristudio.org/installation.html) is also an excellent tool. ROIEditor is a free, open-source application specifically designed for creating and editing regions of interest (ROIs) in medical imaging. Its user-friendly interface facilitates precise segmentation and fine-tuning, making it ideal for isolating and analyzing specific regions on parcellation maps generated by OpenMAP-T1.


## FAQ
* **How much GPU memory do I need to run OpenMAP-T1?** <br>
We ran all our experiments on NVIDIA RTX3090 GPUs with 24 GB memory. For inference you will need less, but since inference in implemented by exploiting the fully convolutional nature of CNNs the amount of memory required depends on your image. Typical image should run with less than 4 GB of GPU memory consumption. If you run into out of memory problems please check the following: 1) Make sure the voxel spacing of your data is correct and 2) Ensure your MRI image only contains the head region.

* **Will you provide the training code as well?** <br>
No. The training code is tightly wound around the data which we cannot make public.

## Citation
```
@techreport{nishimaki2024openmap,
  title={OpenMAP-T1: A Rapid Deep-Learning Approach to Parcellate 280 Anatomical Regions to Cover the Whole Brain},
  author={Nishimaki, Kei and Onda, Kengo and Ikuta, Kumpei and Chotiyanonta, Jill and Uchida, Yuto and Mori, Susumu and Iyatomi, Hitoshi and Oishi, Kenichi and Alzheimer's Disease Neuroimaging Initiative and Australian Imaging Biomarkers and Lifestyle Flagship Study of Ageing},
  year={2024},
  institution={Wiley Online Library}
}
```

## Related Research
The following studies have utilized OpenMAP-T1 for advanced segmentation and analysis in T1-weighted MRI. 
1. **[Acceleration of Brain Atrophy and Progression From Normal Cognition to Mild Cognitive Impairment](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2825474)**  
   *Authors:* Yuto Uchida, MD, PhD; Kei Nishimaki; Anja Soldan, PhD; Abhay Moghekar, MBBS; Marilyn Albert, PhD; Kenichi Oishi, MD, PhD;  
   *Journal:* JAMA Network Open
2. **[A Neural Network Approach to Identify Left–Right Orientation of Anatomical Brain MRI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11808181/)**  
   *Authors:* Kei Nishimaki; Hitoshi Iyatomi, PhD; Kenichi Oishi, MD, PhD;  
   *Journal:* Brain and Behavior
