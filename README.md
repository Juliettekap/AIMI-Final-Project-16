# AIMI-Final-Project

This project is part of the Prostate Imaging Cancer Aritificial Intelligence (PICAI) hosted on [grand-challenge](https://pi-cai.grand-challenge.org/).

This repository builds upon the [provided repository](https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm) for the inference of models of the challenge. It's there to build containers to be evaluated. To get the original model you can clone this directory:

## Clone the provided repository for inference
```bash
git clone https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm.git
```
In this repository are some additional files that help with working with Snellius.
- [replace_path_in_overviews.py](https://github.com/Juliettekap/AIMI-Final-Project-16/blob/main/replace_path_in_overviews.py): Special thanks to Group 13 that also worked on the PICAI Challenge and shared this code with us.
- [trim_model.py](https://github.com/Juliettekap/AIMI-Final-Project-16/blob/main/trim_model.py): Is there to remove unnecessary information from models that are trained with the baseline code
- Shell scripts: Those include the scripts that we used to train our models using the template algorithm provided by the DIAG-Nijmegen group.
- [Juptyter notebook](https://github.com/Juliettekap/AIMI-Final-Project-16/blob/main/DimensionCalc.ipynb): Demonstrates the steps and calculations involved in selecting the region of interest (ROI) for cropping, ensuring that the prostate is accurately targeted within the input MRI images

To get the same containers as us please,
1. Move the desired weights from their folder into the weights folder.
2. Then make sure that your process.py file has the correct image shapes, while processing. (For crop256 weights the correct image shape is [20, 256, 256] while for crop128 weights the image shape should be [20, 128, 128]
3. Run the build.sh/build.bat script.

To get the [PICAI algorithm baseline](https://github.com/DIAGNijmegen/picai_baseline/):

## Clone the algorithm template repository
```bash
git clone https://github.com/DIAGNijmegen/picai_baseline.git
```

We also recommend to follow the entire [tutorial](https://github.com/DIAGNijmegen/picai_baseline/blob/main/README.md) for the baseline.

Once you have done so you should then be able to run our job scripts for the image crop of [[20, 128, 128]](https://github.com/Juliettekap/AIMI-Final-Project-16/blob/main/crop128_script.sh) or [[20, 256, 256]](https://github.com/Juliettekap/AIMI-Final-Project-16/blob/main/crop256_script.sh) in which you can also find and adjust the hyperparameters used for training.
