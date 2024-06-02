# AIMI-Final-Project

This project is part of the Prostate Imaging Cancer Aritificial Intelligence (PICAI) hosted on [grand-challenge](https://pi-cai.grand-challenge.org/).

This repository builds upon the [provided repository](https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm) for the inference of models of the challenge. It's there to build containers to be evaluated. To get the original model you can clone this directory:

## Clone the provided repository for inference
```bash
git clone https://github.com/DIAGNijmegen/picai_unet_semi_supervised_gc_algorithm.git
```
In this repository are some additional files that help with working with Snellius.
- replace_path_in_overviews.py: Special thanks to Group 3 that also worked on the PICAI Challenged and shared this code with us.
- trim_model.py: Is there to remove unnecessary information from models that are trained with the baseline code
- Shell scripts: Those include the scripts that we used to train our models using the template algorithm provided by the DIAG-Nijmegen group.
- Juptyter notebook: Demonstrates the steps and calculations involved in selecting the region of interest (ROI) for cropping, ensuring that the prostate is accurately targeted within the input MRI images

To get the [PICAI algorithm baseline](https://github.com/DIAGNijmegen/picai_baseline/) you can clone it:

## Clone the algorithm template repository
```bash
git clone https://github.com/DIAGNijmegen/picai_baseline.git
```