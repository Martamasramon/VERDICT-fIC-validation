# Histological and statistical analysis for VERDICT validation

This project is organised as follows. There is a Jupyter Notebook (```pipeline.ipynb```) that exemplifies how to use all the Python functions. These make up steps 1, 4 and 5. 

## 1. Divide histology image into patches

Input: 
- NDPI image

Output: 
- Image patches (saved as .png)

## 2. Count cell nuclei per patch in QuPath

Input: 
- Image patches

Output:
- CSV file 

## 3. Get tissue fractions in ImagePro

Input:
- Image patches

Output:
- Image patch masks

## 4. Obtain histologically-derived quantitative maps

Input:
- CSV file with cell density per histo patch
- TXT file with tissue fractions (epithelial, lumen, stroma) per patch

Output:
- Histology-derived maps: cell density, epithlelial fraction, lumen fraction, stroma fraction

## 5. Compare ROIs in histology and VERDICT maps

Input:
- MR Images (mpMRI and VERDICT)
- Histology-derived maps

Output:
- Plots of benign vs csPCa ROIs for fIC/ADC and histology-derived maps
- Stats (AUC, mean and std) of benign and csPCa ROIs for fIC/ADC and histology-derived maps
- Correlation plots with 95% CI of fIC/ADC against histology-derived maps

