import numpy as np
import matplotlib.pyplot as plt
import cv2 
import pandas as pd
from get_stats import *

def bounding_box(mask, padding=5):
    """
    Get a bounding box around edges of a binary mask.

    Args:
        mask (numpy.ndarray): Binary mask image.
        padding (int): Amount of padding to add around the bounding box. Defaults to 5.

    Returns:
        tuple: Bounding box coordinates (x_min, x_max, y_min, y_max) with padding.
    """
    
    # Find the indices where the mask has 1s
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # Get the bounding box coordinates
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Padding around the bounding box
    y_min = max(y_min - padding, 0)
    y_max = min(y_max + padding, mask.shape[0] - 1)
    x_min = max(x_min - padding, 0)
    x_max = min(x_max + padding, mask.shape[1] - 1)
    
    return (x_min, x_max, y_min, y_max)

def process_imgs(mask, img, cancer, benign, verbose=True, per10=False, per90=False, fIC=True):
    """
    Process an image with associated masks for cancer and benign regions.

    Args:
        mask (numpy.ndarray): Binary mask for the region of interest.
        img (numpy.ndarray): The original image to process.
        cancer (numpy.ndarray): Binary mask for cancer regions (optional).
        benign (numpy.ndarray): Binary mask for benign regions (optional).
        verbose (bool): Whether to print diagnostic information. Defaults to True.
        per10 (bool): If True, compute the 10th percentile of pixel values. Defaults to False.
        per90 (bool): If True, compute the 90th percentile of pixel values. Defaults to False.
        fIC (bool): Whether the input image is VERDICT fIC. Defaults to True.

    Returns:
        tuple: Processed masked image, cancer image, benign image, average cancer value, average benign value.
    """
    
    cancer_img, benign_img  = None, None
    avg_cancer, avg_benign  = None, None
    
    dims = bounding_box(mask)
        
    masked_img = img*mask
    if fIC is not None:
        masked_img = cv2.flip(masked_img,1)[dims[0]:dims[1],dims[2]:dims[3]]
    
    if per90:
        reverse=True
    else:
        reverse=False    
            
    if cancer is not None:  
        cancer_img = img*cancer
        if fIC is not None:
            cancer_img = cv2.flip(cancer_img,1)[dims[0]:dims[1],dims[2]:dims[3]]
        if per10 or per90:
            avg_cancer = get_10perc(cancer_img, reverse=reverse)
        else:
            avg_cancer = get_avg(cancer_img)
        if verbose:
            print('Cancer: ', avg_cancer)
        
    if benign is not None:
        benign_img = img*benign
        if fIC is not None:
            benign_img = cv2.flip(benign_img,1)[dims[0]:dims[1],dims[2]:dims[3]]
        if per10 or per90:
            avg_benign = get_10perc(benign_img, reverse=reverse)
        else:
            avg_benign = get_avg(benign_img)
        if verbose:
            print('Benign: ', avg_benign)
    
    return masked_img, cancer_img, benign_img, avg_cancer, avg_benign
    
    
def plot_imgs(masked_img, cancer_img, benign_img, sid, density=None):
    """
    Visualize masked image with cancer and benign overlays.

    Args:
        masked_img (numpy.ndarray): Processed masked image.
        cancer_img (numpy.ndarray): Image with cancer mask applied (optional).
        benign_img (numpy.ndarray): Image with benign mask applied (optional).
        sid (str): Identifier for the sample.
        density (str): Type of density or map being visualized (optional).

    Returns:
        None. Displays a matplotlib figure with the images.
    """
    
    if density is not None:
        if 'density' in density:
            text    = 'Cell density'
            max_val = 10000
        else:
            text    = density + ' fraction'
            max_val = 1
        origin  = 'upper'
    else:
        text    = 'fIC map'
        max_val = 1
        origin  = 'lower'
      
    plt.figure()
    _, axs = plt.subplots(1,3, figsize=(12,4))
    plt.suptitle(sid, fontsize=14)
    axs[0].imshow(masked_img, cmap='jet', vmin=0, vmax=max_val, origin=origin)
    axs[0].axis('off')
    axs[0].set_title(text + ' - prostate mask')
        
    if cancer_img is not None: 
        im = axs[1].imshow(cancer_img, cmap='jet', vmin=0, vmax=max_val, origin=origin)
        axs[1].axis('off')
        axs[1].set_title(text + ' - cancer')
        
    if benign_img is not None:
        im = axs[2].imshow(benign_img, cmap='jet', vmin=0, vmax=max_val, origin=origin)
        axs[2].axis('off')
        axs[2].set_title(text + ' - benign')
        
    plt.colorbar(im, fraction=0.038, pad=0.03, ax=axs[2])
    plt.show()


def results_to_numpy_arrays(results):
    """
    Convert a dictionary of results into structured NumPy arrays for analysis.

    Args:
        results (dict): Dictionary containing analysis results with keys for various tissue properties.

    Returns:
        tuple: NumPy arrays for fIC, density, epithelial, stroma, and lumen values.
    """
    
    fIC_cancer, fIC_benign          = [], []
    density_cancer, density_benign  = [], []
    epith_cancer, epith_benign      = [], []
    stroma_cancer, stroma_benign    = [], []
    lumen_cancer, lumen_benign      = [], []

    # Helper function to check if all required values are not None
    def check_keys(keys, s):
        return all(s[key] is not None for key in keys)

    for s in results.values():
        if check_keys(['fIC-cancer', 'density-cancer', 'epithelial-cancer', 'stroma-cancer', 'lumen-cancer'], s):
            fIC_cancer.append(s['fIC-cancer'])
            density_cancer.append(s['density-cancer'])
            epith_cancer.append(s['epithelial-cancer'])
            stroma_cancer.append(s['stroma-cancer'])
            lumen_cancer.append(s['lumen-cancer'])
        
        if check_keys(['fIC-benign', 'density-benign', 'epithelial-benign', 'stroma-benign', 'lumen-benign'], s):
            fIC_benign.append(s['fIC-benign'])
            density_benign.append(s['density-benign'])
            epith_benign.append(s['epithelial-benign'])
            stroma_benign.append(s['stroma-benign'])
            lumen_benign.append(s['lumen-benign'])

    # Create numpy arrays for cancer and benign values
    fIC_vals        = np.array([fIC_cancer, fIC_benign])
    density_vals    = np.array([density_cancer, density_benign])
    epith_vals      = np.array([epith_cancer, epith_benign])
    stroma_vals     = np.array([stroma_cancer, stroma_benign])
    lumen_vals      = np.array([lumen_cancer, lumen_benign])

    return fIC_vals, density_vals, epith_vals, stroma_vals, lumen_vals


def get_paths(folder, name, hist_name='density', cancer='L1'):
    """
    Construct file paths for input images and masks based on naming conventions.

    Args:
        folder (str): Base folder containing the data.
        name (str): Sample name or identifier.
        hist_name (str): Type of histogram or map (e.g., 'density'). Defaults to 'density'.
        cancer (str): Cancer label (e.g., 'L1'). Defaults to 'L1'.

    Returns:
        tuple: File paths for the main image, mask, cancer mask, and benign mask.
    """
    
    sid = name[:10]
    
    if cancer == 'L1':
        cancer_label = ''
    else:
        cancer_label = f'_{cancer[1]}'
    
    img_path    = f'{folder}/{sid}/{name}_large_{hist_name}.png'
    mask_path   = f'{folder}/{sid}/{name}_mask.png'
    cancer_path = f'{folder}/{sid}/{name}_cancer{cancer_label}.png'
    benign_path = f'{folder}/{sid}/{name}_benign.png'    
    
    return img_path, mask_path, cancer_path, benign_path


def get_all_roi_stats(MR_map, seq):
    """
    Retrieve region-of-interest (ROI) statistics from an external file.

    Args:
        MR_map (str): Type of MR map ('fIC' or 'ADC').
        seq (str): Sequence name or identifier ('mpMRI' or 'VERDICT').

    Returns:
        pandas.DataFrame: DataFrame containing the ROI statistics.
    """

    if MR_map=='fIC':
        file_path = f'/Users/martamasramon/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Project/Images/Stats/VERDICT/Stats of all PCa & benign ROIs-fIC.xlsx'
        df = pd.read_excel(file_path)
        df = df.dropna(how='all')
    if MR_map=='ADC':
        file_path = f'/Users/martamasramon/Library/CloudStorage/OneDrive-ImperialCollegeLondon/Project/Images/Stats/ADC - no b0/all_stats_{seq}_ADC.csv'
        df = pd.read_csv(file_path)
        
    return df


def get_fIC_roi(df, patient_id, stat='Mean', cancer_label='L1', verbose=True):
    """
    Extract fIC statistics for cancer and benign ROIs.

    Args:
        df (pandas.DataFrame): DataFrame with ROI statistics.
        patient_id (str): Identifier for the patient/sample.
        stat (str): Statistical measure to retrieve (e.g., 'Mean'). Defaults to 'Mean'.
        cancer_label (str): Column label for cancer ROIs. Defaults to 'L1'.
        verbose (bool): Whether to print diagnostic information. Defaults to True.

    Returns:
        tuple: Cancer and benign fIC values.
    """
    
    if verbose:
        print('\n* fIC')
    
    chosen_df = df[(df['PATIENT ID'] == patient_id) & (df['Stat'] == stat)]
    cancer = chosen_df[cancer_label].values[0]
    benign = chosen_df['Benign'].values[0]
    
    try:
        cancer = float(cancer.replace(',', '.'))
        benign = float(benign.replace(',', '.'))
    except:
        pass
    
    if verbose:
        print('Cancer: ', cancer)
        print('Benign: ', benign)
    
    return cancer, benign


def get_ADC_roi(df, patient_id, cancer_label='L1', verbose=True):
    """
    Extract ADC statistics for cancer and benign ROIs.

    Args:
        df (pandas.DataFrame): DataFrame with ROI statistics.
        patient_id (str): Identifier for the patient/sample.
        cancer_label (str): Column label for cancer ROIs. Defaults to 'L1'.
        verbose (bool): Whether to print diagnostic information. Defaults to True.

    Returns:
        tuple: Cancer and benign ADC values.
    """
    
    if verbose:
        print('\n* ADC')
    
    chosen_df = df[(df['sid'] == patient_id)]
    cancer = chosen_df[cancer_label].values[0]
    benign = chosen_df['Benign'].values[0]
    
    try:
        cancer = float(cancer.replace(',', '.'))
        benign = float(benign.replace(',', '.'))
    except:
        pass
    
    if verbose:
        print('Cancer: ', cancer)
        print('Benign: ', benign)
    
    return cancer, benign


def get_sid_roi(df, patient_id, MR_map, cancer_label='L1', verbose=True):
    """Get cancer and benign statistics for a given patient/sample.

    Args:
        df (pandas.DataFrame): DataFrame with ROI statistics.
        patient_id (str): Identifier for the patient/sample.
        MR_map (str): Type of MR map ('fIC' or 'ADC').
        cancer_label (str): Column label for cancer ROIs. Defaults to 'L1'.
        verbose (bool): Whether to print diagnostic information. Defaults to True.

    Returns:
        tuple: Cancer and benign statistics.
    """
    
    if MR_map == 'fIC':
        cancer, benign = get_fIC_roi(df, patient_id, cancer_label=cancer_label, verbose=verbose)
    else:
        cancer, benign = get_ADC_roi(df, patient_id, cancer_label=cancer_label, verbose=verbose)
    return cancer, benign 
    
    
def analyse_density(folder, sid, s, plot=True, hist_name='density', cancer='L1', verbose=True, per10=False, per90=False):
    """
    Analyze density and tissue properties for a given sample.

    Args:
        folder (str): Base folder containing the data.
        sid (str): Sample ID.
        s (str): Sub-sample identifier.
        plot (bool): Whether to visualize the results. Defaults to True.
        hist_name (str): Type of histogram or map. Defaults to 'density'.
        cancer (str): Cancer label (e.g., 'L1'). Defaults to 'L1'.
        verbose (bool): Whether to print diagnostic information. Defaults to True.
        per10 (bool): If True, compute the 10th percentile of pixel values. Defaults to False.
        per90 (bool): If True, compute the 90th percentile of pixel values. Defaults to False.

    Returns:
        tuple: Average cancer and benign density values.
    """
    file_name = f'{sid}_{s}'
    if verbose:
        print(f'\n* {hist_name}')
    
    img_path, mask_path, cancer_path, benign_path = get_paths(folder, file_name, hist_name=hist_name, cancer=cancer)
    
    img     = cv2.imread(img_path)[:,:,0]/255
    if hist_name=='density':
        img*=10000
    mask    = cv2.imread(mask_path)[:,:,0]/255
    try:
        cancer = cv2.imread(cancer_path)[:,:,0]/255
    except:
        cancer = None
    try:
        benign  = cv2.imread(benign_path)[:,:,0]/255
    except:
        benign = None
      
    masked_img, cancer_img, benign_img, avg_cancer, avg_benign = process_imgs(mask, img, cancer, benign, verbose=verbose, per10=per10, per90=per90,fIC=None)
    
    if plot:
        plot_imgs(masked_img, cancer_img, benign_img, file_name, density=hist_name)
    
    return avg_cancer, avg_benign


def get_fraction_density(density, epithelial, stroma, fraction='epithelial'):
    """
    Compute density fraction for epithelial or stromal tissues.

    Args:
        density (float): Overall density value.
        epithelial (float): Epithelial density value.
        stroma (float): Stromal density value.
        fraction (str): Tissue type ('epithelial' or 'stromal'). Defaults to 'epithelial'.

    Returns:
        float: Fractional density value.
    """
    
    if fraction=='epithelial':
        fraction_density = density * (epithelial/(epithelial+stroma))
    else:
        fraction_density = density * (stroma/(epithelial+stroma))
    return fraction_density
        
        
def boxplot_single(axis, data, title):
    """
    Create a boxplot for a single tissue property.

    Args:
        axis (matplotlib.axis): Axis object for the plot.
        data (numpy.ndarray): Data array with cancer and benign values.
        title (str): Title for the plot.

    Returns:
        None. Displays a boxplot on the specified axis.
    """
    
    axis.boxplot([data[0,:], data[1,:]], labels=['csPCa', 'Benign'],widths=0.3)
    axis.set_title(title)