import cv2
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import tifffile

def large_size(path_in, name, key=1):
    """
    Get the dimensions of an NDPI file.

    Args:
        path_in (str): Folder containing the NDPI file.
        name (str): Base name of the NDPI file.
        key (int): TIFF image key for reading. Defaults to 1.

    Returns:
        tuple: Width and height of the image.
    """
    histo = tifffile.imread(path_in + name + '.ndpi', key=key)
    w,h,_ = histo.shape
    
    return w, h
    
            
def get_mask(path_ndpi, path_masks, name, pad, size=5120, key=1):
    """
    Generate mask of whole image by stitching smaller mask tiles, adjusting for padding.

    Args:
        path_ndpi (str): Path to the folder containing the NDPI files.
        path_masks (str): Path to the folder containing the mask tiles.
        name (str): Base name of the image.
        pad (list[int]): Padding [top, bottom, left, right] for the mask.
        size (int): Size of each mask tile. Defaults to 5120.
        key (int): TIFF image key for reading. Defaults to 1.

    Returns:
        tuple: Full mask (numpy.ndarray), width, and height of the original image.
    """

    w, h = large_size(path_ndpi, name, key=key)
    w_mask, h_mask = w+pad[2]+pad[3], h+pad[0]+pad[1]
    mask = np.zeros((w_mask, h_mask))
    print('Shape: ',  w, h, w_mask, h_mask)
    
    count   = 0
    for i in range(int(w_mask/size)):
        for j in range(int(h_mask/size)):
            print(i, j, count)
            img = cv2.imread(path_masks + name + '_5120_' + str(count) + '_mask.png')[:,:,0]
            mask[i*size:(i+1)*size, j*size:(j+1)*size] = img
            count += 1
        
    return mask[pad[2]:w+pad[2], pad[0]:h+pad[0]], w, h


def get_txt_from_masks(folder_ndpi, folder_masks, sid, s, pad, patch_size=512, key=1):
    """
    Extract tissue type fractions from mask tiles and save results in a text file.

    Args:
        folder_ndpi (str): Folder containing NDPI files.
        folder_masks (str): Folder containing mask tiles.
        sid (str): Sample ID.
        s (str): Sub-sample identifier.
        pad (list[int]): Padding [top, bottom, left, right] for the mask.
        patch_size (int): Size of individual patches for analysis. Defaults to 512.
        key (int): TIFF image key for reading. Defaults to 1.

    Returns:
        None. Saves tissue type proportions (epithelium, stroma, lumen) to a text file.
    """
    
    path_ndpi   = f'{folder_ndpi}/sid/' 
    path_masks  = f'{folder_masks}/sid/{s}_masks/' 
    name        = f'{sid}_{s}' 
    
    mask, w, h = get_mask(path_ndpi, path_masks, name, pad, size=5120, key=key)

    file = open(name+'.txt', 'w')
    file.close()   
    
    
    for i in range(int(w/patch_size)):
        for j in range(int(h/patch_size)):
            img         = mask[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] 
            epithelium  = np.count_nonzero(img == 1) /262144
            stroma      = np.count_nonzero(img == 2) /262144
            lumen       = 1-epithelium-stroma
        
            f = open(name+".txt", "a")
            f.write(str(epithelium)+'\t'+str(stroma)+'\t'+str(lumen)+'\t'+'\n')
            f.close()
            
            
def num_to_idx(num, w):
    """
    Converts a tile index into row & column indices based on the width of the original image.

    Args:
        num (int): Index
        w (int): Image width

    Returns:
        A tuple (i, j), where i is the row index and j is the column index.
    """
    
    i = int(num/w)
    j = np.mod(num,w)
    
    return (i,j)


def create_image(list, h, w):
    """
    Converts a 1D list into a 2D matrix of specified dimensions.

    Args:
        list (list): 1D list containing image data
        h (int): Image height 
        w (int): Image width 

    Returns:
        A 2D NumPy array of shape (h, w) populated with values from the list.
    """

    image = np.zeros((h,w))
    
    for i in range(h):
        for j in range(w):
            index       = i*w + j
            image[i,j]  = list[index]
            
    return image


def plot_single(img, name):
    """
    Displays a cell density map.

    Args:
        img (numpy.ndarray): Cell density map
        name (str): Title of plot
    """
    
    fig, ax = plt.subplots(figsize=(4, 3))
    img0 = ax.imshow(img, cmap='jet', vmin=0, vmax=10000)
    #ax.set_title(name)
    
    cb = fig.colorbar(img0, ax=ax)
    cb.ax.tick_params(labelsize=10)
    cb.set_label(u"Cells per \u03bcm\u00b2", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    
    ax.axis('off')   
    plt.show()  


def plot_both(img_density, img_fraction, name, mask_name):
    """
    Plots cell density, tissue fraction, and combined density-fraction images side by side.

    Args:
        img_density (numpy.ndarray): Cell density image.
        img_fraction (numpy.ndarray): Tissue fraction image.
        name (str): Title for the plot (currently unused but can be extended).
        mask_name (str): Label for the fraction type (e.g., 'Lumen', 'Stroma', etc.).
    """
    
    fig, axs = plt.subplots(1,3, figsize=(16,3))
    #plt.suptitle(name, fontsize=14)
    
    img0 = axs[0].imshow(img_density, cmap='jet',vmin=0,vmax=10000)
    cb = fig.colorbar(img0, ax=axs[0])
    cb.ax.tick_params(labelsize=10)
    cb.set_label(u"Cells per \u03bcm\u00b2", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    axs[0].axis('off')
    axs[0].set_title('Cell density')      
            
    img1 = axs[1].imshow(img_fraction, cmap='jet',vmin=0,vmax=1)
    cb = fig.colorbar(img1, ax=axs[1])
    cb.ax.tick_params(labelsize=10)
    cb.set_label(mask_name+" fraction", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    axs[1].axis('off')
    axs[1].set_title(mask_name)
    
    img2 = axs[2].imshow(img_density*img_fraction,cmap='jet',vmin=0,vmax=4000)
    cb = fig.colorbar(img2, ax=axs[2])
    cb.ax.tick_params(labelsize=10)
    cb.set_label(u"Cells per \u03bcm\u00b2", fontsize=12)
    cb.ax.tick_params(labelsize=12)
    axs[2].axis('off')
    axs[2].set_title(mask_name+' cell density')
    plt.show()


def save_img(out_folder, img, name, flip_v, flip_h, mask_name=None): 
    """
    Saves an image to disk (with optional flipping).


    Args:
        img (numpy.ndarray): Image to save.
        name (str): Base filename for the saved image.
        flip_v (bool): If True, flips the image vertically.
        flip_h (bool): If True, flips the image horizontally.
        mask_name (str, optional): Additional label for tissue fraction segmentation masks.
    """
    
    saved_img = img.copy()
    
    saved_img *= 255
    
    if flip_v:
        saved_img = cv2.flip(saved_img, 0)        
    if flip_h:
        saved_img = cv2.flip(saved_img, 1)
        
    if mask_name is None:
        saved_img /= 10000 
        cv2.imwrite(f'{out_folder}/{name}_density.png', saved_img)
    else:  
        cv2.imwrite(f'{out_folder}/{name}{mask_name}.png', saved_img) 
        

def get_density_and_fractions(out_folder, sample, csv_name, fractions=True, plot=False):
    """
    Processes cell density and tissue fraction (optional) data, plots (optional) and saves results.

    Args:
        sample (list): Metadata for the sample:
            sample[0] (str): Sample name.
            sample[1] (int): Image height.
            sample[2] (int): Image width.
            sample[3] (bool, optional): Vertical flip flag.
            sample[4] (bool, optional): Horizontal flip flag.
        sid (str): Sample ID.
        fractions (bool, default=True): If True, processes tissue fractions.
        plot (bool, default=False): If True, plots images.
    """
    
    print('*'*10, sid, '*'*10)
    
    try:
        flip_v = sample[3]
        flip_h = sample[4]
    except:
        flip_v = False
        flip_h = False
    
    ##### Cell density #####
    df           = pd.read_csv(f'./CSV/results_{csv_name}.csv')
    density_list = df[' count'].to_numpy()/0.216/1.15 # Area of pixel is 0.216 um^, normalisation factor of 1.15
    img_density  = create_image(density_list, sample[1], sample[2])
    
    ### plot fractions ###
    if plot == True:
        plot_single(img_density,sample[0])
        
    ## Save density ##
    save_img(out_folder, img_density, sample[0], flip_v, flip_h)

    if fractions == True:
        ##### Tissue fractions  #####
        segmentations = np.loadtxt('Segmentation/TXT/'+sample[0]+'.txt')
        epithelial  = create_image(segmentations[:,0], sample[1], sample[2])
        stroma      = create_image(segmentations[:,1], sample[1], sample[2])
        lumen       = create_image(segmentations[:,2], sample[1], sample[2])
        
        ### plot fractions ###
        if plot == True:
            plot_both(img_density, lumen,  sample[0],  'Lumen')
            plot_both(img_density, stroma, sample[0],  'Stroma')
            plot_both(img_density, epithelial,  sample[0],  'Epithelium')
        
        ## Save fractions ##
        save_img(out_folder, lumen,       sample[0], flip_v, flip_h, mask_name='_lumen')
        save_img(out_folder, stroma,      sample[0], flip_v, flip_h, mask_name='_stroma')
        save_img(out_folder, epithelial,  sample[0], flip_v, flip_h, mask_name='_epithelial')
    
    print('\n')
  
  
def resize_maps(out_folder, in_folder, sid, slice, map_name):
    """
    Resize a map image to match the dimensions of a reference image and save the result.

    Args:
        out_folder (str): Folder containing the reference scaled image.
        in_folder (str): Folder containing the map image to resize.
        sid (str): Sample ID.
        slice (str): Slice identifier for the image.
        map_name (str): Name of the map image to be resized.

    Returns:
        None. Saves the resized map image to the output folder.
    """

    path_histo  = f'{out_folder}/{sid}/{sid}_{slice}_scaled_SN.png'        
    path_in     = f'{in_folder}/{sid}/{sid}_{slice}_{map_name}.png' 
    path_out    = f'{out_folder}/{sid}/{sid}_{slice}_{map_name}.png' 
                
    histo      = cv2.imread(path_histo)
    density    = cv2.imread(path_in)
                
    h,w,_      = histo.shape
    density    = cv2.resize(density, (w,h), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(path_out, density)


