"""
This file contains a set of utility functions for processing medical images
particularly images in DICOM format

Author : E. Zemmouri
Date : 2022-04-06
"""

# %%

"""
Importing necessary packages
"""
import pydicom
import nibabel as nib 

import numpy as np
import matplotlib.pyplot as plt

import cv2
import os

from scipy import ndimage
from skimage import morphology
import matplotlib.image as mpimg

# %%

def displayImages(images, num_examples = 0, save_filename = None):
    """
    Display a list of images in a nice grid
    images may be of type list or numpy array
    """
    
    plt.close()
    # creates new figure
    fig = plt.figure(figsize=(10, 10))
    # Do not show axis
    plt.axis('off')

    if isinstance(images, list): n = len(images)
    else : n = images.shape[0]
    if num_examples == 0 or num_examples > n : num_examples = n
    from math import sqrt
    rows = int ( sqrt(num_examples) )
    if rows * rows < num_examples : rows += 1
    columns = rows
    
    for i in range(num_examples):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.axis('off')
        plt.imshow(images[i], cmap=plt.cm.gray)

    fig.tight_layout()
    if save_filename : plt.savefig(save_filename)
    plt.show()



def display_volume(images, num_examples = 0, save_filename = None):
    """
    Display a nibabel volume of images in a nice grid
    images is of type numpy array with 3 dimensions shape
    """
    
    plt.close()
    # creates new figure
    fig = plt.figure(figsize=(10, 10))
    # Do not show axis
    plt.axis('off')

    n = images.shape[-1]
    if num_examples == 0 or num_examples > n : num_examples = n
    from math import sqrt
    rows = int ( sqrt(num_examples) )
    if rows * rows < num_examples : rows += 1
    columns = rows
    
    for i in range(num_examples):
        ax = fig.add_subplot(rows, columns, i+1)
        ax.axis('off')
        plt.imshow(images[:,:,i], cmap=plt.cm.gray)
    
    fig.tight_layout()
    if save_filename : plt.savefig(save_filename)
    plt.show()


# %%

def transform_to_hu(ds):
    """
    Transform the pixel array of a dicom image to Hounsfield Unit
    ds : is an object of the dicom image obtained using pydicom.dcmread(dicom_file)
    Return : numpy array containing the transformed pixels
    """
    pixels = ds.pixel_array
    intercept = ds.RescaleIntercept
    slope = ds.RescaleSlope
    hu_pixels = pixels * slope + intercept
    return hu_pixels

def window_image_copy(pixels, window_center, window_width):
    """
    This function performs Windowing of a CT scan, also known as gray-level mapping
    Return : numpy array copy of pixels after windowing
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_pixels = pixels.copy()
    window_pixels[window_pixels < img_min] = img_min
    window_pixels[window_pixels > img_max] = img_max
    return window_pixels


def window_image(pixels, window_center, window_width):
    """
    This function performs Windowing of a CT scan, also known as gray-level mapping
    The windowing is performed directly on pixels parameter
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    pixels[pixels < img_min] = img_min
    pixels[pixels > img_max] = img_max
    return pixels


def CT_slices_selection (volume, tmin, tmax) :
    """
    This function performs slices selection from a CT scan
    """
    selected = []
    for i in volume.shape[-1]:

        im = volume[:,:,i]
        
        # Crop lung region and count the number of pixel values in the region [tmin, tmax]
        # Left Lobe
        pixels = im[160:350,110:200]
        counted_zero = np.count_nonzero ( pixels < tmax & pixels > tmin )
        if counted_zero / pixels.size > 0.6 : 
            selected.append(im)
        else :
            # Right Lobe
            pixels = im[160:350,260:350]
            counted_zero = np.count_nonzero ( pixels < tmax & pixels > tmin )
            if counted_zero / pixels.size > 0.6 : 
                selected.append(im)

    return np.stack( selected , axis=-1 )


def normalize_scan(volume):
    """
    This function normalizes a CT scan volume
    """
    min =  np.min(volume)
    max =  np.max(volume)
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

# %%

def resize_scan(volume, ratio):
    # Resize width, height 
    volume = ndimage.zoom(volume, (ratio, ratio, 1), order=1)
    return volume


def rotate_scan(volume, angle):
    # Rotate volume with angle  
    volume = ndimage.rotate(volume, angle, reshape=False)
    volume[volume < 0] = 0
    volume[volume > 1] = 1
    return volume

