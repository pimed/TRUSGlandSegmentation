"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2021

"""

import warnings
warnings.filterwarnings("ignore")

#import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import torchio as io

def resample_image(image, desired_voxel_spacing, info=False):
    original_image_spacing = image.GetSpacing()

    if original_image_spacing != desired_voxel_spacing:
        ### HOW TO RESAMPLE SITK_IMAGE TO A NEW SPACING ###
        ### SOURCE: https://github.com/SimpleITK/SimpleITK/issues/561 ###

        # converting to np array for calculations of new_size
        original_size_array = np.array(image.GetSize(), dtype = np.int)
        original_spac_array = np.array(image.GetSpacing())
        desired_spac_array = np.array(desired_voxel_spacing)

        new_size = original_size_array * (original_spac_array / desired_spac_array)
        new_size = np.ceil(new_size).astype(np.int)
        new_size = [int(s) for s in new_size]
        
        # convert new_size back to tuple

        #new_size = [60, new_size[1], new_size[2]]
        new_size = tuple(new_size)
        # create the resample filter
        resample = sitk.ResampleImageFilter()
        resample.SetInterpolator(sitk.sitkLinear) 
        resample.SetSize(new_size)
        resample.SetOutputOrigin(image.GetOrigin()) 
        resample.SetOutputSpacing(desired_voxel_spacing)
        resample.SetOutputDirection(image.GetDirection())

        resampled_image = resample.Execute(image)

        # print the changes
        if info:
            print('Resampling:', source_file_path.parts[-3:-1])
            print('original spacing:', image.GetSpacing())
            print('desired spacing:', desired_voxel_spacing)
            print('resampled spacing:', resampled_image.GetSpacing())
            print('original size:', image.GetSize())
            print('resampled size:', resampled_image.GetSize())
            print('\n')
    else:
        resampled_image = image
    return resampled_image

def get_slice_location(dcm):
    """ 
    Reutrn dicom slice location.
    
        dcm: Dicom File
    """
    return float(dcm[0x0020, 0x1041].value)
