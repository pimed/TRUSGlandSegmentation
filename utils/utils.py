"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
import cv2
import torch
import torch.nn as nn
import os
import SimpleITK as sitk
from medpy import metric
from skimage.morphology import dilation, disk, erosion, closing
from skimage.measure import label
#---------------------------------------Image Plotting----------------------------------------------------


def overlay_image(main, 
                  seg, 
                  pred=None):
    """ 
    Overlay segmentation output on the image
    
        img: input image e.g. Prostate MRI Slice
        seg: segmentation mask
        pred: whether we should overlay the prediction or not
    """
    # Load images as greyscale but make main RGB so we can annotate in colour
    main = cv2.cvtColor(main,cv2.COLOR_GRAY2BGR)
    
    seg = np.array(seg).astype(np.uint8)
    if pred is not None:
        pred = np.array(pred).astype(np.uint8)

    # Dictionary giving RGB colour for label (segment label) - label 1 in red, label 2 in yellow
    RGBforLabel = { 1:(0,255,255) }

    # Find external contours
    contours,_ = cv2.findContours(seg,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    # Iterate over all contours
    for i,c in enumerate(contours):
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(seg.shape, np.uint8)
        cv2.drawContours(mask,[c],-1,255, -1)

        # Get appropriate colour for this label
        #label = 1 if mean > 1.0 else 1
        colour = RGBforLabel.get(1)
        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(main,[c],-1,colour,1)
        
    if pred is not None:
        # Dictionary giving RGB colour for label (segment label) - label 1 in red, label 2 in yellow
        RGBforLabel = { 1:(0,0,255) }

        # Find external contours
        contours,_ = cv2.findContours(pred,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

        # Iterate over all contours
        for i,c in enumerate(contours):
            # Find mean colour inside this contour by doing a masked mean
            mask = np.zeros(seg.shape, np.uint8)
            cv2.drawContours(mask,[c],-1,255, -1)
            # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
            mean,_,_,_ = cv2.mean(seg, mask=mask)
            # DEBUG: print(f"i: {i}, mean: {mean}")

            # Get appropriate colour for this label
            #label = 1 if mean > 1.0 else 1
            colour = RGBforLabel.get(1)
            # Outline contour in that colour on main image, line thickness=1
            cv2.drawContours(main,[c],-1,colour,1)
    
    return main


def plot_images(images, 
                masks, 
                sample_toshow = 10, 
                phase='train'):
    """ 
    Plot images and masks 
    
        images: input image e.g. Prostate MRI Slice
        masks: segmentation masks
    """
    masks_binary = np.where(masks>0, 1, 0)
    
    count = 0
    i = 0
    fig,axs = plt.subplots(sample_toshow,5, figsize=(20,50))
    for i in range(200, sample_toshow+200):
        
        #k = random.randint(1,sample_toshow)
        k = i
        img = images [k]
        binary = masks_binary[k]
        mask = masks[k]

        axs[count][0].title.set_text("Prostate MRI")
        axs[count][0].imshow(img, cmap='gray')
        
        axs[count][1].title.set_text("Histo Intensities")
        axs[count][1].hist(img.ravel())

        axs[count][2].title.set_text("Mask")
        axs[count][2].imshow(mask, cmap='jet')
        
        axs[count][3].title.set_text("Binary Mask")
        axs[count][3].imshow(binary, cmap='gray')

        #img[binary==1] = 255  # change pixel color at the position of mask
        axs[count][4].title.set_text("MRI with Mask")
        axs[count][4].imshow(overlay_image(convert(img, 0, 255, np.uint8), 
                                                    convert(binary, 0, 1, np.uint8)))
        count +=1

    #plt.suptitle("Random {} images".format(phase), fontsize=24)
    plt.show()
    fig.tight_layout() 


def plot_images_series(images,
                       masks,
                       phase='train', start_img = 120):
    """ 
    Plot a sequence of image volume to see if the images and masks are in ocrrect order or not. 
    
        images: input image e.g. Prostate MRI Slice
        masks: segmentation masks
    """

    f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(18, 12))
    for i in range(20):
        img = images[i+start_img, :, :]
        msk = masks[i, :, :]
        plots[i // 5, i % 5].axis('off')
        plots[i // 5, i % 5].imshow(img, cmap=plt.cm.gray)
        plots[i // 5, i % 5].imshow(msk, cmap=plt.cm.jet, alpha =0.6)
        plots[i // 5, i % 5].set_title(str(i))
    plt.suptitle("Random {} images".format(phase), fontsize=24)
    plt.show()
    plt.tight_layout()

    
def plot_history(history_path):
    """
    Plot the training curves for loss and dice meterics.
    """
    
    history = pd.read_csv(history_path)
    plt.figure(figsize=(15,5))
    
    n_e = np.arange(len(history['dice_coe']))
    plt.plot(n_e,history['dice_coe'],'-o',label='Train dice_coe',color='#ff7f0e')
    plt.plot(n_e,history['val_dice_coe'],'-o',label='Val dice_coe',color='#1f77b4')
    x = np.argmax( history['val_dice_coe'] ); y = np.max( history['val_dice_coe'] )
    xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max dice_coe\n%.2f'%y,size=14)
    plt.ylabel('dice_coe',size=14); plt.xlabel('Epoch',size=14)
    plt.legend(loc=2)
    
    plt2 = plt.gca().twinx()
    plt2.plot(n_e,history['loss'],'-o',label='Train Loss',color='#2ca02c')
    plt2.plot(n_e,history['val_loss'],'-o',label='Val Loss',color='#d62728')
    x = np.argmin( history['val_loss'] ); y = np.min( history['val_loss'] )
    ydist = plt.ylim()[1] - plt.ylim()[0]
    plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
    plt.ylabel('Loss',size=14)
    plt.legend(loc=3)
    plt.show()


def plot_3views(img3d, seg):
    """
    Plot the training curves for loss and dice meterics.
    """
    img_shape = img3d.shape
    # plot 3 orthogonal slices
    plt.figure(figsize=(15,5))
    a1 = plt.subplot(1, 3, 1)
    plt.imshow(img3d[:, :, img_shape[2]//2], cmap='gray')
    plt.imshow(seg[:, :, img_shape[2]//2], cmap='jet', alpha=0.5)
    plt.title("axial")

    a2 = plt.subplot(1, 3, 2)
    plt.imshow(img3d[:, img_shape[1]//2, :], cmap='gray')
    plt.imshow(seg[:, img_shape[1]//2, :], cmap='jet', alpha=0.5)
    plt.title("sagittal")

    a3 = plt.subplot(1, 3, 3)
    plt.imshow(img3d[img_shape[0]//2, :, :].T, cmap='gray')
    plt.imshow(seg[img_shape[0]//2, :, :].T, cmap='jet', alpha=0.5)
    plt.title("coronal")

    plt.show()


def to_categorical(mask, num_classes, channel='channel_first'):
    assert mask.ndim == 4, "mask should have 4 dims"
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)
    assert len(unique) <= num_classes, "number of unique values should be smaller or equal to the num_classes"
    assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
    # shape = mask.shape
    #     # h = shape[1]
    #     # w = shape[2]
    #     # if channel == 'channel_first':
    #     #     output = np.zeros((shape[0], num_classes, h, w))
    #     #     for i in range(num_classes):
    #     #         output[:,i] = mask == i
    #     # else:
    #     #     output = np.zeros((shape[0], h, w, num_classes))
    #     #     for i in range(num_classes):
    #     #         output[...,i] = mask == i
    if mask.shape[1] == 1:
        mask = np.squeeze(mask, axis=1)
    if mask.shape[-1] == 1:
        mask = np.squeeze(mask, axis=-1)
    eye = np.eye(num_classes, dtype='uint8')
    output = eye[mask]
    if channel == 'channel_first':
        output = np.moveaxis(output, -1, 1)
    return output


def soft_to_hard_pred(pred, channel_axis=1):
    max_value = np.max(pred, axis=channel_axis, keepdims=True)
    return np.where(pred==max_value, 1, 0)


def remove_files(directory='../weights/*'):
    import os, glob
    files = glob.glob(directory)
    for f in files:
        print(f)
        os.remove(f)
    print("Files removed")


def plot_slices(data_vol, label_vol):
    """
    :return:
    """
    f, plots = plt.subplots(4, 5, sharex='col', sharey='row', figsize=(10, 8))
    for i in range(20):
        intt = np.random.choice(data_vol.shape[0])

        plots[i // 5, i % 5].axis('off')
        plots[i // 5, i % 5].imshow(data_vol[intt, 0, :, :], cmap=plt.cm.bone)
        plots[i // 5, i % 5].imshow(label_vol[intt, 0, :,:], alpha=0.5)
    plt.show()


def load_nii(img_path):
    """
    Function to load a 'nii' or 'nii.gz' file, The function returns
    everyting needed to save another 'nii' or 'nii.gz'
    in the same dimensional space, i.e. the affine matrix and the header

    Parameters
    ----------

    img_path: string
    String with the path of the 'nii' or 'nii.gz' image file name.

    Returns
    -------
    Three element, the first is a numpy array of the image values,
    the second is the affine transformation of the image, and the
    last one is the header of the image.
    """
    nimg = nib.load(img_path)
    return nimg.get_data(), nimg.affine, nimg.header


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''
    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1]:
        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        if struc_id == 1:
            props = measure.regionprops(blobs)

            if not props:
                continue

            area = [ele.area for ele in props]
            largest_blob_ind = np.argmax(area)
            largest_blob_label = props[largest_blob_ind].label

            out_img[blobs == largest_blob_label] = struc_id
        else:
            connected = closing(binary_img)
            lesions, num_lesions = label(connected, return_num=True, connectivity=2)
            for lesion in range(num_lesions):
                if np.sum(lesions == (lesion + 1)) < 50:
                    lesions[lesions == (lesion + 1)] = 0
            lesions_mask = np.copy(lesions)
            lesions_mask[lesions_mask > 0] = 1
            lesions, num_lesions = label(lesions_mask, return_num=True, connectivity=2)
            out_img[blobs == lesions] = struc_id
    return out_img


def generate_lesions(ref_vol, label_np_source, volume_thresh, m_disks):

    label_np = np.copy(label_np_source)

    label_np[label_np != 0] = 1

    spacing = ref_vol.GetSpacing()

    margin = m_disks[0] / spacing[0]
    strel = disk(int(margin))

    margin = m_disks[1] / spacing[0]
    strel2 = disk(int(margin))
    strel2 = np.pad(strel2, int(m_disks[2]), 'constant')

    strel_total = np.stack([strel2, strel, strel2])

    connected = closing(label_np, strel_total)

    lesions, num_lesions = label(connected, return_num=True, connectivity=2)

    for lesion in range(num_lesions):
        if spacing[0] * spacing[1] * spacing[2] * np.sum(lesions == (lesion + 1)) < volume_thresh:
            lesions[lesions == (lesion + 1)] = 0

    lesions_mask = np.copy(lesions)
    lesions_mask[lesions_mask > 0] = 1
    lesions, num_lesions = label(lesions_mask, return_num=True, connectivity=2)

def resize_volume(img_volume, w=288, h=288, view='axial', is_img = True):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    if view == 'axial':
        z = img_volume.shape[2]

    for i in range(z):
        if view == 'axial':
            if is_img and i == 0:
                slice_img = img_volume[:, :, 0:i + 3]
            elif is_img and i == z - 1:
                slice_img = img_volume[:, :, i - 2:i + 1]
            elif is_img:
                slice_img = img_volume[:, :, i - 1:i + 2]
            else:
                slice_img = img_volume[:, :, i]
        img_res.append(cv2.resize(slice_img, dsize=(h, w), interpolation=cv2.INTER_AREA))

    return np.array(img_res)


def resize_volume_back(vol, h=224, w=288, view ='axial'):
    """
    :param vol:
    :return:

    """
    img_shape = vol.shape
    resized_vol = []
    z = img_shape[0]
    for i in range(z):
        view_slice = vol[i, :, :]
        resized_vol.append(cv2.resize(view_slice, (w, h), cv2.INTER_AREA))
    return np.array(resized_vol)


def preprocess_volume(img_volume):

    """
    :param img_volume: A patient volume
    :return: applying CLAHE and Bilateral filter for contrast enhacnmeent and denoising

    """
    prepross_imgs = []
    for i in range(len(img_volume)):
        img = img_volume[i]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        cl1 = clahe.apply(img)
        prepross_imgs.append(cl1)

    return np.array(prepross_imgs)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def _adjust_learning_rate(optimizer, i_iter, learning_rate):
    lr = lr_poly(learning_rate, i_iter, 250000, 0.9)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate(optimizer, i_iter, cfg, learning_rate=2.5e-4):
    """ adject learning rate for main segnet
    """
    _adjust_learning_rate(optimizer, i_iter, learning_rate=learning_rate)


def adjust_learning_rate_discriminator(optimizer, i_iter, cfg):
    _adjust_learning_rate(optimizer, i_iter, learning_rate=1e-4)


def metrics(img_gt, img_pred, voxel_size, label_index = 1):
    """
    Adapted: https://www.creatis.insa-lyon.fr/Challenge/acdc/code/metrics_acdc.py

    Function to compute the metrics between two segmentation maps given as input.
    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.
    img_pred: np.array
    Array of the predicted segmentation map.
    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.
    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    res = []
    # Loop on each classes of the input images
    for c in [label_index]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0
        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Compute the Dice
        dice = metric.binary.dc(gt_c_i, pred_c_i)

        # Compute the Dice
        jac = metric.binary.jc(gt_c_i, pred_c_i)

        # Compute the Housdurf distance
        try:
            hdd = metric.binary.hd95(gt_c_i, pred_c_i, voxelspacing=voxel_size)
        except RuntimeError:
            print("RuntimeError('The first supplied array does not contain any binary object.')")
            hdd = 0

        # Compute the Assymetricd distance
        try:
            assd = metric.binary.assd(gt_c_i, pred_c_i, voxelspacing=voxel_size)
        except RuntimeError:
            print("RuntimeError('The first supplied array does not contain any binary object.')")
            assd = 0

        # Compute the Housdurf distance
        sen = metric.binary.sensitivity(gt_c_i, pred_c_i)

        # Compute the Housdurf distance
        spe = metric.binary.specificity(gt_c_i, pred_c_i)

        # Clip the value to compute the volumes
        gt_c_i = np.clip(gt_c_i, 0, 1)
        pred_c_i = np.clip(pred_c_i, 0, 1)

        # Compute the Dice
        # Computes the linear correlation in binary object volume between the contents
        # of the successive binary images supplied. Measured through the
        # Pearson product-moment correlation coefficient.
        vol, p_value = metric.binary.volume_correlation(gt_c_i, pred_c_i)

        # Compute volume
        volgt = gt_c_i.sum() * np.prod(voxel_size) / 1000.
        volpred = pred_c_i.sum() * np.prod(voxel_size) / 1000.

        res += [dice,jac, hdd, assd, sen, spe, vol, p_value, volgt, volpred, volpred-volgt]
    return res


def metrics2(img_gt, img_pred, voxel_size, label_index = 1):
    """
    Adapted: https://www.creatis.insa-lyon.fr/Challenge/acdc/code/metrics_acdc.py

    Function to compute the metrics between two segmentation maps given as input.
    Parameters
    ----------
    img_gt: np.array
    Array of the ground truth segmentation map.
    img_pred: np.array
    Array of the predicted segmentation map.
    voxel_size: list, tuple or np.array
    The size of a voxel of the images used to compute the volumes.
    Return
    ------
    A list of metrics in this order, [Dice LV, Volume LV, Err LV(ml),
    Dice RV, Volume RV, Err RV(ml), Dice MYO, Volume MYO, Err MYO(ml)]
    """
    if img_gt.ndim != img_pred.ndim:
        raise ValueError("The arrays 'img_gt' and 'img_pred' should have the "
                         "same dimension, {} against {}".format(img_gt.ndim,
                                                                img_pred.ndim))
    res = []
    # Loop on each classes of the input images
    for c in [label_index]:
        # Copy the gt image to not alterate the input
        gt_c_i = np.copy(img_gt)
        gt_c_i[gt_c_i != c] = 0
        # Copy the pred image to not alterate the input
        pred_c_i = np.copy(img_pred)
        pred_c_i[pred_c_i != c] = 0

        # Compute the Dice
        dice = metric.binary.dc(gt_c_i, pred_c_i)

        # Compute the Housdurf distance
        try:
            hdd = metric.binary.hd95(pred_c_i, gt_c_i, voxelspacing=voxel_size)
        except RuntimeError:
            print("RuntimeError('The first supplied array does not contain any binary object.')")
            hdd = 0

        # Compute the Housdurf distance
        sen = metric.binary.sensitivity(pred_c_i, gt_c_i)

        # Compute the Housdurf distance
        spe = metric.binary.specificity(pred_c_i, gt_c_i)
        # Compute precision
        prec = metric.binary.precision(pred_c_i, gt_c_i)
        # Compute recall
        reca = metric.binary.recall(pred_c_i, gt_c_i)

        # Compute True Negative
        tnr = metric.binary.true_negative_rate(pred_c_i, gt_c_i)

        # Compute true positive rate
        tpr = metric.binary.true_positive_rate(pred_c_i, gt_c_i)

        res += [dice, hdd, sen, spe, prec, reca, tpr, tnr]
    return res


def compute_metrics_on_files(path_gt, gt, pred, zooms, header=True, label_index = 1):
    """
    Function to give the metrics for two files
    Parameters
    ----------
    path_gt: string
    Path of the ground truth image.
    path_pred: string
    Path of the predicted image.
    """

    HEADER = ["Name                   ", "Dice [%] ", "Jac [%] ", "HD [mm]", "ASSD [mm]",
              "Sensitivity [%]", "Specificity [%]", "Vol-Corr ", "P-Value", "gt-Vol", "pred-vol", "vol-err"]

    name = os.path.basename(path_gt)
    name = name.split('.')[0]
    res = metrics(gt, pred, zooms, label_index=label_index)
    res = ["{:.3f}".format(r) for r in res]
    formatting = "{:>8}, {:>8}, {:>10}, {:>10}, {:>10}, {:>10},  {:>10}, {:>10}, {:>10}"
    if header:
        print(formatting.format(*HEADER))
    print(formatting.format(name, *res))


if __name__ == '__main__':

    pred = np.random.rand(2, 3, 3)
    print(pred)
    print(soft_to_hard_pred(pred, 0))
    input()

    eye = np.eye(3, dtype='uint8')
    mask = np.array([[1,1,1,1],[1,2,2,1],[1,2,3,1],[1,1,1,1]]) - 1
    print(mask)
    mask1 = np.array([[2,2,2,2],[1,1,2,2],[1,1,1,1],[3,3,3,3]]) - 1
    print(mask1)
    mask = np.array([mask, mask1])
    mask = to_categorical(mask=mask, num_classes=3, channel='channel_first')
    input()
