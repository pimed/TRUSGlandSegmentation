"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2021

"""
import numpy as np
from skimage import measure
import cv2
import os
from medpy import metric

def overlay_image(img, seg, pred=None):
    """
    Overlay segmentation output on the image
    :param img: input image e.g. Prostate MRI Slice
    :param seg: segmentation mask
    :param pred: whether we should overlay the prediction or not
    :return:
    """
    # Load images as greyscale but make main RGB so we can annotate in colour
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    seg = np.array(seg).astype(np.uint8)
    if pred is not None:
        pred = np.array(pred).astype(np.uint8)

    # Dictionary giving RGB colour for label (segment label) - label 1 in red, label 2 in yellow
    RGBforLabel = {1: (0, 255, 255)}

    # Find external contours
    contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Iterate over all contours
    for i, c in enumerate(contours):
        # Find mean colour inside this contour by doing a masked mean
        mask = np.zeros(seg.shape, np.uint8)
        cv2.drawContours(mask, [c], -1, 255, thickness = 3)

        # Get appropriate colour for this label
        # label = 1 if mean > 1.0 else 1
        colour = RGBforLabel.get(1)
        # Outline contour in that colour on main image, line thickness=1
        cv2.drawContours(img, [c], -1, colour, thickness = 3)

    if pred is not None:
        # Dictionary giving RGB colour for label (segment label) - label 1 in red, label 2 in yellow
        RGBforLabel = {1: (0, 0, 255)}

        # Find external contours
        contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Iterate over all contours
        for i, c in enumerate(contours):
            # Find mean colour inside this contour by doing a masked mean
            mask = np.zeros(seg.shape, np.uint8)
            cv2.drawContours(mask, [c], -1, 255, thickness = 3)
            # DEBUG: cv2.imwrite(f"mask-{i}.png",mask)
            mean, _, _, _ = cv2.mean(seg, mask=mask)
            # DEBUG: print(f"i: {i}, mean: {mean}")

            # Get appropriate colour for this label
            # label = 1 if mean > 1.0 else 1
            colour = RGBforLabel.get(1)
            # Outline contour in that colour on main image, line thickness=1
            cv2.drawContours(img, [c], -1, colour, thickness = 3)
    return img


def to_categorical(mask, num_classes, channel='channel_first'):
    """
    :param mask: binary mask image with size WxHxC
    :param num_classes: number of classes within the binary mask
    :param channel: whether the channel is first or last
    :return:
    """
    
    assert mask.ndim == 4, "mask should have 4 dims"
    if channel != 'channel_first' and channel != 'channel_last':
        assert False, r"channel should be either 'channel_first' or 'channel_last'"
    assert num_classes > 1, "num_classes should be greater than 1"
    unique = np.unique(mask)
    assert len(unique) <= num_classes, "number of unique values should be smaller or equal to the num_classes"
    assert np.max(unique) < num_classes, "maximum value in the mask should be smaller than the num_classes"
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
    """
    :param pred: model prediction with size BxCxWxH
    :param channel_axis: which axis the data should be converted
    :return: converting the soft probablity prediction to hard format
    """

    max_value = np.max(pred, axis=channel_axis, keepdims=True)
    return np.where(pred == max_value, 1, 0)

def keep_largest_connected_components(mask):
    """
     Keeps only the largest connected components of each label for a segmentation mask.
    :param mask: model prediction after binarization with size BxCxWxH
    :return: removed the small over/under segmented regions from each class
    """

    out_img = np.zeros(mask.shape, dtype=np.uint8)
    for struc_id in [1]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)
        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label
        out_img[blobs == largest_blob_label] = struc_id
    return out_img


def resize_volume(vol, w=128, h=160, view='axial', is_img=True, cohort='UCL'):

    """
    :param vol: the ultrasound volume with size WxHxC
    :param w: the image width after resizing
    :param h: the image height after resizing
    :param view: from which view the image should be resized: axial, coronal or sagittal
    :param is_img: wether the img_volume is a segmentation or image volume
    :return: resized volume
    """
    img_res = []
    if cohort =='UCL':
        # we loop over the last dimension,to get axial view images, since UCL data is different the axial view
        # will be (z, x, y). Not a fancy indexing, can be corrected later
        z = vol.shape[0]
        for i in range(z):
            if view == 'axial':
                if is_img and i == 0:
                    slice_img = vol[0:i + 3, :, :]
                    slice_img = np.moveaxis(slice_img, 0, 1)
                    slice_img = np.moveaxis(slice_img, 1, 2)
                elif is_img and i == z - 1:
                    slice_img = vol[i - 2:i + 1, :, :]
                    slice_img = np.moveaxis(slice_img, 0, 1)
                    slice_img = np.moveaxis(slice_img, 1, 2)
                elif is_img:
                    slice_img = vol[i - 1:i + 2, :, :]
                    slice_img = np.moveaxis(slice_img, 0, 1)
                    slice_img = np.moveaxis(slice_img, 1, 2)
                else:
                    slice_img = vol[i, :, :]
            # need the rotation of needle sequences as it wrongly processed
            # TODO: Anirudh will update the processed data so we don't need rotation.
            img_res.append(cv2.resize(np.rot90(slice_img, 3), dsize=(h, w), interpolation=cv2.INTER_NEAREST))
        return np.array(img_res)
    elif cohort =='NEEDELSEQ':
        # we loop over the last dimension,to get axial view images, UCLA and Stanford axial view will be (x, y, z)
        z = vol.shape[0]
        # loop over z axis to extract 3 channel inputs for the model.
        for i in range(z):
            if is_img and i == 0:
                slice_img = vol[0:i + 3, ...]
            elif is_img and i == z - 1:
                slice_img = vol[i - 2:i + 1, ...]
            elif is_img:
                slice_img = vol[i - 1:i + 2, ...]
            else:
                slice_img = vol[i, ...]
            slice_img = np.moveaxis(slice_img, 0, 1)
            slice_img = np.moveaxis(slice_img, 1, 2)
            img_res.append(cv2.resize(slice_img, dsize=(h, w), interpolation=cv2.INTER_NEAREST))
        return np.array(img_res)
    else:
        # we loop over the last dimension,to get axial view images, UCLA and Stanford axial view will be (x, y, z)
        z = vol.shape[2]
        # loop over z axis to extract 3 channel inputs for the model.
        for i in range(z):
            if is_img and i == 0:
                slice_img = vol[:, :, 0:i + 3]
            elif is_img and i == z - 1:
                slice_img = vol[:, :, i - 2:i + 1]
            elif is_img:
                slice_img = vol[:, :, i - 1:i + 2]
            else:
                slice_img = vol[:, :, i]
            img_res.append(cv2.resize(slice_img, dsize=(h, w), interpolation=cv2.INTER_NEAREST))

        return np.array(img_res)


def resize_volume_back(vol, w=288, h=224, rotate_slice = False, flip_slice=False):
    """
    :param vol: the ultrasound volume with size WxHxC
    :param w: the image width after resizing
    :param h: the image height after resizing
    :param view: from which view the image should be resized: axial, coronal or sagittal
    :param flip_slice: visually sea if the predictions need to be flipped horzintally, as the model somtimes segment
    out of filed of view and I thought maybe slices are flipped due to rotation.
    :return: resize the model prediction to its original volume size

    """
    img_shape = vol.shape
    resized_vol = []
    z = img_shape[0]
    for i in range(z):
        view_slice = vol[i, :, :]
        if flip_slice:
            resized_vol.append(cv2.resize(np.fliplr(view_slice), (w, h), cv2.INTER_AREA))
        elif rotate_slice:
            resized_vol.append(cv2.resize(np.rot90(view_slice, 1), (w, h), cv2.INTER_AREA))
        else:
            # need the rotation of needle sequences as it wrongly processed
            # TODO: Anirudh will update the processed data so we don't need rotation.
            resized_vol.append(cv2.resize(view_slice, (w, h), cv2.INTER_AREA))
    return np.array(resized_vol)


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


def compute_metrics_on_files(path_gt, gt, pred, voxel_size, header=True, label_index = 1):
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
    res = metrics(gt, pred, voxel_size, label_index=label_index)
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
    mask = np.array([[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 3, 1], [1, 1, 1, 1]]) - 1
    print(mask)
    mask1 = np.array([[2, 2, 2, 2], [1, 1, 2, 2], [1, 1, 1, 1], [3, 3, 3, 3]]) - 1
    print(mask1)
    mask = np.array([mask, mask1])
    mask = to_categorical(mask=mask, num_classes=3, channel='channel_first')
    input()
