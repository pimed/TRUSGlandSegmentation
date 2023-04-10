"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2021

"""
from __future__ import print_function, division
import numpy as np
import pandas as pd
import glob
import os
import torch
import torch as t
import random
from datetime import datetime
import argparse
from utilss import *
from model.dilated_unet import Segmentation_model
from model.unet_variants import AttU_Net, NestedUNet
from scipy.ndimage import median_filter, gaussian_filter
import SimpleITK as sitk
import imgaug.augmenters as iaa
from utils.resampling import resample_image


def get_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_test_dataset_ultrasound(args,
                                img_path,
                                mask_path,
                                h=128,
                                w=160,
                                binary_seg=True,
                                resampling=False):
    """
    Read and process TRUS images and corresponding masks
    :param img_path: TRUS .nii image path
    :param mask_path: TRUS .nii/.nrrd segmentation path
    :param h: the image height after resizing
    :param w: the image width after resizing
    :param binary_seg: make the segmentation mask to binary
    :param resampling: whether resample TRUS images
    :return: resize and resampled images and masks, original image spacing and origin
    """

    try:
        seg_image = sitk.ReadImage(mask_path)
    except TypeError:
        print("the segmentation mask has problem..!")
    try:
        us_image = sitk.ReadImage(img_path)
    except TypeError:
        print("the ultrasound nii image has problem")
    original_image_spacing = us_image.GetSpacing()
    original_image_origin = us_image.GetOrigin()

    if resampling:
        if list(original_image_spacing) != [0.25, 0.25, 0.25]:
            print("resampled")
            us_image = resample_image(us_image, [0.25, 0.25, 0.25])
            seg_image = resample_image(seg_image, [0.25, 0.25, 0.25])

    us_array = sitk.GetArrayFromImage(us_image)
    seg_array = sitk.GetArrayFromImage(seg_image)

    outputshape = np.shape(seg_array)
    #if seg_array.shape[1] != seg_array.shape[2] and us_array.shape[1] != us_array.shape[2]:
    #    return None

    # Resize ultrasound volumes based on axial view
    us_array_orig = us_array
    us_array = resize_volume(us_array, h, w, view=args.ultrasound_view, is_img=True)

    if binary_seg:
        seg_array = np.where(seg_array > 1, 1, seg_array)
    # print("Image:", us_array.shape, seg_array.shape, np.unique(seg_array), us_image.GetSpacing(), original_image_spacing)
    return np.array(us_array), np.array(
        seg_array), us_array_orig, outputshape, original_image_spacing, original_image_origin


def inference(args,
              test_df,
              model,
              apply_clahe=True,
              postpro=True,
              resampling=False):
    """
    Model prediction for each volume
    :param test_df: pandas dataframe that includes image/mask paths
    :param model: segmentation model
    :param batch_size: the number of batch-size
    :param apply_clahe: whether to apply clahe as histogram equalization for input images
    :param cohort: the name of cohort test data -> Stanford/UCLA/UCL
    :param postpro: whether apply any image filtering for smoothing prediction boundaries
    :return:
    """
    metric_output = []
    for i in range(len(test_df)):
        image_path = str(test_df.imagePath.iloc[i])
        mask_path = str(test_df.maskPath.iloc[i])
        try:
            image, mask, image_orig, output_shape, spacing, origin = get_test_dataset_ultrasound(args,
                                                                                                 image_path,
                                                                                                 mask_path)
        except TypeError:
            print("Corrupted images moving to the next one.!")
            continue

        if apply_clahe:
            seq = iaa.Sequential([iaa.CLAHE(1.0)])  # apply histogram equalization
            image = seq(images=image)

        image = np.array(image, np.float32) / 255.  # normalize the volume, min-max normalization
        image = np.moveaxis(image, -1, 1)

        total_pred = []
        for j in range(0, len(image), args.batch_size):
            chunk = image[j:j + args.batch_size, ...]
            # plot_gradcam(model, chunk)
            prediction = model.forward(t.tensor(chunk))#.cuda())
            y_pred = prediction.cpu().detach().numpy()
            total_pred.extend(y_pred)
        prediction = np.array(total_pred)

        y_pred = np.argmax(np.moveaxis(prediction, 1, -1), axis=-1)
        y_pred = keep_largest_connected_components(mask=y_pred)

        # whether apply any postprocessing step like median filter to smooth out the boundaries
        if postpro:
            y_pred = median_filter(y_pred, size=7)
            # y_pred = gaussian_filter(y_pred, sigma=0.1)

        # here we resize the volume back to it's original shape based on different view.
        if args.ultrasound_view == 'sagittal':
            y_pred = resize_volume_back(y_pred, h=output_shape[0], w=output_shape[2], view=args.ultrasound_view)
            y_pred = np.moveaxis(y_pred, 1, 0)
        elif args.ultrasound_view == 'axial':
            y_pred = resize_volume_back(y_pred, h=output_shape[0], w=output_shape[1], view=args.ultrasound_view)
            y_pred = np.moveaxis(y_pred, 1, 0)
            y_pred = np.moveaxis(y_pred, 1, 2)
        elif args.ultrasound_view == 'coronal':
            y_pred = resize_volume_back(y_pred, h=output_shape[1], w=output_shape[2], view=args.ultrasound_view)
            y_pred = np.moveaxis(y_pred, 1, 2)


        # Compute the evaluation metrics
        # if it's the first iter, plot the header
        if i == 0:
            res_metric = compute_metrics_on_files(image_path, y_pred, mask, voxel_size=spacing[1], header=True)
        else:
            res_metric = compute_metrics_on_files(image_path, y_pred, mask, voxel_size=spacing[1], header=False)

        # Save the prediction as .nii file
        if resampling:
            out = sitk.GetImageFromArray(y_pred)
            out.SetSpacing(spacing)
            out.SetOrigin(origin)
        else:
            out = sitk.GetImageFromArray(y_pred)
            out.SetSpacing([0.25, 0.25, 0.25])
            out.SetOrigin(origin)
            out = resample_image(out, spacing)
        sitk.WriteImage(out, "results/{}/{}".format(args.cohort, os.path.basename(image_path.replace('.nii.gz',
                                                                                                     '_pred.nii.gz'))))
        metric_output.append(res_metric)

    df_out = pd.DataFrame(metric_output, columns=["Name",
                                                  "Dice",
                                                  "Jaccard",
                                                  "HD",
                                                  "ASSD",
                                                  "Sensitivity",
                                                  "Specificity",
                                                  "Vol-Corr",
                                                  "P-Value",
                                                  "GT-Vol",
                                                  "Pred-Vol",
                                                  "Vol-Error"])
    df_out.to_csv("results/UCL_evaluation2.xlsx", header=True)
    print(df_out.describe())


if __name__ == '__main__':
    get_seed()
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    print()

    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')

    parser = argparse.ArgumentParser()
    parser.add_argument("-basedir", help="the path to dataset folder", type=str, default="input/")
    parser.add_argument("-mt", "--model_type", help="which model should use for prediction", type=str,
                        default="drunet2.5D")
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=-1)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=32)
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=2)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for DR-UNET", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=3)
    parser.add_argument("-in", "--in_channel", help="number input channels", type=int, default=3)
    parser.add_argument("-us_view", "--ultrasound_view", help="the view of ultrasound image", type=str, default='axial')
    parser.add_argument("-co", "--cohort", help="the cohort name for test dataset", type=str, default='UCL')
    args = parser.parse_args()

    data = []
    #python -u prediction.py --model_type drunet2.5D

    args.model_type == 'drunet2.5D'
    
    for img_path, mask_path in zip(sorted(glob.glob(os.path.join(args.basedir, "images/*.nii.gz"))),
                                   sorted(glob.glob(os.path.join(args.basedir, "masks/*.nii.gz")))):
        base_name = os.path.split(img_path)
        data.append([img_path, mask_path, base_name])
    test_df = pd.DataFrame(data, columns=['imagePath', 'maskPath', 'subject_ID'])
    

    if args.model_type == 'drunet2.5D':
        model = Segmentation_model(filters=args.n_filter,
                                   in_channels=args.in_channel,
                                   n_block=args.n_block,
                                   n_class=args.n_class)
        model.load_state_dict(
            torch.load('weights/prostateUS.unet_lde_all_lr_0.0001_32.gaussian_noise/unet_model_checkpoint.pt', map_location='cpu'))
        print("DR-UNet2.5D model is loaded...!")
        
    elif args.model_type == 'coorddrunet2.5D':
        model = Segmentation_model(filters=args.n_filter,
                                   in_channels=args.in_channel,
                                   n_block=args.n_block,
                                   n_class=args.n_class,
                                   attention = True)
        model.load_state_dict(
            torch.load('weights/prostateUS.unetcoord_100Per_lesion_LKD_lr_1e-05_32.gaussian_noise/unet_model_checkpoint.pt', map_location='cpu'))
        print("DR-UNet2.5D model is loaded...!")
    elif args.model_type == 'attunet2.5D':
        model = AttU_Net(img_ch=args.in_channel, output_ch=args.n_class)
        model.load_state_dict(
            torch.load('weights/prostateUS.attunet_100Per_lr_0.0001_32.gaussian_noise/unet_model_checkpoint.pt'),)
        print("Att-UNet2.5 model is loaded...!")
    else:
        model = NestedUNet(in_ch=args.in_channel, out_ch=args.n_class)
        model.load_state_dict(
            torch.load('weights/prostateUS.nestedunet_100Per_lr_0.0001_32.gaussian_noise/unet_model_checkpoint.pt'))
        print("NestedUNet2.5D model is loaded...!")

    model.eval() #.cuda()
    start = datetime.now()
    t.autograd.set_detect_anomaly(True)
    inference(args, test_df, model)
    end = datetime.now()
    print("time elapsed for training (hh:mm:ss.ms) {}".format(end - start))
