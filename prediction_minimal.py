"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2023

"""
from __future__ import print_function, division

import matplotlib.pyplot as plt
import pandas as pd
import glob
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


def get_test_dataset_ultrasound(img_path,
                                h=128,
                                w=160,
                                resampling=False,
                                plot=True):
    """
    Read and process TRUS images and corresponding masks
    :param plot:
    :param img_path: TRUS .nii image path
    :param mask_path: TRUS .nii/.nrrd segmentation path
    :param h: the image height after resizing
    :param w: the image width after resizing
    :param binary_seg: make the segmentation mask to binary
    :param resampling: whether resample TRUS images
    :return: resize and resampled images and masks, original image spacing and origin
    """
    # read the image
    us_image = sitk.ReadImage(img_path, sitk.sitkInt16)
    # save the image data info
    original_image_spacing = us_image.GetSpacing()
    original_image_origin = us_image.GetOrigin()
    original_image_direction = us_image.GetDirection()

    # whether to resample the image into desired spacing
    if resampling:
        if list(original_image_spacing) != args.image_spacing:
            print("resampled")
            us_image = resample_image(us_image, args.image_spacing)

    # get numpy array from ITK image
    us_array = sitk.GetArrayFromImage(us_image)
    outputshape = np.shape(us_array)

    # plot slices for sanity check before resizing
    if plot:
        plt.imshow(us_array[20, ...], cmap='gray')
        plt.show()
    # Resize ultrasound volumes based on axial view
    us_array = resize_volume(us_array, h, w, cohort=args.cohort, is_img=True)
    # plot slices for sanity check after resizing
    if plot:
        plt.imshow(us_array[20, ...], cmap='gray')
        plt.show()
    return np.array(us_array), outputshape, original_image_spacing, original_image_origin, original_image_direction


def inference(args,
              test_df,
              model,
              apply_clahe=False,
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
    counter = 0
    for i in range(len(test_df)):
        counter += 1
        # get image sample from the folder
        image_path = str(test_df.imagePath.iloc[i])

        # preprocess the image including resizing and rotation
        image, output_shape, spacing, origin, direction = get_test_dataset_ultrasound(image_path)

        # apply CLAHE as preprocessing, less impact
        if apply_clahe:
            seq = iaa.Sequential([iaa.CLAHE(1.0)])  # apply histogram equalization
            image = seq(images=image)

        # normalize the volume, min-max normalization
        image = np.array(image, np.float32) / 255.
        print("This is the input shape for the model:", image.shape)
        image = np.moveaxis(image, -1, 1)

        # predict the input slices using 2.5 Model
        total_pred = []
        for j in range(0, len(image), args.batch_size):
            chunk = image[j:j + args.batch_size, ...]
            prediction = model.forward(t.tensor(chunk).cuda())  # .cuda())
            total_pred.extend(prediction.cpu().detach().numpy())
        prediction = np.array(total_pred)

        # argmax prediction as it use softmax
        y_pred = np.argmax(np.moveaxis(prediction, 1, -1), axis=-1)
        # apply connected component analysis to remove small oversegmented regions.
        y_pred = keep_largest_connected_components(mask=y_pred)

        # whether apply any postprocessing step like median filter to smooth out the boundaries
        if postpro:
            y_pred = median_filter(y_pred, size=7)

        # here we resize the volume back to it's original shape based on different view.
        print("This is the model prediction shape:", y_pred.shape)
        if args.cohort == 'stanford':
            y_pred = resize_volume_back(y_pred, h=output_shape[0], w=output_shape[1])
            y_pred = np.moveaxis(y_pred, 1, 0)
            y_pred = np.moveaxis(y_pred, 1, 2)
        else:
            y_pred = resize_volume_back(y_pred, h=output_shape[1], w=output_shape[2])
        print("This is the prediction after resizing:", y_pred.shape)
        # Create ITK image to save into the disk
        out = sitk.GetImageFromArray(y_pred)
        out.SetSpacing(spacing)
        out.SetOrigin(origin)
        out.SetDirection(direction)

        # if to resample the image to original spacing
        # out = resample_image(out, spacing)
        # save the image into the disk
        sitk.WriteImage(out, "results/{}".format(os.path.basename(image_path.replace('.nii.gz', '_pred.nii.gz'))))


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
    parser.add_argument("-basedir", help="the path to dataset folder", type=str, default="input/Patdbase")
    parser.add_argument("-mt", "--model_type", help="which model should use for prediction", type=str,
                        default="coorddrunet2.5D")
    parser.add_argument("--n_samples", help="number of samples to train", type=int, default=-1)
    parser.add_argument("-bs", "--batch_size", help="batch size of training", type=int, default=128)
    parser.add_argument("-sp", "--image_spacing", help="image spacing", type=list, default=[0.25, 0.25, 0.25])
    parser.add_argument("-nc", "--n_class", help="number of classes to segment", type=int, default=2)
    parser.add_argument("-nf", "--n_filter", help="number of initial filters for DR-UNET", type=int, default=32)
    parser.add_argument("-nb", "--n_block", help="number unet blocks", type=int, default=3)
    parser.add_argument("-in", "--in_channel", help="number input channels", type=int, default=3)
    parser.add_argument("-us_view", "--ultrasound_view", help="the view of ultrasound image", type=str, default='axial')
    parser.add_argument("-co", "--cohort", help="the cohort name for test dataset, please for stanford data use "
                                                "'stanford' and for needle sequences use 'needle'", type=str,
                        default='stanford')
    parser.add_argument("-or", "--outputdir", help="the cohort name for test dataset", type=str,
                        default='NeedleCineLoops')
    args = parser.parse_args()
    if not os.path.join('results/{}/{}/'.format(args.cohort, args.outputdir)):
        os.makedirs(os.path.join('results/{}/{}/'.format(args.cohort, args.outputdir)))
    data = []
    for pat in glob.glob(os.path.join(args.basedir, "*.nii.gz")):
        base_name = os.path.split(pat)
        pat_id = base_name[-1][0:16]
        data.append([pat, pat_id])
    df = pd.DataFrame(data, columns=['imagePath', 'subject_ID'])

    print(df.head)
    # train_df, valid_df, test_df = data_split(df)
    test_df = df

    if args.model_type == 'coorddrunet2.5D':
        model = Segmentation_model(filters=args.n_filter,
                                   in_channels=args.in_channel,
                                   n_block=args.n_block,
                                   n_class=args.n_class,
                                   attention=True)
        model.load_state_dict(
            torch.load(
                'weights/prostateUS.unetcoord_100Per_lesion_LKD_lr_0.0001_32.gaussian_noise/unet_model_checkpoint.pt'))
        print("CoordDR-UNet2.5D model is loaded...!")
    elif args.model_type == 'attunet2.5D':
        model = AttU_Net(img_ch=args.in_channel, output_ch=args.n_class)
        model.load_state_dict(
            torch.load('weights/prostateUS.attunet_100Per_lr_0.0001_32.gaussian_noise/unet_model_checkpoint.pt'), )
        print("Att-UNet2.5 model is loaded...!")
    else:
        model = NestedUNet(in_ch=args.in_channel, out_ch=args.n_class)
        model.load_state_dict(
            torch.load('weights/prostateUS.nestedunet_100Per_lr_0.0001_32.gaussian_noise/unet_model_checkpoint.pt'))
        print("NestedUNet2.5D model is loaded...!")

    model.eval().cuda()
    start = datetime.now()
    t.autograd.set_detect_anomaly(True)
    inference(args, test_df, model)
    end = datetime.now()
    print("time elapsed for training (hh:mm:ss.ms) {}".format(end - start))
