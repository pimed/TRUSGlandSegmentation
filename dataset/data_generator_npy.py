"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020
"""

import cv2
import glob
import random
import torch
import os.path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import imgaug as ia
import imgaug.augmenters as iaa
from matplotlib import pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from utils.resampling import resample_image
import copy 


def get_seed(seed=3):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_volume(img_volume,
                  w=288,
                  h=288,
                  is_img=True):
    """
    :param img_volume:
    :return:
    """
    img_res = []
    z = img_volume.shape[2]

    for i in range(z):
        if is_img and i == 0:
            slice_img = img_volume[:, :, 0:i + 3]
        elif is_img and i == z - 1:
            slice_img = img_volume[:, :, i - 2:i + 1]
        elif is_img:
            slice_img = img_volume[:, :, i - 1:i + 2]
        else:
            slice_img = img_volume[:, :, i]
        img_res.append(cv2.resize(slice_img, dsize=(h, w), interpolation=cv2.INTER_NEAREST))

    return np.array(img_res)


def data_split(df,
               test_size=0.20,
               valid_size=0.10,
               savecsv=False):

    """
    A function to split the data into train/test/validation based on patient ids, using GroupShuffleSplit.
        df: the data frame hold the image paths
        test_size: the test size
        valid_size: the size of valid dataset
        savecesv: if to save the dataframes as csv files.
    """
    gs = GroupShuffleSplit(n_splits=2, test_size=test_size, random_state=42)
    train_ix, test_ix = next(gs.split(df, groups=df.subject_ID))

    train_df = df.loc[train_ix]
    test_df = df.loc[test_ix]
    df = train_df

    df.reset_index(drop=True, inplace=True)
    gs = GroupShuffleSplit(n_splits=2, test_size=valid_size, random_state=42)
    train_ix, valid_ix = next(gs.split(df, groups=df.subject_ID))

    train_df = df.loc[train_ix]
    valid_df = df.loc[valid_ix]

    print("Number of patients in training set:", len(np.unique(train_df.subject_ID.values)))
    print("Number of patients in validation set:", len(np.unique(valid_df.subject_ID.values)))
    print("Number of patients in testing set:", len(np.unique(test_df.subject_ID.values)))
    print("\n")
    print("Number of studies in training set:", len(train_df))
    print("Number of studies in validation set:", len(valid_df))
    print("Number of studies in testing set:", len(test_df))

    if savecsv:
        train_df.to_csv("data_split/train.csv", header=True, index=False)
        test_df.to_csv("data_split/test.csv", header=True, index=False)
        valid_df.to_csv("data_split/valid.csv", header=True, index=False)

    return train_df, valid_df, test_df


def save_slices(x,
                y,
                filename,
                x_path):
    for slice_id in range(len(x)):
        new_filename = os.path.split(filename)[-1]
        # new_filename = new_filename[0:27]
        new_filename = new_filename.replace(".nii.gz", "")
        new_filename = new_filename + "_{}.npz".format(slice_id + 1)
        np.savez_compressed(os.path.join(x_path, new_filename), a=x[slice_id, ...], b=y[slice_id, ...])

def change_orientation(orig_img):
    """
    A function that changes the orientation of the UCL images 
    to match Stanford data set 

    Notes: 
    ----------
    Transformation involves the following:
        1. Transpose and change axis 0 and 2 around
        2. Flip vertically 
        3. Rotate 180 degrees in axes (2,1) to match orientation of Stanford 

    Parameters:
    -----------
    :orig_img: Original image in format (z,x,y) 

    Output:
    ----------
    :transformed_img: Transformed image in format (y,x,z) 

    """
    transposed_img = np.transpose(copy.deepcopy(orig_img), [2, 1, 0])
    flipped_img = np.flipud(copy.deepcopy(transposed_img))
    transformed_img = np.rot90(copy.deepcopy(flipped_img), k = 2, axes = (2,1))
    
    return transformed_img 

def get_dataset_ultrasound(npy_image_dir, df,
                           h=128,
                           w=160,
                           us_view='axial',
                           cohort='UCIL',
                           binary_seg=False,
                           save_npy=False,
                           resample = False):
    """
    Match DCM volumes with corresponding annotation files
    :param vol:
    :return:
    """

    for i in range(len(df)):
        d_nrrd = str(df.imagePath.iloc[i])
        if cohort == 'UCIL':
            d_nrrd = d_nrrd.replace('images', 'masks').replace('image', 'label')
        else:
            d_nrrd = d_nrrd.replace('images', 'masks').replace('.nii.gz', '.nrrd')
        try:
            seg_image = sitk.ReadImage(d_nrrd)
        except:
            print("The nrrd file is not exist.!")
            continue

        d_nii = str(df.imagePath.iloc[i])
        us_image = sitk.ReadImage(d_nii)
        original_image_spacing = us_image.GetSpacing()
        if resample:
            us_image = resample_image(us_image, [0.25, 0.25, 0.25])
            seg_image = resample_image(seg_image, [0.25, 0.25, 0.25])

        voxels = sitk.GetArrayFromImage(us_image)
        seg = sitk.GetArrayFromImage(seg_image)

        voxels = change_orientation(voxels)
        seg = change_orientation(seg)

        #if seg.shape[1] != seg.shape[2] and voxels.shape[1] != voxels.shape[2]:
        #    print("corrupted image, move the next one!")
        #    continue

        # Binarize all the segmentations
        seg = np.where(seg > 1, 1, seg)

        # Resize images
        voxels = resize_volume(voxels, h, w,  is_img=True)
        seg = resize_volume(seg, h, w, is_img=False)

        if save_npy:
            save_slices(voxels, seg, d_nii, npy_image_dir)
        print("Image", i, voxels.shape, seg.shape, np.unique(seg), us_image.GetSpacing(), original_image_spacing)


def volumes_to_slices(input_dir, ultrasound_view='axial'):

    data = []
    for pat in glob.glob(os.path.join(input_dir, "*.nii.gz")):
        base_name = os.path.split(pat)
        pat_id = base_name[-1][0:27]
        data.append([pat, pat_id])
    df = pd.DataFrame(data, columns=['imagePath', 'subject_ID'])

    print(df.head())
    train_df, valid_df, test_df = data_split(df)

    train_npy_dir = os.path.join(input_dir, "npy_train")
    valid_npy_dir = os.path.join(input_dir, "npy_valid")
    if not os.path.exists(train_npy_dir):
        os.makedirs(train_npy_dir)
    if not os.path.exists(valid_npy_dir):
        os.makedirs(valid_npy_dir)

    get_dataset_ultrasound(train_npy_dir,
                           train_df,
                           us_view=ultrasound_view,
                           cohort='UCIL',
                           save_npy=True,
                           binary_seg=True)
    get_dataset_ultrasound(valid_npy_dir,
                           valid_df,
                           us_view=ultrasound_view,
                           cohort='UCIL',
                           save_npy=True,
                           binary_seg=True)


class ImageProcessor:

    @staticmethod
    def augmentation_heavy(image, mask):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        seq = iaa.Sequential(
            [
                iaa.CLAHE(1.0),
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    # translate by -20 to +20 percent (per axis)
                    # rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL  # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                iaa.SomeOf((0, 5),
                           [
                               sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                               # convert images into their superpixel representation
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                               iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),  # emboss images
                               iaa.SimplexNoiseAlpha(iaa.OneOf([
                                   iaa.EdgeDetect(alpha=(0.5, 1.0)),
                                   iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                               ])),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.OneOf([
                                   iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                                   iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                               ]),
                               iaa.Invert(0.05, per_channel=True),  # invert color channels
                               iaa.Add((-10, 10), per_channel=0.5),
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.OneOf([
                                   iaa.Multiply((0.5, 1.5), per_channel=0.5),
                               ]),
                               iaa.Grayscale(alpha=(0.0, 1.0)),
                               sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)),
                               sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                               sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        image_heavy, mask_heavy = seq(images=image, segmentation_maps=mask)
        return image_heavy, mask_heavy

    @staticmethod
    def light_aug(images, masks=None, segmap=False):

        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5),
                # iaa.CropToFixedSize(128, 160),
                iaa.Flipud(0.5),
                iaa.CLAHE(1.0),
                iaa.GaussianBlur(0.5),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={"x": (-0.1, 0.05), "y": (-0.1, 0.1)},
                    rotate=(-10, 10),
                    shear=(-12, 12),
                    order=[0, 1],
                    cval=(0, 255),
                    mode='constant',
                )),
            ],
            random_order=True
        )
        if masks is None:
            image_light = seq(images=images)
            return image_light
        else:
            if segmap:
                segmaps = []
                for mask in masks:
                    segmaps.append(SegmentationMapsOnImage(mask.astype(np.int32), shape=images.shape[-3:]))
            else:
                segmaps = np.array(masks, dtype=np.int32)
            image_light, masks = seq(images=images, segmentation_maps=segmaps)
            if segmap:
                mask_light = []
                for mask in masks:
                    mask_light.append(mask.get_arr())
                masks = np.array(mask_light)
            return image_light, masks

    @staticmethod
    def split_data(img_path):
        """
        Load train csv file and split the data into train and validation!
        :return:
        """
        df_train = pd.DataFrame(columns=['img'])
        df_train['img'] = np.arange(img_path)
        ids_train = df_train['img']
        return ids_train

    @staticmethod
    def crop_volume(vol, crop_size=112):

        """
        :param vol:
        :return:
        """

        return np.array(vol[:,
                        int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                        int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


class DataGenerator:
    def __init__(self, df,
                 x,
                 y,
                 channel="channel_first",
                 apply_noise=False,
                 apply_transform=False,
                 phase="train",
                 apply_online_aug=True,
                 batch_size=16,
                 height=256,
                 width=256,
                 crop_size=0,
                 n_samples=-1,
                 offline_aug=False,
                 toprint=False):
        assert phase == "train" or phase == "valid", r"phase has to be either'train' or 'valid'"
        assert isinstance(apply_noise, bool), "apply_noise has to be bool"
        assert isinstance(apply_online_aug, bool), "apply_online_aug has to be bool"
        self._data = df
        self._len = len(df)
        self._shuffle_indices = np.arange(len(df))
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        self._height, self._width = height, width
        self._apply_aug = apply_online_aug
        self._apply_noise = apply_noise
        self._apply_tranform = apply_transform
        self._crop_size = crop_size
        self._phase = phase
        self._channel = channel
        self._batch_size = batch_size
        self._index = 0
        self._totalcount = 0
        self.xtrain = x
        self.ytrain = y

        if n_samples == -1:
            self._n_samples = len(df)
        else:
            self._n_samples = n_samples
        self._offline_aug = offline_aug
        self._toprint = toprint

    def __len__(self):
        return self._len

    @property
    def apply_aug(self):
        return self._apply_aug

    @apply_aug.setter
    def apply_aug(self, aug):
        assert isinstance(aug, bool), "apply_aug has to be bool"
        self._apply_aug = aug

    def one_hot_encod(self, y, num_classes):
        """ 1-hot encodes a tensor """
        return np.eye(num_classes, dtype='uint8')[y]

    def get_image_paths(self, id):
        if self._phase == "train":
            img_path = 'input/images/npy_train/{}'.format(id[1])
        else:
            img_path = 'input/images/npy_valid/{}'.format(id[1])

        return img_path

    def convert(self, img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()

        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img

    def get_images_masks(self, img_path, mask_path):
        img = self.xtrain[img_path, :, :]
        mask = self.ytrain[mask_path, :, :]
        return img, mask

    def get_images_masks_disk(self, img_path, mask_path):
        npy = np.load(img_path, allow_pickle=True)
        img = npy['a']
        mask = npy['b']
        return img, mask

    def __iter__(self):
        # self._index = 0
        self._totalcount = 0
        return self

    def __next__(self):
        x_batch = []
        y_batch = []
        indices = []

        if self._totalcount >= self._n_samples:
            # self._index = 0
            self._totalcount = 0
            self._shuffle_indices = np.random.permutation(self._shuffle_indices)
            raise StopIteration
        for i in range(self._batch_size):
            indices.append(self._index)
            self._index += 1
            self._totalcount += 1
            self._index = self._index % self._len
            if self._totalcount >= self._n_samples:
                break

        # shuffle batch
        ids_train_batch = self._data.iloc[self._shuffle_indices[indices]]

        # loop over the batch values
        for _id in ids_train_batch.values:
            if self.xtrain is None or self.ytrain is None:
                img_path = self.get_image_paths(id=_id)
                img, mask = self.get_images_masks_disk(img_path=img_path, mask_path=img_path)
            else:
                img, mask = self.get_images_masks(img_path=_id, mask_path=_id)
            mask = np.expand_dims(mask, axis=-1)
            assert mask.ndim == 3

            x_batch.append(img)
            y_batch.append(mask)

        # apply data augmentation for training data.
        if self._apply_aug:
            x_batch, y_batch = ImageProcessor.light_aug(np.array(x_batch), np.array(y_batch))
        # for validation/test data only apply CLAHE.
        else:
            seq = iaa.Sequential(
                [iaa.CLAHE(1.0)])
            x_batch, y_batch = seq(images=x_batch, segmentation_maps=y_batch)

        # min-max batch normalisation
        x_batch = np.array(x_batch, np.float32) / 255.
        if self._channel == "channel_first":
            x_batch = np.moveaxis(x_batch, -1, 1)

        # Convert the segmentation masks into one hot encoding
        # num of classes are two since we have softmax activation
        y_batch = self.one_hot_encod(np.array(y_batch)[..., 0], num_classes=2)
        y_batch = np.moveaxis(y_batch, source=3, destination=1)
        return x_batch, y_batch


if __name__ == "__main__":

    base_dir = "../input/images/"
    volumes_to_slices(base_dir)
    data = []
    for pat in glob.glob("../input/images/npy_train/*.npz"):
        base_name = os.path.split(pat)[-1]
        # pat_id = base_name[-1][0:27]
        data.append([pat, base_name])
    train_df = pd.DataFrame(data, columns=['imagePath', 'subject_ID'])
    data = []
    for pat in glob.glob("../input/images/npy_valid/*.npz"):
        base_name = os.path.split(pat)[-1]
        # pat_id = base_name[-1][0:27]
        data.append([pat, base_name])
    valid_df = pd.DataFrame(data, columns=['imagePath', 'subject_ID'])
    ultrasound_view = 'axial'
    print("----")
    print("The number of train images:", len(train_df))
    print("The number of valid images:", len(valid_df))

    train_df = train_df.head(100)
    valid_df = valid_df.head(100)
    ids_train = ImageProcessor.split_data(len(train_df))
    ids_valid = ImageProcessor.split_data(len(valid_df))

    print("The number of train slices:", len(ids_train))
    print("The number of valid slices:", len(ids_valid))

    #
    bs = 100
    num_samples = 1000

    trainA_generator = DataGenerator(df=train_df,
                                     x=None,
                                     y=None,
                                     channel="channel_first",
                                     apply_noise=False,
                                     phase="train",
                                     apply_online_aug=True,
                                     batch_size=bs,
                                     height=128,
                                     width=160,
                                     n_samples=-1)
    img, mask = trainA_generator.__next__()
    print(np.mean(img), np.std(img), img.shape, mask.shape)

    for i in range(50, 60):
        f = plt.figure(figsize=(12, 18))
        f.add_subplot(1, 2, 1)
        plt.imshow(img[i, 0, :, :], cmap='gray')
        plt.title('Prostate Ultrasound Image')
        f.add_subplot(1, 2, 2)
        #plt.imshow(img[i, 0, :, :], cmap='gray'),
        plt.imshow(mask[i, 0, :, :], cmap='jet')
        plt.title('Prostate Ultrasound Mask')
        plt.show(block=True)
