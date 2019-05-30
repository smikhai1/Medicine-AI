import numpy as np
import pandas as pd
import os
import nibabel as nib
import cv2
from glob import glob
from argparse import ArgumentParser
from tqdm import tqdm

def convert_range(image, max_value, min_value):

    image = np.clip(np.round((image - min_value) / (max_value - min_value) * 255), 0, 255).astype(np.uint8)

    return image

def get_correct_idx(idx, max_digits):
    idx_string = '0'*max_digits
    i = len(str(idx))
    return idx_string[:-i] + str(idx)

def transform_data(source, destination, format='bmp'):
    """
    This function converts original dataset, which is a collection of
    3D images and 3D masks in .nii format to 2D images and masks in .bmp format.
    Each image/mask corresponds to  a slice of 3D image

    :param source: path to the original dataset root directory, str
    :param destination: path to the root directory of transformed dataset, str
    :param format: format of output images, str
    :return: None
    """

    img_paths = sorted(glob(os.path.join(source, 'imagesTr/*.gz')))
    mask_paths = sorted(glob(os.path.join(source, 'labelsTr/*.gz')))

    if not os.path.exists(os.path.join(destination, 'images')):
        os.makedirs(os.path.join(destination, 'images'))

    if not os.path.exists(os.path.join(destination, 'masks')):
        os.makedirs(os.path.join(destination, 'masks'))

    for idx in range(len(img_paths)):
        img = nib.load(img_paths[idx]).get_fdata()
        mask = nib.load(mask_paths[idx]).get_fdata().astype(np.uint8)

        num_slices = img.shape[2]
        for slice_n in range(num_slices):
            img_slice = img[:, :, slice_n]
            img_slice = convert_range(img_slice, np.max(img_slice), np.min(img_slice))
            mask_slice = mask[:, :, slice_n]

            # save image and mask
            idx_str = get_correct_idx(idx, len(str(len(img_paths))))
            slice_n_str = get_correct_idx(slice_n, len(str(num_slices)))
            cv2.imwrite(os.path.join(destination, 'images', f'{idx_str}_{slice_n_str}.{format}'), img_slice)
            np.savez(os.path.join(destination, 'masks', f'{idx_str}_{slice_n_str}'), mask_slice)

def transform_3d_data(source, destination, val_size=0.1):
    """
    This function converts original dataset, which is a collection of
    3D images and 3D masks in .nii format to 3D tensors and masks in .npz format.

    :param source: path to the original dataset root directory, str
    :param destination: path to the root directory of transformed dataset, str
    :param format: format of output images, str
    :return: None
    """

    img_paths = sorted(glob(os.path.join(source, 'imagesTr/*.gz')))
    mask_paths = sorted(glob(os.path.join(source, 'labelsTr/*.gz')))
    m = len(img_paths)

    if not os.path.exists(os.path.join(destination, 'images')):
        os.makedirs(os.path.join(destination, 'images'))

    if not os.path.exists(os.path.join(destination, 'masks')):
        os.makedirs(os.path.join(destination, 'masks'))

    for idx in tqdm(range(len(img_paths))):
        img = nib.load(img_paths[idx]).get_fdata()
        mask = nib.load(mask_paths[idx]).get_fdata().astype(np.uint8)
        idx_correct = get_correct_idx(idx, 3)
        np.savez(os.path.join(destination, 'images', f'{idx_correct}'), img)
        np.savez(os.path.join(destination, 'masks', f'{idx_correct}'), mask)

    names = sorted(os.listdir(destination + '/images'))
    names = list(map(lambda x: x.split('.')[0], names))
    folds = np.zeros(m)
    val_idxs = np.random.choice(m, replace=False, size=int(0.1 * m))
    folds[val_idxs] = 1
    df = pd.DataFrame(np.hstack((np.array(names), folds)), columns=['ImageId', 'fold'])
    df['fold'] = pd.to_numeric(df['fold'])
    df.to_csv(os.path.join(destination, 'folds.csv'), sep='\t')



def split_data(path, split_ratio=0.15):
    """
    Makes .csv file with columns 'ImageId'/'ImageName', 'Fold', where
    the last column contains zeros and ones. For rows with ones in 'Fold'
    column the corresponding image will be put in holdout set for validation

    :param path: path to directory with images, str
    :param split_ratio: ratio between train and validation set, float
    :return: None
    """
    img_names = sorted(os.listdir(path + '/images'))
    img_names = list(map(lambda x: x.split('.')[0], img_names))
    img_names = np.array(img_names).reshape(-1, 1)
    m = len(img_names)
    folds = np.zeros((m, 1), dtype=np.uint8)
    val_idxs = np.random.choice(m, size=int(split_ratio * m), replace=False)
    folds[val_idxs] = 1

    data = np.hstack((img_names, folds))
    df = pd.DataFrame(data, columns=['ImageId', 'fold'])
    df['fold'] = pd.to_numeric(df['fold'])
    df.to_csv(os.path.join(path, 'folds.csv'), sep='\t')
    return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('source', help='path to the original dataset root directory')
    parser.add_argument('dest', help='path to the root directory of transformed dataset')
    args = parser.parse_args()

    transform_data(args.source, args.dest, format='bmp')
    print('Dataset transformed successfully!', end='\n\n')

    split_data(args.dest, split_ratio=0.15)
    print('folds.csv created successfully!', end='\n\n')