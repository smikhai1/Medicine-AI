import cv2
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from youtrain.factory import DataFactory
from transforms import test_transform, mix_transform


class BaseDataset(Dataset):
    def __init__(self, image_dir, ids, transform):
        self.image_dir = image_dir
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        raise NotImplementedError

class TrainDataset(BaseDataset):
    def __init__(self, image_dir, mask_dir, ids, transform):
        super().__init__(image_dir, ids, transform)
        self.transform = transform
        self.ids = ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir


    def __getitem__(self, index):
        name = self.ids[index]
        image = cv2.imread(os.path.join(self.image_dir, name + '.bmp'), 0)
        mask = np.load(os.path.join(self.mask_dir, name + '.npz'))['arr_0']

        return self.transform({'image': image, 'mask': mask})

class TestDataset(BaseDataset):
    def __init__(self, image_dir, ids, transform):
        super().__init__(image_dir, ids, transform)
        self.transform = transform
        self.image_dir = image_dir
        self.ids = sorted(os.listdir(self.image_dir))

    def __getitem__(self, index):
        name = self.ids[index]
        image = cv2.imread(os.path.join(self.image_dir, name), 0)
        mask = np.copy(image) # kostyl for inference :)
        return self.transform({'image': image, 'mask': mask})


class TaskDataFactory(DataFactory):
    def __init__(self, params, paths, **kwargs):
        super().__init__(params, paths, **kwargs)
        self.fold = kwargs['fold']
        self._folds = None

    @property
    def data_path(self):
        return Path(self.paths['path'])

    def make_transform(self, stage, is_train=False):
        if is_train:
            if stage['augmentation'] == 'mix_transform':
                transform = mix_transform(**self.params['augmentation_params'])
            else:
                raise KeyError('augmentation does not found')
        else:
            transform = test_transform(**self.params['augmentation_params'])
        return transform

    def make_dataset(self, stage, is_train):
        transform = self.make_transform(stage, is_train)
        ids = self.train_ids if is_train else self.val_ids
        return TrainDataset3D(
            image_dir=self.data_path / self.paths['train_images'],
            mask_dir=self.data_path / self.paths['train_masks'],
            ids=ids,
            transform=transform)

    def make_loader(self, stage, is_train=False):
        dataset = self.make_dataset(stage, is_train)
        return DataLoader(
            dataset=dataset,
            batch_size=self.params['batch_size'],
            shuffle=is_train,
            drop_last=is_train,
            num_workers=self.params['num_workers'],
            pin_memory=torch.cuda.is_available(),
        )
    @property
    def folds(self):
        if self._folds is None:
            self._folds = pd.read_csv(self.data_path / self.paths['folds'], sep='\t', dtype={'ImageId':object})
        return self._folds

    @property
    def train_ids(self):
        return self.folds.loc[self.folds['fold'] != self.fold, 'ImageId'].values

    @property
    def val_ids(self):
        return self.folds.loc[self.folds['fold'] == self.fold, 'ImageId'].values

class TrainDataset3D(BaseDataset):

    def __init__(self, image_dir, mask_dir, ids, transform):
        super().__init__(image_dir, ids, transform)
        self.transform = transform
        self.ids = ids
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.NUM_SLICES = 96

    def get_slices_idxs(self, num_slices):
        return np.linspace(0, num_slices-1, self.NUM_SLICES).astype('int')

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.image_dir, self.ids[idx] + 'npz'))['arr_0']
        mask = np.load(os.path.join(self.mask_dir, self.ids[idx] + 'npz'))['arr_0']

        num_slices = image.shape[2]

        if num_slices <= self.NUM_SLICES:
            image = image[:, :, self.get_slices_idxs(num_slices)]
            mask = mask[:, :, self.get_slices_idxs(num_slices)]
        else:
            low_slice_id = int(0.2*num_slices)
            high_slice_id = int(0.8*num_slices)

            if high_slice_id - low_slice_id >= self.NUM_SLICES:
                start_slice = np.random.randint(low_slice_id, high_slice_id - self.NUM_SLICES)
            elif num_slices - low_slice_id >= self.NUM_SLICES:
                start_slice = np.random.randint(low_slice_id, num_slices - self.NUM_SLICES)
            else:
                start_slice = np.random.randint(0, num_slices - self.NUM_SLICES)

            image = image[:, :, start_slice:start_slice+self.NUM_SLICES]
            mask = mask[:, :, start_slice:start_slice+self.NUM_SLICES]
        print(image.shape[2])
        assert (image.shape[2] != self.NUM_SLICES) or (mask.shape[2] != self.NUM_SLICES), \
               'problems with z-dimension'

        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.ids)