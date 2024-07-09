import torch.utils.data as data
import os
from glob import glob
import torch
from torchvision import transforms
import random
import numpy as np
import cv2


def random_crop(RGB, T, den, size, num_patch=4):
    half_h = size
    half_w = size

    result_RGB = np.zeros([num_patch, RGB.shape[0], half_h, half_w])
    result_T = np.zeros([num_patch, T.shape[0], half_h, half_w])
    result_den = []
    # crop num_patch for each image
    for i in range(num_patch):
        start_h = random.randint(0, RGB.shape[1] - half_h)
        start_w = random.randint(0, RGB.shape[2] - half_w)
        end_h = start_h + half_h
        end_w = start_w + half_w
        # copy the cropped rect
        result_RGB[i] = RGB[:, start_h:end_h, start_w:end_w]
        result_T[i] = T[:, start_h:end_h, start_w:end_w]
        # copy the cropped points
        idx = (den[:, 0] >= start_w) & (den[:, 0] <= end_w) & (den[:, 1] >= start_h) & (den[:, 1] <= end_h)
        # shift the corrdinates
        record_den = den[idx]
        record_den[:, 0] -= start_w
        record_den[:, 1] -= start_h
        result_den.append(record_den)

    result_RGB = torch.Tensor(result_RGB)
    result_T = torch.Tensor(result_T)

    return result_RGB, result_T, result_den


class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size=256,
                 downsample_ratio=8,
                 method='train'):

        self.root_path = root_path
        self.gt_list = sorted(glob(os.path.join(self.root_path, '*.npy')))  # change to npy for gt_list
        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")
        self.method = method

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        self.RGB_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.407, 0.389, 0.396],
                std=[0.241, 0.246, 0.242]),
        ])
        self.T_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.492, 0.168, 0.430],
                std=[0.317, 0.174, 0.191]),
        ])

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, item):
        gt_path = self.gt_list[item]
        rgb_path = gt_path.replace('GT', 'RGB').replace('npy', 'jpg')
        t_path = gt_path.replace('GT', 'T').replace('npy', 'jpg')

        RGB = cv2.imread(rgb_path)[..., ::-1].copy()
        T = cv2.imread(t_path)[..., ::-1].copy()

        RGB = self.RGB_transform(RGB)
        T = self.T_transform(T)
        if self.method == 'train':
            keypoints = np.load(gt_path)
            return self.train_transform(RGB, T, keypoints)

        elif self.method == 'val' or self.method == 'test':  # TODO
            keypoints = np.load(gt_path)

            point = [keypoints]
            target = [{} for i in range(len(point))]
            for i, _ in enumerate(point):
                target[i]['point'] = torch.Tensor(point[i])
                image_id = 2
                image_id = torch.Tensor([image_id]).long()
                target[i]['image_id'] = image_id
                target[i]['labels'] = torch.ones([point[i].shape[0]]).long()

            input = [RGB, T]
            return input, target

        else:
            raise Exception("Not implement")

    def train_transform(self, RGB, T, keypoints):
        c, ht, wd = RGB.shape
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) > 0
        result_RGB, result_T, result_den = random_crop(RGB, T, keypoints, self.c_size, 4)

        input = [result_RGB, result_T]

        target = [{} for i in range(len(result_den))]
        for i, _ in enumerate(result_den):
            target[i]['point'] = torch.Tensor(result_den[i])
            image_id = 1
            image_id = torch.Tensor([image_id]).long()
            target[i]['image_id'] = image_id
            target[i]['labels'] = torch.ones([result_den[i].shape[0]]).long()
        return input, target

