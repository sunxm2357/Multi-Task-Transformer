# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import os
import json
import sys
import tarfile
import cv2

from PIL import Image
import numpy as np
import torch.utils.data as data
import scipy.io as sio
from six.moves import urllib

class CityScapes_MT(data.Dataset):
    """
    from MTI-Net, changed for using ATRC data
    Cityscapes dataset for multi-task learning.
    Includes semantic segmentation and depth prediction.

    Data can also be found at:
    https://drive.google.com/file/d/14EAEMXmd3zs2hIMY63UhHPSFPDAkiTzw/view?usp=sharing

    """


    def __init__(self,
                 root=None,
                 download=False,
                 split='val',
                 transform=None,
                 retname=True,
                 overfit=False,
                 do_semseg=False,
                 do_depth=False,
                 task_file=None
                 ):

        self.root = root

        if download:
            raise NotImplementedError

        self.crop_h = 224
        self.crop_w = 224
        self.transform = transform

        json_file = os.path.join(root, 'cityscape.json')
        with open(json_file, 'r') as f:
            info = json.load(f)
        self.root = root
        self.groups = info[split]

        self.split = split
        self.retname = retname

        # Original Images
        self.im_ids = []
        self.images = []
        _image_dir = root

        # Semantic segmentation
        self.do_semseg = do_semseg
        self.semsegs = []
        _semseg_gt_dir = root

        # Depth
        self.do_depth = do_depth
        self.depths = []
        _depth_gt_dir = root

        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(root, 'lists')

        print('Initializing dataloader for Cityscapes {} set'.format(self.split))
        if task_file is not None:
            with open(os.path.join(os.path.join(_splits_dir, task_file)), 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                idx = int(line.strip())
                self.im_ids.append(idx)
                _image = os.path.join(_image_dir, 'train/image/%d.npy' % idx)
                assert os.path.isfile(_image)
                self.images.append(_image)

                _semseg = os.path.join(self.root, _semseg_gt_dir, 'train/label_19/%d.npy' % idx)
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                _depth = os.path.join(self.root, _depth_gt_dir, 'train/depth/%d.npy'%idx)
                assert os.path.isfile(_depth)
                self.depths.append(_depth)
        else:
            for group in self.groups:
                self.im_ids.append(group[0].split('/')[-1].split('.')[0])

                _image = os.path.join(_image_dir, group[0])
                assert os.path.isfile(_image)
                self.images.append(_image)

                _semseg = os.path.join(self.root, _semseg_gt_dir, group[-1])
                assert os.path.isfile(_semseg)
                self.semsegs.append(_semseg)

                _depth = os.path.join(self.root, _depth_gt_dir, group[1])
                assert os.path.isfile(_depth)
                self.depths.append(_depth)

        if self.do_semseg:
            assert (len(self.images) == len(self.semsegs))
        if self.do_depth:
            assert (len(self.images) == len(self.depths))

        # Uncomment to overfit to one image
        if overfit:
            n_of = 64
            self.images = self.images[:n_of]
            self.im_ids = self.im_ids[:n_of]

        print(self.transform)
        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

    def __getitem__(self, index):
        sample = {}

        _img = self._load_img(index)
        sample['image'] = _img.astype(np.float32)

        if self.do_semseg:
            _semseg = self._load_semseg(index)
            if _semseg.shape[:2] != _img.shape[:2]:
                print('RESHAPE SEMSEG')
                _semseg = cv2.resize(_semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['semseg'] = _semseg.astype(np.float32)


        if self.do_depth:
            _depth = self._load_depth(index)
            if _depth.shape[:2] != _img.shape[:2]:
                print('RESHAPE DEPTH')
                _depth = cv2.resize(_depth, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            sample['depth'] = _depth.astype(np.float32)

        if self.retname:
            sample['meta'] = {'img_name': str(self.im_ids[index]),
                              'img_size': (_img.shape[0], _img.shape[1])}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.load(self.images[index])[:, :, ::-1] * 255
        return _img

    def _load_semseg(self, index):
        # Note: We ignore the background class (40-way classification), as in related work:
        _semseg = np.expand_dims(np.load(self.semsegs[index]), axis=-1)
        return _semseg

    def _load_depth(self, index):
        _depth = np.load(self.depths[index])
        _depth = _depth.astype(np.float32)
        if _depth.ndim == 2:
            _depth = np.expand_dims(_depth, axis=2)
        return _depth

    def __str__(self):
        return 'CityScapes Multitask (split=' + str(self.split) + ')'

