import os
import numpy as np
import random
import torch
import cv2
from torch.utils import data
from utils.transforms import get_affine_transform
from utils.ImgTransforms import AugmentationBlock, autoaug_imagenet_policies
from dataset.target_generation import generate_edge, generate_hw_gt

class LIPDataSet(data.Dataset):
    def __init__(self, root, dataset, crop_size=[473, 473], scale_factor=0.25,
                 rotation_factor=30, ignore_label=255, transform=None):
        """
        Dataset for LIP parsing task with training augmentation and preprocessing.
        """
        self.root = root
        self.aspect_ratio = crop_size[1] * 1.0 / crop_size[0]
        self.crop_size = np.asarray(crop_size)
        self.ignore_label = ignore_label
        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip_prob = 0.5
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [11, 14], [12, 13], [10, 15]]
        self.transform = transform
        self.dataset = dataset

        subfolder = {
            'train': 'Training',
            'val': 'Validation',
            'test': 'Testing'
        }[dataset]

        list_path = os.path.join(self.root, subfolder, f'{dataset}_id.txt')
        self.im_list = [i_id.strip() for i_id in open(list_path)]
        self.number_samples = len(self.im_list)

        # Augmentation block
        self.augBlock = AugmentationBlock(autoaug_imagenet_policies)

    def __len__(self):
        return self.number_samples

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        im_name = self.im_list[index]

        subfolder = {
            'train': 'Training',
            'val': 'Validation',
            'test': 'Testing'
        }[self.dataset]

        im_path = os.path.join(self.root, subfolder, 'Images', im_name + '.jpg')
        parsing_anno_path = os.path.join(self.root, subfolder, 'Segmentations', im_name + '.png')

        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape
        parsing_anno = np.zeros((h, w), dtype=np.long)

        # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0

        if self.dataset != 'test':
            parsing_anno = cv2.imread(parsing_anno_path, cv2.IMREAD_GRAYSCALE)

            if self.dataset == 'train':
                sf = self.scale_factor
                rf = self.rotation_factor
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
                r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                    if random.random() <= 0.6 else 0

                if random.random() <= self.flip_prob:
                    im = im[:, ::-1, :]
                    parsing_anno = parsing_anno[:, ::-1]
                    center[0] = im.shape[1] - center[0] - 1

        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        meta = {
            'name': im_name,
            'center': center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        if self.dataset != 'train':
            return input, meta
        else:
            label_parsing = cv2.warpAffine(
                parsing_anno,
                trans,
                (int(self.crop_size[1]), int(self.crop_size[0])),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255))
            hgt, wgt, hwgt = generate_hw_gt(label_parsing)

            label_parsing = torch.from_numpy(label_parsing)
            return input, label_parsing, hgt, wgt, hwgt, meta


# For validation set
class LIPDataValSet(data.Dataset):
    def __init__(self, root, dataset='val', crop_size=[512, 512], transform=None, flip=False):
        self.root = root
        self.crop_size = crop_size
        self.transform = transform
        self.flip = flip
        self.dataset = dataset

        subfolder = 'Validation'
        list_path = os.path.join(self.root, subfolder, f'{dataset}_id.txt')
        self.val_list = [i_id.strip() for i_id in open(list_path)]
        self.number_samples = len(self.val_list)

    def __len__(self):
        return len(self.val_list)

    def _box2cs(self, box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def _xywh2cs(self, x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)
        return center, scale

    def __getitem__(self, index):
        val_item = self.val_list[index]

        subfolder = 'Validation'
        im_path = os.path.join(self.root, subfolder, 'Images', val_item + '.jpg')
        im = cv2.imread(im_path, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        h, w, _ = im.shape

        person_center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(person_center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))

        if self.transform:
            input = self.transform(input)

        flip_input = input.flip(dims=[-1]) if self.flip else None
        batch_input_im = torch.stack([input, flip_input]) if self.flip else input

        meta = {
            'name': val_item,
            'center': person_center,
            'height': h,
            'width': w,
            'scale': s,
            'rotation': r
        }

        return batch_input_im, meta
