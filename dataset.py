import os
from collections import OrderedDict

import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import glob
import cv2
from PIL import Image

COLOR_MAP = OrderedDict(
    Background=(0, 0, 0),
    Building=(255, 255, 255),

)


LABEL_MAP = OrderedDict(
    Background=0,
    Building=255,
)



def reclassify(cls):
    cls_mtx = np.array(cls)
    new_cls = np.zeros((cls_mtx.shape[0],cls_mtx.shape[1]))
    for idx, label in enumerate(COLOR_MAP.values()):
        new_cls = np.where(cls == label, np.ones_like(new_cls)*idx, new_cls)
    return new_cls
class DataSet(data.Dataset):
    def __init__(self, data_root, transforms=None):
        super(DataSet, self).__init__()
        root = os.path.join(data_root)
        assert os.path.exists(root), "path '{}' does not exist.".format(root)
        image_dir = os.path.join(root, 'images')
        mask_dir = os.path.join(root, 'labels')

        # txt_path = os.path.join(root, "ImageSets", "Segmentation", txt_name)
        # assert os.path.exists(txt_path), "file '{}' does not exist.".format(txt_path)
        # with open(os.path.join(txt_path), "r") as f:
        #     file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]

        self.images = sorted(glob.glob(os.path.join(image_dir , "*.tif")))
        self.masks = sorted(glob.glob(os.path.join(mask_dir , "*.png")))
        assert (len(self.images) == len(self.masks))
        self.transforms = transforms

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])


        if self.transforms is not None:
            img = self.transforms(img)
            target = self.transforms(target)
            # target=torch.LongTensor(target)
        return img, target
    def __len__(self):
        return len(self.images)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs
