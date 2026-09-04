import cv2
import numpy as np
from torch.utils import data as data

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.utils.matlab_functions import imresize
from torchvision.transforms.functional import normalize


@DATASET_REGISTRY.register()
class LRGenDataset(data.Dataset):
    """Paired dataset that generates LR images on-the-fly from a GT folder.

    It performs random cropping for training and center cropping for validation.
    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other settings.
            scale (int): Scale factor.
            gt_size (int): Cropped patch size for gt patches.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(LRGenDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder = opt['dataroot_gt']
        if self.io_backend_opt['type'] == 'lmdb':
            self.paths = paths_from_lmdb(self.gt_folder)
        else:
            self.paths = sorted(list(scandir(self.gt_folder, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt image
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        # generate lq on-the-fly
        img_lq = imresize(img_gt, 1 / scale)

        # augmentation
        if self.opt.get('phase') == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
        else:
            # validation: center crop
            gt_size = self.opt['gt_size']
            img_gt = self.center_crop(img_gt, gt_size)
            lq_size = gt_size // scale
            img_lq = self.center_crop(img_lq, lq_size)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'gt_path': gt_path}

    def center_crop(self, img, target_size):
        h, w, _ = img.shape
        if h < target_size or w < target_size:
            img = cv2.resize(img, (max(w, target_size), max(h, target_size)), interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape

        top = (h - target_size) // 2
        left = (w - target_size) // 2
        img = img[top:top + target_size, left:left + target_size, ...]
        return img

    def __len__(self):
        return len(self.paths)
