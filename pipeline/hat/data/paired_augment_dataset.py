import cv2
import numpy as np
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import normalize


@DATASET_REGISTRY.register()
class PairedAugmentDataset(data.Dataset):
    """Paired image dataset with specific augmentations for training and validation.

    It reads HR and LR images from two folders.
    During training, it performs random cropping.
    During validation, it performs center cropping.

    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other settings.
            scale (int): Scale factor.
            gt_size (int): Cropped patch size for gt patches.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedAugmentDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', None)
        self.std = opt.get('std', None)

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.gt_folder)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']

        # Load gt and lq images
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Augmentation
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

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def center_crop(self, img, target_size):
        h, w, _ = img.shape
        if h < target_size or w < target_size:
            # If image is smaller than target size, resize it up
            img = cv2.resize(img, (max(w, target_size), max(h, target_size)), interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape

        top = (h - target_size) // 2
        left = (w - target_size) // 2
        img = img[top:top + target_size, left:left + target_size, ...]
        return img

    def __len__(self):
        return len(self.paths)
