import cv2
import numpy as np
from torch.utils import data as data

from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torchvision.transforms.functional import normalize


@DATASET_REGISTRY.register()
class PairedImageResizeDataset(data.Dataset):
    """Paired image dataset that resizes and crops images to a fixed size.

    It reads HR and LR images from two folders.
    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other settings.
            scale (int): Scale factor.
            gt_size (int): Cropped patch size for gt patches.
    """

    def __init__(self, opt):
        super(PairedImageResizeDataset, self).__init__()
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
        gt_size = self.opt['gt_size']

        # Load gt and lq images
        gt_path = self.paths[index]['gt_path']
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        lq_path = self.paths[index]['lq_path']
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        # Resize and crop to fixed size
        img_gt = self.resize_and_crop(img_gt, gt_size)
        lq_size = gt_size // scale
        img_lq = self.resize_and_crop(img_lq, lq_size)

        # Augmentation
        img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        # Normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def resize_and_crop(self, img, target_size):
        h, w, _ = img.shape
        # resize to maintain aspect ratio
        if h < w:
            new_h = target_size
            new_w = int(w * new_h / h)
        else:
            new_w = target_size
            new_h = int(h * new_w / w)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # center crop
        top = (new_h - target_size) // 2
        left = (new_w - target_size) // 2
        img = img[top:top + target_size, left:left + target_size, ...]
        return img

    def __len__(self):
        return len(self.paths)
