import cv2
import numpy as np
import os.path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.data_util import paths_from_lmdb, scandir
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class ImageNetUnpairedDataset(data.Dataset):
    """Unpaired ImageNet dataset for MAE pre-training.

    Args:
        opt (dict): Config for dataset. It contains the following keys:
            dataroot (str): Data root path.
            io_backend (dict): IO backend type and other settings.
            mean (list[float]): Mean value of RGB channels.
            std (list[float]): Standard deviation of RGB channels.
            gt_size (int): Cropped patch size for training.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flips).
    """

    def __init__(self, opt):
        super(ImageNetUnpairedDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])
        self.dataroot = opt['dataroot']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.dataroot]
            self.io_backend_opt['client_keys'] = ['lq']
            self.paths = paths_from_lmdb(self.dataroot)
        elif 'meta_info_file' in self.opt:
            with open(self.opt['meta_info_file'], 'r') as fin:
                self.paths = [osp.join(self.dataroot, line.split(' ')[0]) for line in fin]
        else:
            self.paths = sorted(list(scandir(self.dataroot, full_path=True)))

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # Load image. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        img_path = self.paths[index]
        img_bytes = self.file_client.get(img_path, 'lq')
        img = imfrombytes(img_bytes, float32=True)

        gt_size = self.opt.get('gt_size', 224)

        # augmentation
        if self.opt.get('phase') == 'train':
            # random crop
            img = self.random_crop(img, gt_size, img_path)
            # flip, rotation
            img = augment(img, self.opt['use_hflip'], self.opt['use_rot'])
        else:
            # validation: center crop
            img = self.center_crop(img, gt_size)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img = img2tensor(img, bgr2rgb=True, float32=True)

        # normalize
        normalize(img, self.mean, self.std, inplace=True)

        return {'lq': img, 'lq_path': img_path}

    def center_crop(self, img, target_size):
        h, w, _ = img.shape
        if h < target_size or w < target_size:
            img = cv2.resize(img, (max(w, target_size), max(h, target_size)), interpolation=cv2.INTER_AREA)
            h, w, _ = img.shape

        top = (h - target_size) // 2
        left = (w - target_size) // 2
        img = img[top:top + target_size, left:left + target_size, ...]
        return img

    def random_crop(self, img, crop_size, img_path):
        h, w, _ = img.shape
        if h < crop_size or w < crop_size:
            # pad if image is smaller than crop size
            pad_h = max(0, crop_size - h)
            pad_w = max(0, crop_size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            h, w, _ = img.shape

        top = np.random.randint(0, h - crop_size + 1)
        left = np.random.randint(0, w - crop_size + 1)
        img = img[top:top + crop_size, left:left + crop_size, ...]
        return img

    def __len__(self):
        return len(self.paths)
