import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util
import albumentations as A
import numpy as np
import cv2
import torch

class AdaptationDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_source = os.path.join(opt.dataroot, 'ade/ADEChallengeData2016/images/training')  # create a path '/path/to/data/trainA'
        self.dir_source_label = os.path.join(opt.dataroot, 'ade/ADEChallengeData2016/annotations/training')  # create a path '/path/to/data/trainA'
        self.dir_target = os.path.join(opt.dataroot, 'soda/InfraredSemanticLabel/images/training')  # create a path '/path/to/data/trainB'
        self.dir_target_label = os.path.join(opt.dataroot, 'soda/InfraredSemanticLabel/annotations/training')  # create a path '/path/to/data/trainB'
        self.label_file = os.path.join(opt.dataroot, 'ade/ADEChallengeData2016/objectInfo150.txt')

        if opt.phase == "test":
            self.dir_source = self.dir_source.replace('training', 'validataion')
            self.dir_source_label = self.dir_source_label.replace('training', 'validation')
            self.dir_target = self.dir_target.replace('training', 'validation')

        self.source_paths = sorted(make_dataset(self.dir_source, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.source_label_paths = sorted(make_dataset(self.dir_source_label, opt.max_dataset_size))
        self.target_paths = sorted(make_dataset(self.dir_target, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.source_size = len(self.source_paths)  # get the size of dataset A
        self.target_size = len(self.target_paths)  # get the size of dataset B
        self.source_label_mapping = dict()
        with open(self.label_file, 'r') as f:
            for line in f:
                line_list = [i for i in line.split()]
                self.source_label_mapping[int(line_list[0])] = int(line_list[1])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        source_path = self.source_paths[index % self.source_size]  # make sure index is within then range
        source_label_path = self.source_label_paths[index % self.source_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_target = index % self.target_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_target = random.randint(0, self.target_size - 1)
        target_path = self.target_paths[index_target]
        target_label_path = target_path.replace('images', 'annotations').replace('jpg', 'png')
        
        # Read images
        source_img = cv2.imread(source_path, cv2.IMREAD_GRAYSCALE)
        source_label = cv2.imread(source_label_path, cv2.IMREAD_UNCHANGED)
        target_img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
        target_label = np.zeros(target_img.shape, np.uint8)
        if os.path.exists(target_label_path):
            target_label = cv2.imread(target_label_path, cv2.IMREAD_UNCHANGED)
        
        # Apply image transformation
        transform = A.Compose([
            A.RandomResizedCrop(height=self.opt.crop_size, width=self.opt.crop_size, scale=(0.5, 2.0), ratio=(1.0, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5)
        ])
        source_augmented = transform(image=source_img, mask=source_label)
        source = source_augmented['image']
        source = 2*(source/255) - 1
        source = torch.from_numpy(source).view(1, source.shape[0], source.shape[1]).float()
        source = source.repeat(3, 1, 1)
        source_label = source_augmented['mask']
        source_label_copy = np.zeros(source_label.shape, dtype=np.uint8)
        for k, v in self.source_label_mapping.items():
            source_label_copy[source_label==k] = v
        source_label = torch.from_numpy(source_label_copy).type(torch.LongTensor)

        target_augmented = transform(image=target_img, mask=target_label)
        target = target_augmented['image']
        target = 2*(target/255) - 1
        target = torch.from_numpy(target).view(1, target.shape[0], target.shape[1]).float()
        target = target.repeat(3, 1, 1)
        target_label = target_augmented['mask']
        target_label = torch.from_numpy(target_label).type(torch.LongTensor)

        return {'A': source, 'A_label': source_label, 'B': target, 'B_label': target_label, 'A_paths' : source_path, 'B_paths': target_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.source_size, self.target_size)
