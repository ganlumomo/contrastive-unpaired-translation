import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import albumentations as A
import numpy as np
import cv2
import torch


class SegmentationDataset(BaseDataset):
    """This dataset class can load a set of images specified by the path --dataroot /path/to/data.

    It can be used for generating CycleGAN results only for one side with the model option '-model test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_image = os.path.join(opt.dataroot, 'soda/InfraredSemanticLabel/images/training')
        self.dir_label = os.path.join(opt.dataroot, 'soda/InfraredSemanticLabel/annotations/training')
        
        if opt.phase == "test":
            self.dir_image = self.dir_image.replace('training', 'validation')
            self.dir_label = self.dir_label.replace('training', 'validation')

        self.image_paths = sorted(make_dataset(self.dir_image, opt.max_dataset_size))
        self.label_paths = sorted(make_dataset(self.dir_label, opt.max_dataset_size))

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A and A_paths
            A(tensor) - - an image in one domain
            A_paths(str) - - the path of the image
        """
        image_path = self.image_paths[index]
        label_path = self.label_paths[index]
        
        # Read images
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # Apply image transformation
        transform = A.Compose([
            A.RandomResizedCrop(height=self.opt.crop_size, width=self.opt.crop_size, scale=(0.5, 2.0), ratio=(1.0, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5)
        ])
        if self.opt.phase == "test":
            transform = A.Compose([
                A.Resize(height=self.opt.crop_size, width=self.opt.crop_size, always_apply=True),
            ])

        augmented = transform(image=image, mask=label)
        image = augmented['image']
        image = 2*(image/255) - 1
        image = torch.from_numpy(image).view(1, image.shape[0], image.shape[1]).float()
        image = image.repeat(3, 1, 1)
        label = augmented['mask']
        label = torch.from_numpy(label).type(torch.LongTensor)

        return {'A': image, 'A_label': label, 'A_paths': image_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.image_paths)
