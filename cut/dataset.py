import numpy as np
import os.path
from PIL import Image
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from . import hyperparameters as hyperparams


class SingleImageDataset(data.Dataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """
    phase = "train"
    random_scale_max=3
    def __init__(self, A_img,B_img):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """

        self.A_img = A_img
        self.B_img = B_img
        # In single-image translation, we augment the data loader by applying
        # random scaling. Still, we design the data loader such that the
        # amount of scaling is the same within a minibatch. To do this,
        # we precompute the random scaling values, and repeat them by |batch_size|.
        A_zoom = 1 / self.random_scale_max
        zoom_levels_A = np.random.uniform(A_zoom, 1.0, size=(len(self) // hyperparams.cut_batch_size + 1, 1, 2))
        self.zoom_levels_A = np.reshape(np.tile(zoom_levels_A, (1, hyperparams.cut_batch_size, 1)), [-1, 2])

        B_zoom = 1 / self.random_scale_max
        zoom_levels_B = np.random.uniform(B_zoom, 1.0, size=(len(self) // hyperparams.cut_batch_size + 1, 1, 2))
        self.zoom_levels_B = np.reshape(np.tile(zoom_levels_B, (1, hyperparams.cut_batch_size, 1)), [-1, 2])

        # While the crop locations are randomized, the negative samples should
        # not come from the same location. To do this, we precompute the
        # crop locations with no repetition.
        self.patch_indices_A = list(range(len(self)))
        random.shuffle(self.patch_indices_A)
        self.patch_indices_B = list(range(len(self)))
        random.shuffle(self.patch_indices_B)
        

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
        A_img = self.A_img
        B_img = self.B_img
        # apply image transformation
        if self.phase == "train":
            param = {'scale_factor': self.zoom_levels_A[index],
                     'patch_index': self.patch_indices_A[index],
                     'flip': random.random() > 0.5}

            transform_A = get_transform( params=param, method=Image.BILINEAR)
            A = transform_A(torch.squeeze(A_img).cpu())

            param = {'scale_factor': self.zoom_levels_B[index],
                     'patch_index': self.patch_indices_B[index],
                     'flip': random.random() > 0.5}
            transform_B = get_transform( params=param, method=Image.BILINEAR)
            B = transform_B(torch.squeeze(B_img).cpu())
        else:
            transform = get_transform( method=Image.BILINEAR)
            A = transform(A_img)
            B = transform(B_img)

        return A,B

    def __len__(self):
        
        """ Let's pretend the single image contains 100,000 crops for convenience.
        """
        return 1


def get_transform( params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    crop_size=64
    transform_list = []
    transform_list.append(transforms.ToPILImage())
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    transform_list.append(transforms.Lambda(lambda img: __random_zoom(img, crop_size, method, factor=params["scale_factor"])))
    transform_list.append(transforms.Lambda(lambda img: __patch(img, params['patch_index'], crop_size)))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __random_zoom(img,  crop_width, method=Image.BICUBIC, factor=None):
    if factor is None:
        zoom_level = np.random.uniform(0.8, 1.0, size=[2])
    else:
        zoom_level = (factor[0], factor[1])
    iw, ih = img.size
    zoomw = max(crop_width, iw * zoom_level[0])
    zoomh = max(crop_width, ih * zoom_level[1])
    img = img.resize((int(round(zoomw)), int(round(zoomh))), method)
    return img

def __patch(img, index, size):
    ow, oh = img.size
    nw, nh = ow // size, oh // size
    roomx = ow - nw * size
    roomy = oh - nh * size
    startx = np.random.randint(int(roomx) + 1)
    starty = np.random.randint(int(roomy) + 1)

    index = index % (nw * nh)
    ix = index // nh
    iy = index % nh
    gridx = startx + ix * size
    gridy = starty + iy * size
    return img.crop((gridx, gridy, gridx + size, gridy + size))