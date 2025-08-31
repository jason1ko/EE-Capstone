import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import ImageFilter
import random

'''
torch utils
'''
class ShufflePatches(object):
    def __init__(self, patch_size):
        self.ps = patch_size

    def __call__(self, x):
        # divide the batch of images into non-overlapping patches
        u = F.unfold(x.unsqueeze(0), kernel_size=self.ps, stride=self.ps)
        # permute the patches of each image in the batch
        pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
        # fold the permuted patches back together
        f = F.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps).squeeze(dim=0)
        k = torch.randint(low=1,high=4,size=(1,))
        f = torch.rot90(f,k=int(k), dims=(1,2))
        return f


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class DualTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class QKNTransform(object):
    def __init__(self, base_transform, patch_size):
        self.base_transform = base_transform
        self.neg_transform = ShufflePatches(patch_size=patch_size)

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        n = self.neg_transform(q)
        return [q, k, n]


def obtain_crop_augmentations(img_size, scale=(0.08, 1.0)):
    # csi
    # augmentations = [
    #         transforms.RandomResizedCrop(img_size, scale=scale),
    #         transforms.RandomHorizontalFlip(p=0.5),
    #         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    #         transforms.RandomGrayscale(p=0.2)
    # ]

    # moco v2 imagenet
    augmentations = [
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(p=0.5)
    ]
    return augmentations


def obtain_mask_augmentations(img_size, p=0.75):
    # TODO: work in progress
    return 0

def obtain_data_stats(data_name):
    if data_name == 'mnist':
        data_mean = torch.FloatTensor([0.5])
        data_std = torch.FloatTensor([0.5])
    elif data_name in ['cifar10', 'cifar10+', 'cifar50+']:
        data_mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
        data_std = torch.FloatTensor([0.2023, 0.1994, 0.2010])
    elif data_name == 'cub':
        data_mean = (0.4914, 0.4822, 0.4465)
        data_std = (0.2023, 0.1994, 0.2010)
    else:
        data_mean = torch.FloatTensor([0.5, 0.5, 0.5])
        data_std = torch.FloatTensor([0.5, 0.5, 0.5])

    if data_name == 'mnist':
        img_size = 28
    elif data_name == 'tinyimagenet':
        img_size = 64
    elif data_name == 'cub':
        img_size = 128      # for the pretrained (moco_v2_places), should be 448
    else:
        img_size = 32

    return data_mean, data_std, img_size

'''
general utils
'''
import os
# download link: https://github.com/rmccorm4/Tiny-Imagenet-200
def create_val_img_folder(root):
    '''
    This method is responsible for separating validation images into separate sub folders
    Run this before running TinyImageNet experiments
    :param root: Root dir for TinyImageNet, e.g /work/sagar/datasets/tinyimagenet/tiny-imagenet-200/
    '''
    dataset_dir = os.path.join(root)
    val_dir = os.path.join(dataset_dir, 'val')
    img_dir = os.path.join(val_dir, 'images')

    fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
    data = fp.readlines()
    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create folder if not present and move images into proper folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))


'''
data making utils
'''
import numpy as np
from skimage.filters import gaussian as gblur
def blobs(size=32):
    """
    Minimal version since we use this dataset only for OOD evaluation.
    """
    data = np.float32(np.random.binomial(n=1, p=0.7, size=(10000, size, size, 3)))
    for i in range(10000):
        data[i] = gblur(data[i], sigma=1.5, multichannel=False)
        data[i][data[i] < 0.75] = 0.0

    # dummy_targets = torch.ones(10000)
    # data = torch.cat(
    #     [
    #         norm_layer(x).unsqueeze(0)
    #         for x in torch.from_numpy(data.transpose((0, 3, 1, 2)))
    #     ]
    # )
    # dataset = torch.utils.data.TensorDataset(data, dummy_targets)
    # loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    # )
    data = np.array(255 * data, dtype=np.uint8)

    return data