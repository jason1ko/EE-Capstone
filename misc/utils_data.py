import torch
import random
from PIL import ImageFilter
import numpy as np

'''
Data statistics
'''

def obtain_data_stats(data_name):

    if data_name in ['cifar10', 'cifar10+', 'cifar50+', 'cub']:
        data_mean = torch.FloatTensor([0.4914, 0.4822, 0.4465])
        data_std = torch.FloatTensor([0.2023, 0.1994, 0.2010])
    elif data_name in ['imagenet']:
        data_mean = torch.FloatTensor([0.485, 0.456, 0.406])
        data_std = torch.FloatTensor([0.229, 0.224, 0.225])
    else:
        data_mean = torch.FloatTensor([0.5, 0.5, 0.5])
        data_std = torch.FloatTensor([0.5, 0.5, 0.5])


    if data_name == 'tinyimagenet':
        img_size = 64
    elif data_name == 'cub':
        img_size = 128      # for the pretrained (moco_v2_places), should be 448
    else:
        img_size = 32

    return data_mean, data_std, img_size


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

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