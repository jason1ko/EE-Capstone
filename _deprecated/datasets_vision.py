import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np
import os

# from data_builder.misc.superclass_cifar100 import label_cifar100_by_superclasses
# from data_builder.misc.osr_splits import osr_splits
# from data_builder.misc.data_utils import blobs

from misc.utils_data import blobs
from misc.data_misc.misc_cifar100 import label_cifar100_by_superclasses
from misc.data_misc.misc_osr import osr_splits


def preprocess_image(img_path):
    img = Image.open(img_path)
    if img.size[0] != 32:
        img = img.resize((32, 32))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

def preprocess_mnist(data):
    N = len(data)
    out_data = [[]]*N

    for i in range(N):
        _img = data[i]
        _img = Image.fromarray(_img)
        _img = _img.resize((32, 32))
        _img = _img.convert('RGB')
        out_data[i] = np.array(_img)

    return np.array(out_data)

class ToyDataset(Dataset):
    def __init__(self, data_root_path, data_name, train, transform):
        self.transform = transform

        if data_name == 'cifar10':
            data_path = data_root_path + '/' + data_name
            self.img_shape = (3, 32, 32)
            _dataset = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True)
            self.samples = np.array(_dataset.data)
            self.targets = np.array(_dataset.targets, dtype=np.int64)
        elif data_name == 'svhn':
            data_path = data_root_path + '/' + data_name
            _dataset = torchvision.datasets.SVHN(root=data_path, split=train, download=True)
            self.samples = np.array(_dataset.data.transpose((0, 2, 3, 1)))
            self.targets = np.array(_dataset.labels, dtype=np.int64)
        elif data_name == 'mnist':
            data_path = data_root_path + '/' + data_name
            _dataset = torchvision.datasets.MNIST(root=data_path, train=train, download=True)
            self.samples = preprocess_mnist(np.array(_dataset.data))
            self.targets = np.array(_dataset.targets, dtype=np.int64)
        else:
            raise NotImplementedError()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img, target = self.samples[index], self.targets[index]  # get imgs in numpy uint8
        img = Image.fromarray(img)  # change numpy to PIL
        out = self.transform(img)
        return out, target



class ToyDatasetOSR(Dataset):
    def __init__(self, data_root_path, data_name, split_idx, train, transform, ood=False):
        self.transform = transform

        if data_name == 'cifar10':
            data_path = data_root_path + '/' + data_name

            self.img_shape = (3, 32, 32)

            _dataset_ind = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True)
            _dataset_ood = _dataset_ind

            ind_predata = _dataset_ind.data
            ood_predata = _dataset_ood.data

            ind_pretargets = _dataset_ind.targets
            ood_pretargets = _dataset_ood.targets

            ind_classes = np.array(osr_splits['cifar-10-10'][split_idx])
            ood_classes = np.array(list(set(np.arange(10)).difference(set(ind_classes))))

        elif data_name == 'cifar10+':
            self.img_shape = (3, 32, 32)

            _dataset_ind = torchvision.datasets.CIFAR10(root=data_root_path + '/cifar10', train=train, download=True)
            _dataset_ood = torchvision.datasets.CIFAR100(root=data_root_path + '/cifar100', train=False, download=True)

            ind_predata = _dataset_ind.data
            ood_predata = _dataset_ood.data

            ind_pretargets = _dataset_ind.targets
            ood_pretargets = _dataset_ood.targets

            ind_classes = np.array(osr_splits['cifar-10-100'][split_idx])
            ood_classes = np.array(osr_splits['cifar-10-100-10'][split_idx])

        elif data_name == 'cifar50+':
            self.img_shape = (3, 32, 32)

            _dataset_ind = torchvision.datasets.CIFAR10(root=data_root_path + '/cifar10', train=train, download=True)
            _dataset_ood = torchvision.datasets.CIFAR100(root=data_root_path + '/cifar100', train=False, download=True)

            ind_predata = _dataset_ind.data
            ood_predata = _dataset_ood.data

            ind_pretargets = _dataset_ind.targets
            ood_pretargets = _dataset_ood.targets

            ind_classes = np.array(osr_splits['cifar-10-100'][split_idx])
            ood_classes = np.array(osr_splits['cifar-10-100-50'][split_idx])

        elif data_name == 'svhn':
            data_path = data_root_path + '/' + data_name

            self.img_shape = (3, 32, 32)

            fold = 'train' if train else 'test'
            _dataset_ind = torchvision.datasets.SVHN(root=data_path, split=fold, download=True)
            _dataset_ood = torchvision.datasets.SVHN(root=data_path, split='test', download=True)

            ind_predata = _dataset_ind.data.transpose((0, 2, 3, 1))
            ood_predata = _dataset_ood.data.transpose((0, 2, 3, 1))

            ind_pretargets = _dataset_ind.labels
            ood_pretargets = _dataset_ood.labels

            ind_classes = np.array(osr_splits['svhn'][split_idx])
            ood_classes = np.array(list(set(np.arange(10)).difference(set(ind_classes))))

        elif data_name == 'mnist':
            data_path = data_root_path + '/' + data_name

            self.img_shape = (3, 32, 32)
            _dataset_ind = torchvision.datasets.MNIST(root=data_path, train=train, download=True)
            _dataset_ood = torchvision.datasets.MNIST(root=data_path, train=False, download=True)

            ind_predata = preprocess_mnist(np.array(_dataset_ind.data))
            ood_predata = preprocess_mnist(np.array(_dataset_ood.data))

            ind_pretargets = _dataset_ind.targets
            ood_pretargets = _dataset_ood.targets

            ind_classes = np.array(osr_splits['mnist'][split_idx])
            ood_classes = np.array(osr_splits['mnist'][split_idx])

        else:
            raise ValueError('Incorrect root')

        self.num_ind_classes = len(ind_classes)

        idxes_ind_classes = np.in1d(np.array(ind_pretargets), ind_classes)
        idxes_ood_classes = np.in1d(np.array(ood_pretargets), ood_classes)

        ind_data = np.array(ind_predata)[idxes_ind_classes]
        ood_data = np.array(ood_predata)[idxes_ood_classes]

        ind_targets_ = np.array(ind_pretargets)[idxes_ind_classes]
        ood_targets_ = np.array(ood_pretargets)[idxes_ood_classes]

        dict_class2label = {k: v for v, k in enumerate(ind_classes)}
        ind_targets = np.vectorize(dict_class2label.get)(np.array(ind_targets_))
        ood_targets = np.ones_like(ood_targets_)*self.num_ind_classes
        ind_reject_targets = np.zeros_like(ind_targets)
        ood_reject_targets = np.ones_like(ood_targets)

        if ood:
            data = ood_data
            reject_targets = ood_reject_targets
            targets = ood_targets
        else:
            data = ind_data
            reject_targets = ind_reject_targets
            targets = ind_targets

        self.reject_targets = np.array(reject_targets, dtype=np.int64)
        self.targets = np.array(targets, dtype=np.int64)
        self.data = data

    def __len__(self):
        return self.targets.__len__()

    def __getitem__(self, index):
        img, reject_target, target = self.data[index], self.reject_targets[index], self.targets[index]  # get imgs in numpy uint8
        img = Image.fromarray(img)  # change numpy to PIL
        out = self.transform(img)
        return out, reject_target, target

# tiny-imagenet-200 (original): https://github.com/sgvaze/osr_closed_set_all_you_need
# tiny-imagenet-200 (organized): https://github.com/iCGY96/ARPL
class DatasetOSR(Dataset):
    def __init__(self, data_root_path, data_name, split_idx, train, transform, ood=False):
        self.transform = transform

        if data_name == 'tinyimagenet':

            data_path = data_root_path + '/' + data_name

            self.img_shape = (3, 64, 64)

            if train:
                data_root_ind = data_path + '/tiny-imagenet-200/train'
            else:
                data_root_ind = data_path + '/tiny-imagenet-200/val'

            data_root_ood = data_path + '/tiny-imagenet-200/val'

            _dataset_ind = ImageFolder(root=data_root_ind)
            _dataset_ood = ImageFolder(root=data_root_ood)

            ind_predata = [sample[0] for sample in _dataset_ind.samples]
            ood_predata = [sample[0] for sample in _dataset_ood.samples]

            ind_pretargets = _dataset_ind.targets
            ood_pretargets = _dataset_ood.targets

            ind_classes = np.array(osr_splits['tinyimagenet'][split_idx])
            ood_classes = np.array(list(set(np.arange(200)).difference(set(ind_classes))))

            idxes_ind_classes = np.in1d(np.array(ind_pretargets), ind_classes)
            idxes_ood_classes = np.in1d(np.array(ood_pretargets), ood_classes)

        else:
            raise ValueError('Incorrect root')

        self.num_ind_classes = len(ind_classes)

        ind_data = np.array(ind_predata)[idxes_ind_classes]
        ood_data = np.array(ood_predata)[idxes_ood_classes]

        ind_targets_ = np.array(ind_pretargets)[idxes_ind_classes]
        ood_targets_ = np.array(ood_pretargets)[idxes_ood_classes]

        dict_class2label = {k: v for v, k in enumerate(ind_classes)}
        ind_targets = np.vectorize(dict_class2label.get)(np.array(ind_targets_))
        ood_targets = np.ones_like(ood_targets_)*self.num_ind_classes
        ind_reject_targets = np.zeros_like(ind_targets)
        ood_reject_targets = np.ones_like(ood_targets)

        if ood:
            data = ood_data
            reject_targets = ood_reject_targets
            targets = ood_targets
        else:
            data = ind_data
            reject_targets = ind_reject_targets
            targets = ind_targets

        self.reject_targets = np.array(reject_targets, dtype=np.int64)
        self.targets = np.array(targets, dtype=np.int64)
        self.data = data

    def __len__(self):
        return self.targets.__len__()

    def __getitem__(self, index):
        img, reject_target, target = self.data[index], self.reject_targets[index], self.targets[index]  # get imgs in numpy uint8
        img = Image.open(img)  # change numpy to PIL
        if img.mode != 'RGB':
            img = img.convert('RGB')
        out = self.transform(img)
        return out, reject_target, target


class ToyDatasetOCC(Dataset):
    def __init__(self, data_root_path, data_name, split_idx, train, transform, ood=False):

        self.transform = transform

        data_path = data_root_path + '/' + data_name
        if data_name == 'mnist':
            loaded_data = torchvision.datasets.MNIST(root=data_path, train=train, download=True)
            self.img_shape = (1, 28, 28)

        elif data_name == 'cifar10':
            loaded_data = torchvision.datasets.CIFAR10(root=data_path, train=train, download=True)
            self.img_shape = (3, 32, 32)

        elif data_name == 'cifar100':
            loaded_data = torchvision.datasets.CIFAR100(root=data_path, train=train, download=True)
            super_targets = label_cifar100_by_superclasses(loaded_data.targets, loaded_data.classes)
            loaded_data.targets = list(super_targets)
            self.img_shape = (3, 32, 32)

        else:
            raise ValueError('Incorrect root')

        idxes_ind_classes = np.array(loaded_data.targets) == split_idx
        idxes_ood_classes = np.logical_not(idxes_ind_classes)

        ind_data = np.array(loaded_data.data)[idxes_ind_classes]
        ood_data = np.array(loaded_data.data)[idxes_ood_classes]

        ind_reject_targets = np.zeros(shape=(len(ind_data), ))
        ood_reject_targets = np.ones(shape=(len(ood_data), ))

        if ood:
            data = ood_data
            reject_targets = ood_reject_targets
        else:
            data = ind_data
            reject_targets = ind_reject_targets

        # tile
        self.reject_targets = np.array(reject_targets, dtype=np.int64)
        self.data = data

    def __len__(self):
        return self.reject_targets.__len__()

    def __getitem__(self, index):
        img, target = self.data[index], self.reject_targets[index]     # get imgs in numpy uint8
        img = Image.fromarray(img)      # change numpy to PIL
        out = self.transform(img)
        return out, target


class ToyDatasetOoD(Dataset):
    def __init__(self, data_root_path, data_name, split_idx, train, transform, ood=False):
        self.transform = transform

        '''
        Set up InD data
        '''
        if data_name == 'cifar10':
            ind_data_root_path = data_root_path + '/' + data_name
            self.img_shape = (3, 32, 32)
            loaded_data_ind = torchvision.datasets.CIFAR10(root=ind_data_root_path, train=train, download=True)

            ind_data = np.array(loaded_data_ind.data)
            ind_targets = np.array(loaded_data_ind.targets, dtype=np.int64)

            preprocess_ind = Image.fromarray

        elif data_name == 'cifar100':
            ind_data_root_path = data_root_path + '/' + data_name
            self.img_shape = (3, 32, 32)
            loaded_data_ind = torchvision.datasets.CIFAR100(root=ind_data_root_path, train=train, download=True)

            ind_data = np.array(loaded_data_ind.data)
            ind_targets = np.array(loaded_data_ind.targets, dtype=np.int64)

            preprocess_ind = Image.fromarray

        elif data_name == 'svhn':
            ind_data_root_path = data_root_path + '/' + data_name
            self.img_shape = (3, 32, 32)
            fold = 'train' if train else 'test'
            loaded_data_ind = torchvision.datasets.SVHN(root=ind_data_root_path, split=fold, download=True)

            ind_data = loaded_data_ind.data.transpose((0,2,3,1))
            ind_targets = loaded_data_ind.labels

            preprocess_ind = Image.fromarray

        elif data_name == 'mnist':

            ind_data_dir_path = data_root_path + '/' + data_name
            self.img_shape = (3, 32, 32)
            ind_data_source = torchvision.datasets.MNIST(root=ind_data_dir_path, train=train, download=True)
            ind_data = np.array(ind_data_source.data)
            ind_targets = np.array(ind_data_source.targets)

            def preprocess_ind(x):
                img = Image.fromarray(x)
                img = img.resize((32, 32))
                img = img.convert('RGB')
                return img

        else:
            raise ValueError('Incorrect root')

        '''
        Set up OoD data
        '''
        # download link - 1: https://github.com/facebookresearch/odin
        # download link - 2: https://github.com/alinlab/CSI

        if split_idx in [0, 1, 2, 3, 4, 5, 6]:

            if split_idx == 0:
                ood_data_path = data_root_path + '/lsun_fix/LSUN_pil/LSUN_pil'
            elif split_idx == 1:
                ood_data_path = data_root_path + '/lsun_resize/LSUN_resize/LSUN_resize'
            elif split_idx == 2:
                ood_data_path = data_root_path + '/lsun_crop/LSUN/test'
            elif split_idx == 3:
                ood_data_path = data_root_path + '/imagenet_fix/Imagenet_pil/Imagenet_pil'
            elif split_idx == 4:
                ood_data_path = data_root_path + '/imagenet_resize/Imagenet_resize/Imagenet_resize'
            elif split_idx == 5:
                ood_data_path = data_root_path + '/imagenet_crop/Imagenet/test'
            elif split_idx == 6:
                ood_data_path = data_root_path + '/isun/iSUN/iSUN_patches'
            else:
                raise NotImplementedError()

            ood_data = []

            for file_path in os.listdir(ood_data_path):
                ood_data.append(ood_data_path + '/' + file_path)

            preprocess_ood = preprocess_image

        elif split_idx == 7:    # ood = cifar10
            ood_data_path = data_root_path + '/cifar10'
            ood_set = torchvision.datasets.CIFAR10(root=ood_data_path, train=False, download=True)
            ood_data = ood_set.data
            ood_data = np.array(ood_data)
            preprocess_ood = Image.fromarray

        elif split_idx == 8:    # ood = cifar100
            ood_data_path = data_root_path + '/cifar100'
            ood_set = torchvision.datasets.CIFAR100(root=ood_data_path, train=False, download=True)
            ood_data = ood_set.data
            ood_data = np.array(ood_data)
            preprocess_ood = Image.fromarray

        elif split_idx == 9:  # ood = svhn
            ood_data_path = data_root_path + '/svhn'
            fold = 'train' if train else 'test'
            ood_set = torchvision.datasets.SVHN(root=ood_data_path, split=fold, download=True)
            ood_data = ood_set.data.transpose((0, 2, 3, 1))
            ood_data = np.array(ood_data)
            preprocess_ood = Image.fromarray

        elif split_idx == 10:   # ood = mnist
            ood_data_path = data_root_path + '/mnist'

            ood_set = torchvision.datasets.MNIST(root=ood_data_path, train=train, download=True)
            ood_data = np.array(ood_set.data)
            ood_data = preprocess_mnist(ood_data)
            preprocess_ood = Image.fromarray

        elif split_idx == 11:    # ood = texture (dtd)
            ood_data_path = data_root_path + '/texture/dtd/images'
            _ood_dataset = ImageFolder(ood_data_path)
            ood_data = [sample[0] for sample in _ood_dataset.samples]
            del _ood_dataset
            preprocess_ood = preprocess_image

        elif split_idx == 12:   # ood = blob
            ood_data = blobs(32)
            preprocess_ood = Image.fromarray

        else:
            raise NotImplementedError()

        self.num_ind_classes = len(set(ind_targets))

        ood_targets = np.ones((len(ood_data)), dtype=np.int64)*self.num_ind_classes
        ind_reject_targets = np.zeros_like(ind_targets)
        ood_reject_targets = np.ones_like(ood_targets)

        if ood:
            data = ood_data
            reject_targets = ood_reject_targets
            targets = ood_targets
            self.preprocess = preprocess_ood
        else:
            data = ind_data
            reject_targets = ind_reject_targets
            targets = ind_targets
            self.preprocess = preprocess_ind

        self.reject_targets = np.array(reject_targets, dtype=np.int64)
        self.targets = np.array(targets, dtype=np.int64)
        self.data = data

    def __len__(self):
        return self.targets.__len__()

    def __getitem__(self, index):
        img, reject_target, target = self.data[index], self.reject_targets[index], self.targets[index]  # get imgs in numpy uint8
        img = self.preprocess(img)  # change numpy to PIL
        out = self.transform(img)
        return out, reject_target, target



# TODO: fix to 'task, data_name, split, transform'
def get_dataset(data_root_path, task, data_name, split_idx, train, transform, ood=False):

    if task == 'osr':
        if data_name in ['tinyimagenet']:
            DatasetTask = DatasetOSR
        else:
            DatasetTask = ToyDatasetOSR
    elif task == 'occ':
        DatasetTask = ToyDatasetOCC
    elif task == 'ood':
        DatasetTask = ToyDatasetOoD
    else:
        raise NotImplementedError()

    dataset = DatasetTask(data_root_path=data_root_path, data_name=data_name, train=train, transform=transform, ood=ood,
                          split_idx=split_idx)

    return dataset
    # TODO: fix so as to return only one dataset


def get_dataloaders(args, transform_train, transform_test):
    trainset = get_dataset(args.data_root_path, args.task, args.data_name, args.split_idx, train=True,
                           transform=transform_train, ood=False)
    bankset_ind = get_dataset(args.data_root_path, args.task, args.data_name, args.split_idx, train=True,
                              transform=transform_test, ood=False)
    queryset_ind = get_dataset(args.data_root_path, args.task, args.data_name, args.split_idx, train=False,
                               transform=transform_test, ood=False)

    '''
    Preparing OoD dataset
    '''
    if args.ood_data_name is None:
        queryset_ood = get_dataset(args.data_root_path, args.task, args.data_name, args.split_idx, train=False,
                                   transform=transform_test, ood=True)
    else:
        queryset_ood = get_dataset(args.data_root_path, args.ood_task, args.ood_data_name, args.ood_split_idx,
                                   train=False, transform=transform_test, ood=True)

    if args.task != 'occ':
        queryset_ood.targets = np.ones((len(queryset_ood)), dtype=np.int64) * queryset_ind.num_ind_classes

    '''
    Preparing dataloaders
    '''
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                             pin_memory=False, drop_last=True)
    bankloader_ind = DataLoader(bankset_ind, batch_size=args.batch_size_test, shuffle=False,
                                num_workers=args.num_workers)
    queryloader_ind = DataLoader(queryset_ind, batch_size=args.batch_size_test, shuffle=False,
                                 num_workers=args.num_workers)

    queryloader_ood = DataLoader(queryset_ood, batch_size=args.batch_size_test, shuffle=False,
                                 num_workers=args.num_workers)

    return trainloader, bankloader_ind, queryloader_ind, queryloader_ood


def modify_dataset_intra_class_size(dataset, intra_class_size=1):
    n_samples_original = len(dataset)

    classes = list(set(dataset.targets))

    n_samples_per_class = {}

    # initialize
    for k in classes:
        n_samples_per_class[k] = 0

    new_data = []
    new_targets = []

    for i in range(len(dataset)):
        _k = dataset.targets[i]
        if n_samples_per_class[_k] < intra_class_size:
            new_data.append(dataset.data[i])
            new_targets.append(dataset.targets[i])
            n_samples_per_class[_k] += 1
        else:
            pass

    # debug
    for _k in n_samples_per_class:
        assert n_samples_per_class[_k] == intra_class_size

    # make the same size as the original
    _factor = n_samples_original // len(new_data)
    new_data *= _factor+1
    new_targets *= _factor+1

    new_data = new_data[:n_samples_original]
    new_targets = new_targets[:n_samples_original]

    # make the same type
    if isinstance(dataset.data, np.ndarray):
        new_data = np.array(new_data)

    new_targets = np.array(new_targets)

    dataset.data = new_data
    dataset.targets = new_targets


def make_analysis_dataloaders(trainloader, bankloader_ind, intra_class_size, class_label_scheme, label_noise_ratio):
    '''
    Modify dataloader for analysis
    '''
    if intra_class_size is not None:
        modify_dataset_intra_class_size(trainloader.dataset, intra_class_size)  # modify number of samples per each class

    if label_noise_ratio > 0:
        assert class_label_scheme in ['object']

    num_samples = trainloader.dataset.__len__()
    if class_label_scheme == 'instance':
        trainloader.dataset.targets = np.arange(0, num_samples, dtype=np.int64)
        num_classes = num_samples
    elif class_label_scheme == 'object':
        noised_targets = np.array(trainloader.dataset.targets)
        if label_noise_ratio > 0:
            np.random.shuffle(noised_targets[0:int(label_noise_ratio * num_samples)])
        trainloader.dataset.targets = np.array(noised_targets)
        num_classes = len(set(noised_targets))
    elif class_label_scheme == 'random_binary':
        noised_targets = np.random.choice([0, 1], num_samples)
        trainloader.dataset.targets = np.array(noised_targets, dtype=np.int64)
        num_classes = 2
    else:
        raise NotImplementedError()

    # Modify bankloder_ind accordingly
    bankloader_ind.dataset.data = trainloader.dataset.data
    bankloader_ind.dataset.targets = trainloader.dataset.targets

    return num_classes

if __name__ == '__main__':
    data_root_path = '/home/jay/savespace/database/generic_toy'


    # ### TEST 1 - START
    # task = 'osr'
    # data_name = 'tinyimagenet'
    # split_idx = 0
    # ind_data = get_dataset(data_root_path, task, data_name, split_idx, train=True, transform=None, ood=False)
    # modify_dataset_intra_class_size(ind_data, intra_class_size=10)
    # ### TEST 1 - END (debug success)

    # ### TEST 2 - START
    # task = 'ood'
    # data_name = 'cifar10'
    # split_idx = 10
    # ind_data = get_dataset(data_root_path, task, data_name, split_idx, train=True, transform=None, ood=False)
    # ### TEST 2 - END

    ### TEST 3 - START
    task = 'osr'
    data_name = 'mnist'
    split_idx = 0
    ind_data = get_dataset(data_root_path, task, data_name, split_idx, train=True, transform=None, ood=False)
    ### TEST 3 - END

    print()


