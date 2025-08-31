import numpy as np


CIFAR100_SUPERCLASS_OLD = [     # from CSI github: https://github.com/alinlab/CSI/blob/master/datasets/datasets.py
    [4, 31, 55, 72, 95],
    [1, 33, 67, 73, 91],
    [54, 62, 70, 82, 92],
    [9, 10, 16, 29, 61],
    [0, 51, 53, 57, 83],
    [22, 25, 40, 86, 87],
    [5, 20, 26, 84, 94],
    [6, 7, 14, 18, 24],
    [3, 42, 43, 88, 97],
    [12, 17, 38, 68, 76],
    [23, 34, 49, 60, 71],
    [15, 19, 21, 32, 39],
    [35, 63, 64, 66, 75],
    [27, 45, 77, 79, 99],
    [2, 11, 36, 46, 98],
    [28, 30, 44, 78, 93],
    [37, 50, 65, 74, 80],
    [47, 52, 56, 59, 96],
    [8, 13, 48, 58, 90],
    [41, 69, 81, 85, 89],
]

CIFAR100_SUPERCLASS_STR = [
    ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    ['bottle', 'bowl', 'can', 'cup', 'plate'],
    ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    ['crab', 'lobster', 'snail', 'spider', 'worm'],
    ['baby', 'boy', 'girl', 'man', 'woman'],
    ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
]

def get_cifar100_superclasses(classes):
    sup_class_list = [[]]*20
    for i, sub_class_list in enumerate(CIFAR100_SUPERCLASS_STR):
        tmp_list = []
        for sub_class in sub_class_list:
            tmp_list.append(classes.index(sub_class))

        sup_class_list[i] = tmp_list

    return sup_class_list

def label_cifar100_by_superclasses(labels, classes):
    sup_class_list = get_cifar100_superclasses(classes)         # TODO: select this if you want a correct version
    # sup_class_list = CIFAR100_SUPERCLASS_OLD                    # TODO: select this if you want the same environment as CSI
    labels_sup = -np.ones_like(labels)
    for i in range(len(sup_class_list)):
        idxes = np.in1d(labels, sup_class_list[i]).nonzero()[0]
        labels_sup[idxes] = i
    assert -1 not in labels_sup

    return labels_sup