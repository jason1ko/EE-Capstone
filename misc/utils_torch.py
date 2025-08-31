import os
import json
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import getpass
import torch
import random



'''
Torch Utils
'''
def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def iter2epoch(iter_total, batch_size, num_samples):
    iter_per_epoch = num_samples/batch_size
    num_epochs = iter_total / iter_per_epoch
    return num_epochs

def save_full_model(model, optimizer, scheduler, model_path):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }, model_path)

def load_full_model(device, model, optimizer, scheduler, model_path):
    check_point = torch.load(model_path, map_location='cuda:%d' % device.index)
    model.load_state_dict(check_point['model_state_dict'], strict=False)
    optimizer.load_state_dict(check_point['optimizer_state_dict'])
    scheduler.load_state_dict(check_point['scheduler_state_dict'])
    return model, optimizer, scheduler

def initialize_model_toy(args, model_modules, num_supervised_classes):
    if args.model_name == 'moco':
        model = model_modules.MoCo(args.device, args.model_config,
                                   backbone_name=args.backbone_name)
    elif args.model_name == 'ce':
        model = model_modules.CE(args.device, num_supervised_classes, args.model_config,
                                 backbone_name=args.backbone_name)
    elif args.model_name == 'recos':
        model = model_modules.ReCos(args.device, num_supervised_classes, args.model_config,
                                    backbone_name=args.backbone_name)
    elif args.model_name == 'maxjnd':
        model = model_modules.MaxJND(args.device, num_supervised_classes, args.model_config,
                                     backbone_name=args.backbone_name)
    else:
        raise NotImplementedError()
    return model

def set_seed(seed_number):
    random.seed(seed_number)
    np.random.seed(seed_number)
    torch.manual_seed(seed_number)
    torch.backends.cudnn.deterministic = True


