import os
import json
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import getpass
import yaml
from datetime import datetime


'''
Result Save Utils
'''
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# def import_json_config(args):
#     # args: namespace instance
#     with open('./configs/{}.json'.format(args.config_name)) as f:
#         conf_json = json.load(f)
#
#     for k, v in conf_json.items():
#         args.__setattr__(k, v)
#
#     return args

def import_yaml_config(args):
    # args: namespace instance
    with open("./configs/{}.yaml".format(args.config_name), 'r') as stream:
        conf_yaml = yaml.safe_load(stream)

    for k, v in conf_yaml.items():
        args.__setattr__(k, v)

    return args

def dict2csv(dict, filename):
    ptable = dict2ptable(dict)
    ptable2csv(ptable, filename)

def dict2ptable(dict):
    # dict contains scalar values as its elements
    table_header = list(dict.keys())
    result_table = PrettyTable(table_header, align='r', float_format='.4')
    result_table.add_row(list(dict.values()))
    return result_table

def ptable2csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))

def get_paths(args):

    args.save_dir_path = f'{args.save_root_path}/{args.config_name}'
    args.log_dir_path = f'./logs/result_log/{args.config_name}'
    now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    args.tb_dir_path = f'./logs/tb_log/{args.config_name}/{now_time}'

    mkdir(args.save_dir_path)
    mkdir(args.log_dir_path)
    mkdir(args.tb_dir_path)

    return args


'''
Tensorbord utils
'''
def write_on_tb(tb_writer, value_dict, epoch, header=None):
    for key in value_dict.keys():

        if header is not None:
            _key = header + '/' + key
        else:
            _key = key

        tb_writer.add_scalar(_key, value_dict[key], epoch)

        # tb_writer.add_histogram(key, value_dict[key], epoch)

'''
misc
'''
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def generate_pairs(n):
    pairs = []
    for i in range(n):
        for j in range(n):
            if i==j:
                pass
            else:
                pairs.append((i,j))

    return pairs


#
# TODO: test the method visualize_nodes
#
# if __name__ == '__main__':
#     fea = np.random.randn(1,32)
#     visualize_fea_nodes(fea, sort=True)

if __name__ == '__main__':
    pairs = generate_pairs(4)
    print(pairs)

