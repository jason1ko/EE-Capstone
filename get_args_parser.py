import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-cn', default='ckd_config01', type=str,
                        help='config name')  # TODO: Set config name

    parser.add_argument('--gpu_num', '-g', default=1, type=int,
                        help='gpu number')  # TODO: Set GPU number
    parser.add_argument('--num_workers', '-nw', default=8, type=int,
                        help='number of workers')  # TODO: Set GPU number (e.g. 8, 16)

    parser.add_argument('--tb_log', '-tl', default=False, type=bool, help='log by tensorboard')
    parser.add_argument('--seed', '-sd', default=None, type=int, help='random seed number')

    parser.add_argument('--data_root_path', '-drp',
                        default='C:\\Users\\chris\\image_datasets',
                        type=str, help='data root path')
    parser.add_argument('--save_root_path', '-srp', default='./saved_models', type=str,
                        help='save root path')

    return parser