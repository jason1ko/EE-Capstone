import torch
import importlib
import torch.nn as nn
import numpy as np
import time

from tabulate import tabulate

from get_args_parser import get_args_parser
from engine_test import evaluate_p2p_cmc_cos_sim, evaluate_f2f_cmc_cos_sim, evaluate_p2f_cmc_cos_sim, evaluate_f2p_cmc_cos_sim
from engine_test import evaluate_p2f_cmc_kl_div, evaluate_f2p_cmc_kl_div
from engine_test import evaluate_p2p_eer, evaluate_f2f_eer, evaluate_p2f_eer_cos_sim, evaluate_f2p_eer_cos_sim
from engine_test import evaluate_p2f_eer_kl_div, evaluate_f2p_eer_kl_div
from dataset import get_identification_dataloaders, get_verification_dataloaders

from models.load_trained_models import load_trained_ckd
from misc.utils_torch import set_seed
from misc.utils_python import get_paths, import_yaml_config, generate_pairs


def test_p2p_identification(args, model):
    
    batch_size = 32

    model.eval()
    model.to(args.device)

    data_names = ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
    data_folds ={
        'ytf': ['gallery', 'probe'],
        'ethnic': ['gallery', 'probe'],
        'pubfig': ['gallery', 'probe1', 'probe2', 'probe3'],
        'facescrub': ['gallery', 'probe1', 'probe2'],
        'imdb_wiki': ['gallery', 'probe1', 'probe2', 'probe3'],
        'ar': ['gallery', 'blur', 'exp_illum', 'occlude', 'scarf']
    }

    cmc_dict = {}

    for data_name in data_names:
        print("The periocular-to-periocular CMC of {} start\n".format(data_name))
        _folds = data_folds[data_name]
        _pairs = generate_pairs(len(_folds))
        for n, (i, j) in enumerate(_pairs):
            _gallery_name = _folds[i]
            _probe_name = _folds[j]

            _gallery_loader, _probe_loader \
                = get_identification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers,
                                                 _gallery_name, _probe_name)

            #_cmc = evaluate_p2p_cmc_in_prod(args.device, model, _gallery_loader, _probe_loader)
            _cmc = evaluate_p2p_cmc_cos_sim(args.device, model, _gallery_loader, _probe_loader)

            if n==0:
                cmc = np.array(_cmc)
            else:
                cmc += np.array(_cmc)

            del _gallery_loader, _probe_loader
            torch.cuda.empty_cache()

        cmc /= len(_pairs)
        cmc_dict[data_name] = cmc
        print("\nThe periocular-to-periocular CMC of {} :\n{}\n".format(data_name, cmc))
    
    
    print("The periocular-to-periocular CMC dictionary : ")
    print(cmc_dict)


#my try
def test_f2f_identification(args, model): # periocular features only. peri-to-peri identification

    start = time.time()
    
    batch_size = 32

    model.eval()
    model.to(args.device)

    data_names = ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
    data_folds ={
        'ytf': ['gallery', 'probe'],
        'ethnic': ['gallery', 'probe'],
        'pubfig': ['gallery', 'probe1', 'probe2', 'probe3'],
        'facescrub': ['gallery', 'probe1', 'probe2'],
        'imdb_wiki': ['gallery', 'probe1', 'probe2', 'probe3'],
        'ar': ['gallery', 'blur', 'exp_illum', 'occlude', 'scarf']
    }

    cmc_dict = {}

    for data_name in data_names:
        print("The face-to-face CMC of {} start\n".format(data_name))
        _folds = data_folds[data_name] # ['gallery', 'probe1', ....]
        _pairs = generate_pairs(len(_folds)) # list of tuples. [(0,1), (0,2), ..., (1,0), (1,2), ...]
        for n, (i, j) in enumerate(_pairs):
            _gallery_name = _folds[i]
            _probe_name = _folds[j]

            _gallery_loader, _probe_loader \
                = get_identification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers,
                                                 _gallery_name, _probe_name)

            _cmc = evaluate_f2f_cmc_in_prod(args.device, model, _gallery_loader, _probe_loader)

            if n==0:
                cmc = np.array(_cmc)
            else:
                cmc += np.array(_cmc)

            del _gallery_loader, _probe_loader
            torch.cuda.empty_cache()

        cmc /= len(_pairs)
        cmc_dict[data_name] = cmc
        print("\nThe face-to-face CMC of {} :\n{}\n".format(data_name, cmc))

    print("The face-to-face CMC dictionary : ")
    print(cmc_dict)


#my try
def test_p2f_identification_cos_sim(args, model):

    batch_size = 32


    model.eval()
    model.to(args.device)

    data_names = ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
    data_folds ={
        'ytf': ['gallery', 'probe'],
        'ethnic': ['gallery', 'probe'],
        'pubfig': ['gallery', 'probe1', 'probe2', 'probe3'],
        'facescrub': ['gallery', 'probe1', 'probe2'],
        'imdb_wiki': ['gallery', 'probe1', 'probe2', 'probe3'],
        'ar': ['gallery', 'blur', 'exp_illum', 'occlude', 'scarf']
    }

    cmc_dict = {}
    start = time.time()
    
    for data_name in data_names:
        folder_start = time.time()
        print("The periocular-to-face CMC of {} start\n".format(data_name))
        _folds = data_folds[data_name] # ['gallery', 'probe1', ....]
        _pairs = generate_pairs(len(_folds)) # list of tuples. [(0,1), (0,2), ..., (1,0), (1,2), ...]
        for n, (i, j) in enumerate(_pairs):
            _gallery_name = _folds[i]
            _probe_name = _folds[j]

            _gallery_loader, _probe_loader \
                = get_identification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers,
                                                 _gallery_name, _probe_name)

            _cmc = evaluate_p2f_cmc_cos_sim(args.device, model, _gallery_loader, _probe_loader)

            if n==0:
                cmc = np.array(_cmc)
            else:
                cmc += np.array(_cmc)


            del _gallery_loader, _probe_loader

            torch.cuda.empty_cache()

        cmc /= len(_pairs)
        cmc_dict[data_name] = cmc
            
        print("\nThe periocular-to-face CMC of {} :\n{}".format(data_name, cmc))
        
        folder_finish = time.time() # sec
        folder_elapsed = round(folder_finish - folder_start)
        
        print("It took {}m {}s\n".format(folder_elapsed//60, folder_elapsed%60))

    print("The periocular-to-face CMC dictionary : ")
    print(cmc_dict)
    
    finish = time.time()
    elapsed = (finish - start) // 60 # min
    print("It took {}h {}m".format(elapsed//60, elapsed%60))
    
    
#my try
def test_p2f_identification_kl_div(args, model,T):

    batch_size = 8


    model.eval()
    model.to(args.device)

    data_names = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']
    data_folds ={
        'ytf': ['gallery', 'probe'],
        'ethnic': ['gallery', 'probe'],
        'pubfig': ['gallery', 'probe1', 'probe2', 'probe3'],
        'facescrub': ['gallery', 'probe1', 'probe2'],
        'imdb_wiki': ['gallery', 'probe1', 'probe2', 'probe3'],
        'ar': ['gallery', 'blur', 'exp_illum', 'occlude', 'scarf']
    }

    cmc_dict = {}
    start = time.time()
    
    for data_name in data_names:
        if data_name not in ['ethnic', 'ar']:
            continue
        folder_start = time.time()
        print("The periocular-to-face CMC of {} with KL Div start\n".format(data_name))
        _folds = data_folds[data_name] # ['gallery', 'probe1', ....]
        _pairs = generate_pairs(len(_folds)) # list of tuples. [(0,1), (0,2), ..., (1,0), (1,2), ...]
        for n, (i, j) in enumerate(_pairs):
            _gallery_name = _folds[i]
            _probe_name = _folds[j]

            _gallery_loader, _probe_loader \
                = get_identification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers,
                                                 _gallery_name, _probe_name)

            _cmc = evaluate_p2f_cmc_kl_div(args.device, model, _gallery_loader, _probe_loader,T, top_k=5)
            

            if n==0:
                cmc = np.array(_cmc)
            else:
                cmc += np.array(_cmc)


            del _gallery_loader, _probe_loader

            torch.cuda.empty_cache()

        cmc /= len(_pairs)
        cmc_dict[data_name] = cmc
            
        print("\nThe periocular-to-face CMC of {} with KL Div :\n{}".format(data_name, cmc))
        
        folder_finish = time.time() # sec
        folder_elapsed = round(folder_finish - folder_start)
        
        print("It took {}m {}s\n".format(folder_elapsed//60, folder_elapsed%60))

    print("The periocular-to-face CMC dictionary with KL Div : ")
    print(cmc_dict)
    
    finish = time.time()
    elapsed = (finish - start) // 60 # min
    print("It took {}h {}m".format(elapsed//60, elapsed%60))

    
    
#my try
def test_f2p_identification_cos_sim(args, model):

    batch_size = 32


    model.eval()
    model.to(args.device)

    data_names = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']
    data_folds ={
        'ytf': ['gallery', 'probe'],
        'ethnic': ['gallery', 'probe'],
        'pubfig': ['gallery', 'probe1', 'probe2', 'probe3'],
        'facescrub': ['gallery', 'probe1', 'probe2'],
        'imdb_wiki': ['gallery', 'probe1', 'probe2', 'probe3'],
        'ar': ['gallery', 'blur', 'exp_illum', 'occlude', 'scarf']
    }

    cmc_dict = {}
    start = time.time()
    
    for data_name in data_names:
        folder_start = time.time()
        print("The face-to-periocular CMC of {} start\n".format(data_name))
        _folds = data_folds[data_name] # ['gallery', 'probe1', ....]
        _pairs = generate_pairs(len(_folds)) # list of tuples. [(0,1), (0,2), ..., (1,0), (1,2), ...]
        for n, (i, j) in enumerate(_pairs):
            _gallery_name = _folds[i]
            _probe_name = _folds[j]

            _gallery_loader, _probe_loader \
                = get_identification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers,
                                                 _gallery_name, _probe_name)

            _cmc = evaluate_f2p_cmc_cos_sim(args.device, model, _gallery_loader, _probe_loader)

            if n==0:
                cmc = np.array(_cmc)
            else:
                cmc += np.array(_cmc)


            del _gallery_loader, _probe_loader

            torch.cuda.empty_cache()

        cmc /= len(_pairs)
        cmc_dict[data_name] = cmc
            
        print("\nThe face-to-periocular CMC of {} :\n{}".format(data_name, cmc))
        
        folder_finish = time.time() # sec
        folder_elapsed = round(folder_finish - folder_start)
        
        print("It took {}m {}s\n".format(folder_elapsed//60, folder_elapsed%60))

    print("The face-to-periocular CMC dictionary : ")
    print(cmc_dict)
    
    finish = time.time()
    elapsed = (finish - start) // 60 # min
    print("It took {}h {}m".format(elapsed//60, elapsed%60))


    
    
def test_f2p_identification_kl_div(args, model,T):

    batch_size = 8


    model.eval()
    model.to(args.device)

    data_names = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']
    data_folds ={
        'ytf': ['gallery', 'probe'],
        'ethnic': ['gallery', 'probe'],
        'pubfig': ['gallery', 'probe1', 'probe2', 'probe3'],
        'facescrub': ['gallery', 'probe1', 'probe2'],
        'imdb_wiki': ['gallery', 'probe1', 'probe2', 'probe3'],
        'ar': ['gallery', 'blur', 'exp_illum', 'occlude', 'scarf']
    }

    cmc_dict = {}
    start = time.time()
    
    for data_name in data_names:
        if data_name not in ['ethnic', 'ar']:
            continue
        folder_start = time.time()
        print("The face-to-periocular CMC of {} with KL Div start\n".format(data_name))
        _folds = data_folds[data_name] # ['gallery', 'probe1', ....]
        _pairs = generate_pairs(len(_folds)) # list of tuples. [(0,1), (0,2), ..., (1,0), (1,2), ...]
        for n, (i, j) in enumerate(_pairs):
            _gallery_name = _folds[i]
            _probe_name = _folds[j]

            _gallery_loader, _probe_loader \
                = get_identification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers,
                                                 _gallery_name, _probe_name)

            _cmc = evaluate_f2p_cmc_kl_div(args.device, model, _gallery_loader, _probe_loader,T, top_k=5)
            

            if n==0:
                cmc = np.array(_cmc)
            else:
                cmc += np.array(_cmc)


            del _gallery_loader, _probe_loader

            torch.cuda.empty_cache()

        cmc /= len(_pairs)
        cmc_dict[data_name] = cmc
            
        print("\nThe face-to-periocular CMC of {} with KL Div :\n{}".format(data_name, cmc))
        
        folder_finish = time.time() # sec
        folder_elapsed = round(folder_finish - folder_start)
        
        print("It took {}m {}s\n".format(folder_elapsed//60, folder_elapsed%60))

    print("The face-to-periocular CMC dictionary with KL Div : ")
    print(cmc_dict)
    
    finish = time.time()
    elapsed = (finish - start) // 60 # min
    print("It took {}h {}m".format(elapsed//60, elapsed%60))

    
    
    
##########################   verification #############################
# my try
def test_p2p_verification(args, model):
    
    batch_size = 32
    
    model.eval()
    model.to(args.device)
    
    data_names = ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
    
    eer_dict = {}
    
    for data_name in data_names:
        print('The periocular-to-periocular EER of {} start\n'.format(data_name))
        dloader = get_verification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers)
        
        eer = evaluate_p2p_eer(args.device, model, dloader)
        eer_dict[data_name] = eer
        print("\nThe periocular-to-periocular EER of {} : {}\n".format(data_name, eer))
        
        del dloader
        torch.cuda.empty_cache()
        
    print("The periocular-to-periocular EER dictionary : ")
    print(eer_dict)


# my try
def test_f2f_verification(args, model):
    
    batch_size = 32
    
    model.eval()
    model.to(args.device)
    
    data_names = ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
    
    eer_dict = {}
    
    for data_name in data_names:
        print('The face-to-face EER of {} start\n'.format(data_name))
        dloader = get_verification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers)
        
        eer = evaluate_f2f_eer(args.device, model, dloader)
        eer_dict[data_name] = eer
        print("\nThe face-to-face EER of {} : {}\n".format(data_name, eer))
        
        del dloader
        torch.cuda.empty_cache()
        
    print("The face-to-face EER dictionary : ")
    print(eer_dict)



# my try
def test_p2f_verification_cos_sim(args, model):
    
    batch_size = 32
    
    model.eval()
    model.to(args.device)
    
    data_names = ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
    
    eer_dict = {}
    
    for data_name in data_names:
        
        print('The periocular-to-face EER of {} start\n'.format(data_name))
        dloader = get_verification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers)
        
        eer = evaluate_p2f_eer_cos_sim(args.device, model, dloader)
        eer_dict[data_name] = eer
        print("\nThe periocular-to-face EER of {} : {}\n".format(data_name, eer))
        
        del dloader
        torch.cuda.empty_cache()
        
    print("The periocular-to-face EER dictionary : ")
    print(eer_dict)

        
        
# my try
def test_p2f_verification_kl_div(args, model, T):
    
    batch_size = 8
    
    model.eval()
    model.to(args.device)
    
    data_names = ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
    
    eer_dict = {}
    
    for data_name in data_names:
        if data_name not in ['ethnic', 'ar']:
            continue
        
        print('The periocular-to-face EER of {} with KL divergence start'.format(data_name))
        dloader = get_verification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers)
        
        eer = evaluate_p2f_eer_kl_div(args.device, model, dloader, T)
        eer_dict[data_name] = eer
        print("\nThe periocular-to-face EER of {} with KL divergence: {}\n".format(data_name, eer))
        
        del dloader
        torch.cuda.empty_cache()
        
    print("The periocular-to-face EER dictionary with KL divergence : ")
    print(eer_dict)        
        


def test_f2p_verification_cos_sim(args, model):
    
    batch_size = 32
    
    model.eval()
    model.to(args.device)
    
    data_names = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']
    
    eer_dict = {}
    
    for data_name in data_names:
        
        print('The face-to-periocular EER of {} start\n'.format(data_name))
        dloader = get_verification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers)
        
        eer = evaluate_f2p_eer_cos_sim(args.device, model, dloader)
        eer_dict[data_name] = eer
        print("\nThe face-to-periocular EER of {} : {}\n".format(data_name, eer))
        
        del dloader
        torch.cuda.empty_cache()
        
    print("The face-to-periocular EER dictionary : ")
    print(eer_dict)        

    
    
def test_f2p_verification_kl_div(args, model, T):
    
    batch_size = 8
    
    model.eval()
    model.to(args.device)
    
    data_names = ['ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar', 'ytf']
    
    eer_dict = {}
    
    for data_name in data_names:
        if data_name not in ['ethnic', 'ar']:
            continue
        
        print('The face-to-periocular EER of {} with KL divergence start'.format(data_name))
        dloader = get_verification_dataloaders(args.data_root_path, data_name, batch_size, args.num_workers)
        
        eer = evaluate_f2p_eer_kl_div(args.device, model, dloader, T)
        eer_dict[data_name] = eer
        print("\nface-to-periocular EER of {} with KL divergence: {}\n".format(data_name, eer))
        
        del dloader
        torch.cuda.empty_cache()
        
    print("The face-to-periocular EER dictionary with KL divergence : ")
    print(eer_dict)      
    
#####################################
def main():
    parser = get_args_parser()
    args = parser.parse_args('')
    args = import_yaml_config(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # args.device = torch.device('cpu')

    if args.seed != None:
        set_seed(0)
    args = get_paths(args)

    print(tabulate(list(vars(args).items()), headers=['arguments', 'values']))

    model_modules = importlib.import_module('models.{}'.format(args.model_name))

    args.num_classes = 1054

    model = model_modules.make_model(args)
    model = model.to(args.device)

    '''
    load pretrain
    '''
    load_trained_ckd(model, f"{args.save_dir_path}/model_best.pth" )


    test_p2p_identification(args, model)
    test_p2p_verification(args, model)

    pass


if __name__ == '__main__':
    main()