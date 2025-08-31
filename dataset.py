from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm

import time

class DatasetTrain(Dataset):
    def __init__(self, data_root_path, fold='train'):
        assert fold in ['train', 'val', 'test']

        dataset_path = f'{data_root_path}/trainingdb/{fold}'

        self.samples_face = []
        self.samples_ocular = []
        self.labels = []

        y = 0
        for id in natsorted(os.listdir(dataset_path)):

            face_id_dir = f'{dataset_path}/{id}/face'
            ocular_id_dir = f'{dataset_path}/{id}/periocular'

            for sample in natsorted(os.listdir(face_id_dir)):
                self.samples_face.append(face_id_dir + '/' + sample)
                self.labels.append(y)

            for sample in natsorted(os.listdir(ocular_id_dir)):
                self.samples_ocular.append(ocular_id_dir + '/' + sample)

            y += 1

        self.labels = np.array(self.labels, dtype=np.int64)

        if fold == 'train':
            self.transform_face = transforms.Compose([transforms.Resize(128),
                                                      transforms.RandomHorizontalFlip(),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                      ])

            self.transform_ocular = transforms.Compose([transforms.Resize((48, 128)),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                        ])
        elif fold in ['val', 'test']:
            self.transform_face = transforms.Compose([transforms.Resize(128),
                                                      transforms.ToTensor(),
                                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                      ])

            self.transform_ocular = transforms.Compose([transforms.Resize((48, 128)),
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                        ])
        else:
            raise ValueError()


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample_ocular, sample_face, label = self.samples_ocular[idx], self.samples_face[idx], self.labels[idx]
        sample_ocular, sample_face = Image.open(sample_ocular), Image.open(sample_face)
        sample_ocular, sample_face = self.transform_ocular(sample_ocular), self.transform_face(sample_face)

        return sample_ocular, sample_face, label


class DatasetIdentification(Dataset):
    def __init__(self, data_root_path, data_name, fold): #fold : whether gallery or probe
        assert data_name in ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
        if data_name == 'ethnic':
            dataset_path = f'{data_root_path}/{data_name}/Recognition/{fold}'
        else:
            dataset_path = f'{data_root_path}/{data_name}/{fold}'
            
        self.data_name = data_name



        self.samples_face = [] # to save the directory of each face picture of all people
        self.samples_ocular = []
        self.labels = []

        self.names = [] #my try
        self.idxes = [] # my try
        self.name_to_idx = {} # my try

        y = 0
        for id in natsorted(os.listdir(dataset_path)):

            self.idxes.append(y) #my try
            self.names.append(id) #my try
            self.name_to_idx[id] = y #my try

            face_id_dir = f'{dataset_path}/{id}/face'
            ocular_id_dir = f'{dataset_path}/{id}/periocular'

            for sample in natsorted(os.listdir(face_id_dir)):
                self.samples_face.append(face_id_dir + '/' + sample)
                self.labels.append(y)

            for sample in natsorted(os.listdir(ocular_id_dir)):
                self.samples_ocular.append(ocular_id_dir + '/' + sample)

            y += 1

        self.labels = np.array(self.labels, dtype=np.int64)

        self.transform_face = transforms.Compose([transforms.Resize(128),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                  ])

        self.transform_ocular = transforms.Compose([transforms.Resize((48, 128)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                    ])
        # my try


        # my try
        if self.data_name == 'ytf':
            
            self.ytf_tensors_ocular = [] # save torch value of pictures by transforms
            self.ytf_tensors_face = []
            self.ytf_labels = []

            y = 0
            temp = 0
            
            print("ytf dataloading takes some time, please wait")
            
            for name in self.names:
                face_list_dir = f'{dataset_path}/{name}/face'
                num_of_pictures = len(os.listdir(face_list_dir))
                
                quotient = (num_of_pictures // 10) + 1
                remainder = num_of_pictures % 10
                
                for i in range(quotient):
                    if i == quotient-1 and remainder == 0: # when the num_of_pictures is multiple of 10, finish
                        break # stop the "for i in range(quotient)" loop
                        
                    ytf_ocular = torch.zeros(3,48,128)
                    ytf_face = torch.zeros(3,128,128)
                    
                    for j in range(10): # add 10 pictures
                        sample_ocular = self.samples_ocular[temp + 10*i + j]
                        sample_face = self.samples_face[temp + 10*i + j]
                        
                        sample_ocular = Image.open(sample_ocular)
                        sample_face = Image.open(sample_face)
                        
                        sample_ocular = self.transform_ocular(sample_ocular)
                        sample_face = self.transform_face(sample_face)
                        
                        ytf_ocular += sample_ocular
                        ytf_face += sample_face
                        
                        if i == quotient-1 and remainder == j+1:
                            break
                            
                    if i == quotient-1 and remainder == j+1:
                        ytf_ocular = ytf_ocular / remainder
                        ytf_face = ytf_face / remainder
                        
                        self.ytf_tensors_ocular.append(ytf_ocular)
                        self.ytf_tensors_face.append(ytf_face)
                        self.ytf_labels.append(y)
                        
                    else:
                        ytf_ocular = ytf_ocular / 10
                        ytf_face = ytf_face / 10
                        
                        self.ytf_tensors_ocular.append(ytf_ocular)
                        self.ytf_tensors_face.append(ytf_face)
                        self.ytf_labels.append(y)
                                      
#               remainder = num_of_pictures % 10
                y += 1 # in for loop of self.names
                temp += num_of_pictures
        

#           y = 0
        
        
        
#    def __len__(self):
#        return len(self.labels)

    def __len__(self):
        if self.data_name == 'ytf':
            return len(self.ytf_labels)
        else:
            return len(self.labels)

#    def __getitem__(self, idx): # if dataname is 'ytf' or not...
#        sample_ocular, sample_face, label = self.samples_ocular[idx], self.samples_face[idx], self.labels[idx] # one directory, str type
#        sample_ocular, sample_face = Image.open(sample_ocular), Image.open(sample_face) # change to image
#        sample_ocular, sample_face = self.transform_ocular(sample_ocular), self.transform_face(sample_face) # change to torch.tensor

#        return sample_ocular, sample_face, label


    def __getitem__(self, idx):
        if self.data_name == 'ytf':
            sample_ocular, sample_face, label = self.ytf_tensors_ocular[idx], self.ytf_tensors_face[idx], self.ytf_labels[idx]
            
            return sample_ocular, sample_face, label
        else:
            sample_ocular, sample_face, label = self.samples_ocular[idx], self.samples_face[idx], self.labels[idx] # one directory, str type
            sample_ocular, sample_face = Image.open(sample_ocular), Image.open(sample_face) # change to image
            sample_ocular, sample_face = self.transform_ocular(sample_ocular), self.transform_face(sample_face) # change to torch.tensor
            
            return sample_ocular, sample_face, label
            


class DatasetVerification(Dataset):
    def __init__(self, data_root_path, data_name, fold):

        assert data_name in ['ytf', 'ethnic', 'pubfig', 'facescrub', 'imdb_wiki', 'ar']
        if data_name == 'ethnic':
            dataset_path = f'{data_root_path}/{data_name}/Recognition/{fold}' # why not 'Verification' ??
        else:
            dataset_path = f'{data_root_path}/{data_name}/{fold}'

        self.samples_face = [] # save 4 pictures of each person
        self.samples_ocular = []
        self.labels = []
        self.num_identity = len(os.listdir(dataset_path)) # added

        num_images_per_id = 4
        y = 0
        for id in natsorted(os.listdir(dataset_path)):
            ocular_id_dir_path = f'{dataset_path}/{id}/periocular'
            face_id_dir_path = f'{dataset_path}/{id}/face'
            ocular_id_dir = sorted(os.listdir(ocular_id_dir_path))  # this should be 'sorted' not 'natsorted'
            face_id_dir = sorted(os.listdir(face_id_dir_path))      # this should be 'sorted' not 'natsorted'
            offset = len(ocular_id_dir) // num_images_per_id
            for i in range(num_images_per_id): # totally 4 iterations
                _ocular_sample_path = ocular_id_dir[offset*i]
                _face_sample_path = face_id_dir[offset*i]
                assert _ocular_sample_path == _face_sample_path
                # save 4 pictures of one person in samples_ list
                self.samples_ocular.append(f'{ocular_id_dir_path}/{_ocular_sample_path}')
                self.samples_face.append(f'{face_id_dir_path}/{_face_sample_path}')
                self.labels.append(y) # [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
            y += 1

        self.labels = np.array(self.labels, dtype=np.int64)

        #self.onehot_label = np.zeros((len(self.labels), self.num_identity)) # 4 * 1
        #for i in range(len(self.labels)):
          #self.onehot_label[i, self.labels[i]] = 1

        self.transform_face = transforms.Compose([transforms.Resize(128),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                  ])

        self.transform_ocular = transforms.Compose([transforms.Resize((48, 128)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                                    ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample_ocular, sample_face, label = self.samples_ocular[idx], self.samples_face[idx], self.labels[idx]
        sample_ocular, sample_face = Image.open(sample_ocular), Image.open(sample_face)
        sample_ocular, sample_face = self.transform_ocular(sample_ocular), self.transform_face(sample_face)

        return sample_ocular, sample_face, label


def get_train_dataloaders(data_root_path, batch_size, num_workers):
    train_dataset = DatasetTrain(data_root_path, fold='train')
    val_dataset = DatasetTrain(data_root_path, fold='val')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader


def get_identification_dataloaders(data_root_path, data_name, batch_size, num_workers, gallery_fold, probe_fold):
    assert gallery_fold != probe_fold

    start = time.time()
    
    gallery_dataset = DatasetIdentification(data_root_path, data_name, fold=gallery_fold)
    probe_dataset = DatasetIdentification(data_root_path, data_name, fold=probe_fold)
    gallery_dataloader = DataLoader(gallery_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    probe_dataloader = DataLoader(probe_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    elapsed = time.time() - start
    if data_name == 'ytf':
        print('dataloader time elapsed : {}m {}s\n'.format(int(elapsed // 60), round(elapsed % 60, 2)))

    return gallery_dataloader, probe_dataloader


# my try
def get_verification_dataloaders(data_root_path, data_name, batch_size, num_workers):
    if data_name == 'ethnic':
        _dataset = DatasetVerification(data_root_path, data_name, fold='probe')
    else:
        _dataset = DatasetVerification(data_root_path, data_name, fold='gallery')
        
    _dataloader = DataLoader(_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return _dataloader  

if __name__ == '__main__':
    data_root_path = '/home/jay/savespace/database/periocular'
    # train_dataset = DatasetTrain(data_root_path, fold='train')
    verify_dataset = DatasetVerification(data_root_path, 'pubfig', 'gallery')

    print()

    _dataset = verification_dataset('pubfig')



    print()
    


            