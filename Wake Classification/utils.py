import torch
import torchvision
from dataset import ShipDataset
from torch.utils.data import DataLoader
import json
import os
import numpy as np

def save_checkpoint(state, savename="my_checkpoint.pth.tar"):
    print("=> saving checkpoint")
    torch.save(state, savename)

def save_dictionary(dict, savename="my_dictionary.json"):
    print("=> saving dictionary")
    jsonObj = json.dumps(dict)
    f = open(savename,"w")
    f.write(jsonObj)
    f.close()

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    print('load complete')

def get_loaders(
        train_img_files,
        valid_img_files,
        test_img_files,
        batch_size,
        train_transform,
        valid_transform,
        test_transform,
        num_workers=4,
        pin_memory=True,
):

    # TRAIN 
    train_ds = ShipDataset(
        image_files=train_img_files,
        transform=train_transform,
    )

    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )


    # VALIDATION
    valid_ds = ShipDataset(
        image_files=valid_img_files,
        transform=valid_transform,
    )

    
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )


    # TESTING
    test_ds = ShipDataset(
        image_files=test_img_files,
        transform=test_transform,
    )

 
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    
    return train_loader, valid_loader, test_loader



def dir2files(image_dirs):

    if isinstance(image_dirs, list): # in-case multi folder is supplied
        image_files = []
        for i in range(len(image_dirs)):
            ifiles = os.listdir(image_dirs[i])
            for ii in range(len(ifiles)):
                ifiles[ii]=os.path.join(image_dirs[i], ifiles[ii])

            image_files +=ifiles

    else:

        image_files = os.listdir(image_dirs)
        for ii in range(len(image_files)):
            image_files[ii]=os.path.join(image_dirs, image_files[ii])

    return image_files


def get_splitted_files(image_dir = None,
                        train_ratio = 0.8,
                        valid_ratio = 0.1,
                        test_ratio = 0.1,
                        ):

    folders = [ os.path.join(image_dir, item) for item in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, item)) ]
    nfolder = len(folders)

    train_files = []
    valid_files = []
    test_files  = []

    if nfolder == 0:

        all_files =[]
        for root, _, files in os.walk(image_dir):
            for name in files:
                all_files.append(os.path.join(root,name))

        NFiles = len(all_files)

        idx         = np.random.permutation(NFiles)
        n_valid     = int(valid_ratio*NFiles)
        n_test      = int(test_ratio*NFiles)
        n_train     = NFiles - (n_valid + n_test)

        train_idx   = idx[0:n_train]
        valid_idx   = idx[n_train:n_train+n_valid]
        test_idx    = idx[n_train+n_valid:]

        for idx in train_idx:
            train_files.append(all_files[idx])

        for idx in valid_idx:
            valid_files.append(all_files[idx])

        for idx in test_idx:
            test_files.append(all_files[idx]) 



    else: #if path contains multiple folders
        for subfolder in (folders) :
            all_files =[]
            for root, _, files in os.walk(subfolder):
                for name in files:
                    all_files.append(os.path.join(root,name))

            NFiles = len(all_files)

            idx         = np.random.permutation(NFiles)
            n_valid     = int(valid_ratio*NFiles)
            n_test      = int(test_ratio*NFiles)
            n_train     = NFiles - (n_valid + n_test)

            train_idx   = idx[0:n_train]
            valid_idx   = idx[n_train:n_train+n_valid]
            test_idx    = idx[n_train+n_valid:]

            for idx in train_idx:
                train_files.append(all_files[idx])

            for idx in valid_idx:
                valid_files.append(all_files[idx])

            for idx in test_idx:
                test_files.append(all_files[idx])                



    #return nfolder
    return train_files, valid_files, test_files
