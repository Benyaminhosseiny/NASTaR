import sys
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate


MAIN_FOLDER = '/user/work/xo23898/NovaSAR/IGARSS_2023/Classification/AlexNet'
sys.path.insert(0,MAIN_FOLDER)

from model import (AlexNet)
from helper_evaluate import compute_accuracy, mean_std, compute_accuracy_single_batch
from utils import (get_loaders)
from helper_train import train_classifier_simple_v1
import time
from utils import*
import json
from datetime import datetime

import psutil
import platform


BASE_NUM_CLASSES    = 10
NEW_NUM_CLASSES     = 2
BATCH_SIZE          = 64 
imgSize             = 227 
NUM_EPOCHS          = 50
N_TRIALS            = 30
NUM_WORKERS         = 8

Exp_Type            = 2


NVReal_Percentage_Train     = 30 #portion of nvreal included into the training process
NVSim_train_ratio           = 0.8
NVSim_valid_ratio           = 0.1
NVSim_test_ratio            = 0.1

DEVICE              = 'cpu' 


NVSim_path      = r'/user/work/xo23898/NovaSAR/IGARSS_2023/Classification/NVSim_Narrow_Dataset/NF/'
NVReal_path     = r'/user/work/xo23898/NovaSAR/20231004/Dataset/NVReal_Rotated_22dot5_augmented/'
base_model_path = r'/user/work/xo23898/NovaSAR/IGARSS_2023/Classification/AlexNet/BO_Output_Base_Model_20231219_024008/optimum_model_checkpoint.pth.tar'



if NVReal_Percentage_Train == 0:
    NVReal_train_ratio          = 0.8
    NVReal_valid_ratio          = 0.1
    NVReal_test_ratio           = 0.1
else:
    NVReal_test_ratio           = 1-(NVReal_Percentage_Train/100.0)
    NVReal_train_ratio          = (1-NVReal_test_ratio) * (NVSim_train_ratio+NVSim_valid_ratio)
    NVReal_valid_ratio          = (1-NVReal_test_ratio) * NVSim_test_ratio


def train_evaluate(parameterization):

    if Exp_Type != 3:
        model    = AlexNet(BASE_NUM_CLASSES)
        load_checkpoint(torch.load(base_model_path), model)
        
        if Exp_Type == 2: #FT
            for param in model.parameters():
                param.requires_grad = True

        if Exp_Type == 1: #FE
            for param in model.parameters():
                param.requires_grad = False

        model.classifier[6] = nn.Linear(4096, NEW_NUM_CLASSES)

    else:
        model    = AlexNet(NEW_NUM_CLASSES)


    loss_fn             = nn.CrossEntropyLoss()
    lr                  = parameterization.get("lr")
    momentum            = parameterization.get("momentum")
    weight_decay        = parameterization.get("weight_decay")


    optimizer           = torch.optim.SGD(model.parameters(),
                                lr              = lr,
                                momentum        = momentum,
                                weight_decay    = weight_decay)

    _,trained_model     = train_classifier_simple_v1(
                                num_epochs          = NUM_EPOCHS,
                                model               = model,
                                optimizer           = optimizer,
                                loss_fn             = loss_fn,
                                train_loader        = train_loader,
                                valid_loader        = None,
                                test_loader         = None,
                                skip_epoch_stats    = True,
                                save_model          = False,
                                skip_progress       = True,
                                device              = DEVICE,
                                )


    acc     = compute_accuracy(model=trained_model, data_loader=valid_loader,device=DEVICE)
    acc     = float(acc)

    return acc


def main ():

    global train_loader, valid_loader, test_loader

    # =============== OUTPUT FOLDER NAME
    start_time          = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_folder       = 'BO_Exp' + str(Exp_Type) + '_Portion_' +str(NVReal_Percentage_Train) +'_'+ start_time
        
    output_folder       = os.path.join(MAIN_FOLDER, output_folder)


    NVReal_train_files, NVReal_valid_files, NVReal_test_files   = get_splitted_files(image_dir = NVReal_path,
                                                                train_ratio = NVReal_train_ratio,
                                                                valid_ratio = NVReal_valid_ratio,
                                                                test_ratio  = NVReal_test_ratio,
                                                                )

    NVSim_train_files, NVSim_valid_files, NVSim_test_files      = get_splitted_files(image_dir = NVSim_path,
                                                                train_ratio = NVSim_train_ratio,
                                                                valid_ratio = NVSim_valid_ratio,
                                                                test_ratio  = NVSim_test_ratio,
                                                                )

    if NVReal_Percentage_Train == 0:
        train_files = NVSim_train_files + NVSim_valid_files 
        valid_files = NVSim_test_files
        test_files  = NVReal_train_files  + NVReal_valid_files + NVReal_test_files

    else:
        train_files = NVSim_train_files + NVSim_valid_files + NVReal_train_files
        valid_files = NVSim_test_files + NVReal_valid_files
        test_files  = NVReal_test_files


    train_transforms0  = transforms.Compose([transforms.Resize((imgSize, imgSize)),
                                                           transforms.ToTensor(),
                                                           ])
    valid_transforms0  = transforms.Compose([transforms.Resize((imgSize, imgSize)),
                                                           transforms.ToTensor(),
                                                           ])
    test_transforms0   = transforms.Compose([transforms.Resize((imgSize, imgSize)),
                                                           transforms.ToTensor(),
                                                           ])

    train_loader0, valid_loader0, test_loader0 = get_loaders(
                        train_img_files=train_files,
                        valid_img_files=valid_files,
                        test_img_files=test_files,
                        batch_size=BATCH_SIZE,
                        train_transform=train_transforms0,
                        valid_transform=valid_transforms0,
                        test_transform=test_transforms0,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
                )

    [mean_train, std_train]     = mean_std(train_loader0)
    [mean_valid, std_valid]     = mean_std(valid_loader0)
    [mean_test, std_test]       = mean_std(test_loader0)

    print(mean_test, std_test)
    quit()

    train_transforms  = transforms.Compose([transforms.Resize((imgSize, imgSize)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=(mean_train), std=(std_train)),
                                                           transforms.RandomApply([transforms.RandomRotation([-5, 5])], p=1.0),
                                                           ])
    valid_transforms  = transforms.Compose([transforms.Resize((imgSize, imgSize)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=(mean_valid), std=(std_valid)),
                                                           #transforms.RandomApply([transforms.RandomRotation([-5, 5])], p=1.0),
                                                           ])
    test_transforms   = transforms.Compose([transforms.Resize((imgSize, imgSize)),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=(mean_test), std=(std_test)),
                                                           #transforms.RandomApply([transforms.RandomRotation([-5, 5])], p=1.0), 
                                                         ])
                
    train_loader, valid_loader, test_loader = get_loaders(
                        train_img_files=train_files,
                        valid_img_files=valid_files,
                        test_img_files=test_files,
                        batch_size=BATCH_SIZE,
                        train_transform=train_transforms,
                        valid_transform=valid_transforms,
                        test_transform=test_transforms,
                        num_workers=NUM_WORKERS,
                        pin_memory=True,
        )

    

    TIC = time.time()
    best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-5, 0.5], "log_scale": True, "value_type": "float"},
        {"name": "momentum", "type": "range", "bounds": [0.1, 0.98], "log_scale": False, "value_type": "float"},
        {"name": "weight_decay", "type": "range", "bounds": [1e-10, 1e-2], "log_scale": True, "value_type": "float"},
                ],

    evaluation_function=train_evaluate,
                objective_name='accuracy',
                minimize=False,
                total_trials=N_TRIALS

            )
    TOC = time.time()
    means, covariances  = values

    lr_opt              = best_parameters['lr']
    momentum_opt        = best_parameters['momentum']
    weight_decay_opt    = best_parameters['weight_decay']
    acc_BO              = means['accuracy']

    
    # TRAIN NETWORK USING OPTIMUM PARAMS
    save_model_path    = os.path.join(output_folder, 'optimum_model_checkpoint.pth.tar');
    saved_dict_path    = os.path.join(output_folder, 'optimum_model_dictionary.json');


    loss_fn             = nn.CrossEntropyLoss()
    lr                  = lr_opt
    momentum            = momentum_opt
    weight_decay        = weight_decay_opt

    model               = AlexNet(NEW_NUM_CLASSES)
    optimizer           = torch.optim.SGD(model.parameters(),
                                lr              = lr,
                                momentum        = momentum,
                                weight_decay    = weight_decay)

    os.mkdir(output_folder) 

    _,trained_model     = train_classifier_simple_v1(
                                num_epochs              = NUM_EPOCHS,
                                model                   = model,
                                optimizer               = optimizer,
                                loss_fn                 = loss_fn,
                                train_loader            = train_loader,
                                valid_loader            = valid_loader,
                                test_loader             = None,
                                skip_epoch_stats        = False,
                                save_epoch_stats_name   = saved_dict_path,
                                save_model              = True,
                                save_model_name         = save_model_path,
                                skip_progress           = True,
                                device                  = DEVICE,
                                )

    acc_end     = compute_accuracy(model=trained_model, data_loader=test_loader,device=DEVICE)
    acc_end     = float(acc_end)

    

    # LOGGING OPTIMUM PARAMETERS
    opt_params_fname = os.path.join(output_folder, 'Optimized_Parameters.txt')
    file = open(opt_params_fname, "a")  # append mode
    file.write('Path : '            + output_folder + '\n')
    file.write('\n')
    file.write('learning_rate = '   + str(lr_opt) + '\n')
    file.write('momentum = '        + str(momentum_opt) + '\n')
    file.write('weight_decay = '    + str(weight_decay_opt) + '\n')
    file.write('accuracy_BO (%) = '    + str(acc_BO) + '\n')
    file.write('accuracy_END (%) = '    + str(acc_end) + '\n')
    file.write('Running time (sec.) = '    + str(TOC-TIC) + '\n')
    file.close()


    # SAVING INPUT PARAMETERS
    input_params_fname = os.path.join(output_folder, 'Input_Parameters.txt')
    file = open(input_params_fname, "a")  # append mode
    file.write('Path : ' + output_folder + '\n')
    file.write('\n')
    file.write('BASE_NUM_CLASSES \t= '     + str(BASE_NUM_CLASSES) + '\n')
    file.write('NEW_NUM_CLASSES \t= '     + str(NEW_NUM_CLASSES) + '\n')
    file.write('BATCH_SIZE \t\t= '      + str(BATCH_SIZE) + '\n')
    file.write('imgSize \t\t= '         + str(imgSize) + '\n')
    file.write('NUM_EPOCHS \t\t= '      + str(NUM_EPOCHS) + '\n')
    file.write('N_TRIALS \t\t= '        + str(N_TRIALS) + '\n')
    file.write('NUM_WORKERS \t= '     + str(NUM_WORKERS) + '\n')
    file.write('DEVICE \t\t\t= '          + DEVICE + '\n')
    file.write('NVSim_path \t= '     + NVSim_path + '\n')
    file.write('NVReal_path \t= '     + NVReal_path + '\n')
    file.write('NVReal_ratio \t= '    + str(NVReal_train_ratio) + '_'+ str(NVReal_valid_ratio) + '_' + str(NVReal_test_ratio) + '\n')
    file.write('NVSim_ratio \t= '    + str(NVSim_train_ratio) + '_'+ str(NVSim_valid_ratio) + '_' + str(NVSim_test_ratio) + '\n')
    file.write('NVReal_percentage_train \t= '    + str(NVReal_Percentage_Train) + '\n')
    file.write('Experiment_Type \t= '    + str(Exp_Type) + '\n')
    file.close()



    # SYSTEM AND HARDWARE
    
    with open("/proc/cpuinfo", "r")  as f:
        info    = f.readlines()

    cpuinfo     = [x.strip().split(":")[1] for x in info if "model name"  in x]
    CPUName     = str(cpuinfo[0])
    CPUCount    = str(len(cpuinfo))
    CPU_arch    = str(platform.system())

    RAM_total   = str(float(psutil.virtual_memory().total/1024**3))  # total physical memory in Bytes
    PythonName  = str(sys.version_info) # Python 
    OS_name     = str(platform.platform())
    GPUName     = str(torch.cuda.get_device_name()) #GPU

    sys_and_hw_fname = os.path.join(output_folder, 'Systems_and_Hardware.txt')
    file = open(sys_and_hw_fname, "a")  # append mode
    file.write('Path : ' + output_folder + '\n')
    file.write('\n')
    file.write('CPU Name \t= '     + CPUName + '\n')
    file.write('CPU Count \t\t= '      + CPUCount + '\n')
    file.write('CPU Architecture  \t\t= '         + CPU_arch + '\n')
    file.write('Total RAM (GB) \t\t= '      + RAM_total + '\n')
    file.write('Python \t\t= '        + PythonName + '\n')
    file.write('Opr. System \t= '     + OS_name + '\n')
    file.write('GPU Name \t\t\t= '          + GPUName + '\n')

    file.close()
    

    print('DONE : Saved to ' + output_folder)



if __name__ == '__main__':
    main()
