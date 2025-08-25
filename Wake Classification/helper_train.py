from helper_evaluate import compute_accuracy
from helper_evaluate import compute_epoch_loss
from helper_evaluate import compute_accuracy_single_batch
from helper_evaluate import compute_epoch_loss_single_batch
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import subprocess
import sys
import xml.etree.ElementTree
from utils import*

import time

from tqdm import tqdm
def train_classifier_simple_v1(num_epochs, model, optimizer, loss_fn,     
                               train_loader=None, 
                               valid_loader=None,
                               test_loader=None,
                               device="cuda",
                               skip_epoch_stats=False,
                               save_epoch_stats_name = None,
                               save_model = False,
                               save_model_name = None,
                               skip_progress = False,
                               ):
    
    #st = time.time()

    model.to(device)
    log_dict = {'train_acc_per_epoch': [],
                'train_loss_per_epoch': [],
                'valid_acc_per_epoch': [],
                'valid_loss_per_epoch': [],
                }
    is_single_batch = (len(train_loader)==1)
    if is_single_batch :
        features, targets = next(iter(train_loader))
        features = features.to(device)
        targets = targets.to(device)

        if valid_loader is not None:
            valid_features, valid_targets = next(iter(valid_loader))
            valid_features = valid_features.to(device)
            valid_targets = valid_targets.to(device)


    if not skip_progress:
        epoch_range = tqdm(range(num_epochs))
    else:
        epoch_range = range(num_epochs)

    #for epoch in (tqdm(range(num_epochs))):
    for epoch in epoch_range:

        #st = time.time()
        model.train()
        if is_single_batch :

            logits = model(features)
            loss = loss_fn(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        else:
            
            for batch_idx, (features, targets) in enumerate(train_loader):

                features = features.to(device)
                targets = targets.to(device)

                # FORWARD AND BACK PROP
                logits = model(features)
                loss = loss_fn(logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if not skip_progress:
            epoch_range.set_postfix(loss=loss.item())
            

        


        #print('Execution time:', time.time()-st, 'seconds')

        #st = time.time()
        # LOGGING PER EPOCH
        if not skip_epoch_stats:
            model.eval()
            with torch.set_grad_enabled(False):  # save memory during inference

                if is_single_batch :
                    train_acc = compute_accuracy_single_batch(model, features, targets)
                    train_loss = compute_epoch_loss_single_batch(model, features, targets)
                else:
                    train_acc = compute_accuracy(model, train_loader, device)
                    train_loss = compute_epoch_loss(model, train_loader, device)

                log_dict['train_loss_per_epoch'].append(train_loss.item())
                log_dict['train_acc_per_epoch'].append(train_acc.item())

                if valid_loader is not None:
                    if is_single_batch :
                        valid_acc = compute_accuracy_single_batch(model, valid_features, valid_targets)
                        valid_loss = compute_epoch_loss_single_batch(model, valid_features, valid_targets)
                    else:
                        valid_acc = compute_accuracy(model, valid_loader, device)
                        valid_loss = compute_epoch_loss(model, valid_loader, device)


                    log_dict['valid_acc_per_epoch'].append(valid_acc.item())
                    log_dict['valid_loss_per_epoch'].append(valid_loss.item())

                    

        #print('Execution time:', time.time()-st, 'seconds')

        if epoch==num_epochs-1:
            if save_model:
                checkpoint = {
                    "state_dict":model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, savename = save_model_name)

            if not skip_epoch_stats:
                save_dictionary(log_dict,   savename = save_epoch_stats_name)



    #print('Execution time:', time.time()-st, 'seconds')
    return log_dict, model


def train_classifier_simple_v2(
        model, num_epochs, train_loader,
        valid_loader, test_loader, optimizer,
        device, logging_interval=50,
        best_model_save_path=None,
        scheduler=None,
        skip_train_acc=False,
        scheduler_on='valid_acc'):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    best_valid_acc, best_epoch = -float('inf'), 0

    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
                print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                      f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                      f'| Loss: {loss:.4f}')

        model.eval()
        with torch.no_grad():  # save memory during inference
            if not skip_train_acc:
                train_acc = compute_accuracy(model, train_loader, device=device).item()
            else:
                train_acc = float('nan')
            valid_acc = compute_accuracy(model, valid_loader, device=device).item()
            train_acc_list.append(train_acc)
            valid_acc_list.append(valid_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc, best_epoch = valid_acc, epoch+1
                if best_model_save_path:
                    torch.save(model.state_dict(), best_model_save_path)

            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                  f'| Train: {train_acc :.2f}% '
                  f'| Validation: {valid_acc :.2f}% '
                  f'| Best Validation '
                  f'(Ep. {best_epoch:03d}): {best_valid_acc :.2f}%')

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')

        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError('Invalid `scheduler_on` choice.')

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')

    return minibatch_loss_list, train_acc_list, valid_acc_list