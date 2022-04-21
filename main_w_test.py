# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import comet_ml
import argparse
import os
import random
import wget
import time
import warnings
import json
import collections
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Subset
from train import train, validate, cuda_transfer
from utils import get_dataset, get_model, get_optimizer, get_scheduler, get_curr_dl
from utils import  LossTracker,run_cmd
from torch.utils.data import DataLoader
from utils import get_pacing_function,balance_order_val, get_args, set_seed, get_order, setup_comet

args = get_args()

def main():
    set_seed(args.seed) 
    # create training and validation datasets and intiate the dataloaders
    tr_set = get_dataset(args, 'train',)
    if args.dataset == "cifar100N":
        val_set = get_dataset(args, 'val')
        tr_set_clean = get_dataset(args, 'train')
    else:
        val_set = get_dataset(args, 'val')        
    train_loader = DataLoader(tr_set, batch_size=args.batchsize,\
                              shuffle=True, num_workers=args.workers, pin_memory=True)  
    test_loader = DataLoader(val_set, batch_size=args.batchsize*2,
                      shuffle=False, num_workers=args.workers, pin_memory=True)

    # initiate a recorder for saving and loading stats and checkpoints
    order = get_order(args)
    
    order,order_val = balance_order_val(order, tr_set, num_classes=len(tr_set.classes)) 
   
    #decide CL, Anti-CL, or random-CL
    if args.ordering == "random":
        np.random.shuffle(order)
    elif  args.ordering == "anti_curr":
        order = [x for x in reversed(order)]
      
    #check the statistics 
    bs = args.batchsize
    N = len(order)
    myiterations = (N//bs+1)*args.epochs
    
    #initial training
    model = get_model(args.arch, tr_set.nchannels, tr_set.imsize, len(tr_set.classes))
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.momentum, args.wd)
    scheduler = get_scheduler(args.scheduler, optimizer, num_epochs=myiterations)

    start_epoch = 0
    total_iter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],"test_loss": [], "test_acc": [], "iter": [0,] }
    start_time = time.time()
    
    if args.dataset == "cifar100N":
        val_set = Subset(tr_set_clean, order_val)
    else:
        val_set = Subset(tr_set, order_val)    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batchsize*2,
                              shuffle=False, num_workers=args.workers, pin_memory=True)                           
    trainsets = Subset(tr_set, order)
    train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.workers, pin_memory=True) 
    criterion = nn.CrossEntropyLoss().cuda()

    # Comet.ml logging
    exp = setup_comet(args)
    exp.log_parameters({k:w for k,w in vars(args).items() if "comet" not in k})
    model.comet_experiment_key = exp.get_key() # To retrieve existing experiment

    if args.ordering == "standard":
        iterations = 0
        for epoch in range(1,args.epochs+1): 
            tr_loss, tr_acc1, iterations = train(args, train_loader, model, criterion, optimizer,scheduler, epoch,iterations)
            val_loss, val_acc1 = validate(args, val_loader, model, criterion)
            test_loss, test_acc1 = validate(args, test_loader, model, criterion)                 
            print ("%s epoch %s iterations w/ LEARNING RATE %s"%(epoch, iterations,optimizer.param_groups[0]["lr"])) 
            history["test_loss"].append(test_loss)
            history["test_acc"].append(test_acc1)                           
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc1)  
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc1)
            history["iter"].append(iterations)
            exp.log_metrics({'loss': tr_loss, "acc": tr_acc1}, prefix="train", step=iterations, epoch=epoch)
            exp.log_metrics({'loss': val_loss, "acc": val_acc1}, prefix="val", step=iterations, epoch=epoch)
            exp.log_metrics({'loss': test_loss, "acc": test_acc1}, prefix="test", step=iterations, epoch=epoch)
                
            torch.save(history,"stat.pt")  
    else:    
        pre_iterations = 0
        startIter = 0
        pacing_function = get_pacing_function(myiterations, N, args)
        # Define starting pacing
        startIter_next = pacing_function(0) # <=======================================
        print (f'0 iter data between {startIter} and {startIter_next} w/ Pacing {args.pacing_f}')
        train_loader = get_curr_dl(args, tr_set, order, startIter, startIter_next)
        #trainsets = Subset(tr_set, list(order[startIter:max(startIter_next,256)]))
        '''train_loader = torch.utils.data.DataLoader(trainsets, batch_size=args.batchsize,
                              shuffle=True, num_workers=args.workers, pin_memory=True)'''
        step = 0
        
        while step < myiterations:   
            tracker = LossTracker(len(train_loader), f'iteration : [{step}]', args.printfreq)
            for images, target in train_loader:
                step += 1
                images, target = cuda_transfer(images, target)
                output = model(images)
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                tracker.update(loss, output, target)
                tracker.display(step-pre_iterations)
             
            #If we hit the end of the dynamic epoch build a new data loader
            pre_iterations = step          
            if startIter_next <= N:            
                startIter_next = pacing_function(step)# <=======================================
                print ("%s iter data between %s and %s w/ Pacing %s and LEARNING RATE %s "%(step,startIter,startIter_next,args.pacing_f, optimizer.param_groups[0]["lr"]))
                train_loader = get_curr_dl(args, tr_set, order, startIter, startIter_next)
                
                       # start your record
            if step > 50: 
                tr_loss, tr_acc1 = tracker.losses.avg, tracker.top1.avg 
                val_loss, val_acc1 = validate(args, val_loader, model, criterion) 
                test_loss, test_acc1 = validate(args, test_loader, model, criterion)                       
                # record
                history["test_loss"].append(test_loss)
                history["test_acc"].append(test_acc1)                                       
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc1)                 
                history["train_loss"].append(tr_loss)
                history["train_acc"].append(tr_acc1)  
                history['iter'].append(step)
                exp.log_metrics({'loss': tr_loss, "acc": tr_acc1}, prefix="train", step=step, epoch=step//352+1)
                exp.log_metrics({'loss': val_loss, "acc": val_acc1}, prefix="val", step=step, epoch=step/352+1)
                exp.log_metrics({'loss': test_loss, "acc": test_acc1}, prefix="test", step=step, epoch=step//352+1)
            
                torch.save(history,"stat.pt")  
                # reinitialization<=================
                model.train()

if __name__ == '__main__':
    main()

