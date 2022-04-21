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

from comet_ml import Experiment, ExistingExperiment
import collections
import torch
from torch import Tensor
import torch.nn as nn
import torch.backends.cudnn as cudnn
import subprocess
import os
import time
import shutil
from datetime import datetime
import torch.optim as optim
from torch.optim import lr_scheduler
import sys
sys.path.append("..")
from third_party import models
import numpy as np
import torch.nn as nn
import argparse
import random
import warnings

def get_args():
  parser = argparse.ArgumentParser(description='PyTorch Training')
  parser.add_argument('--data-dir', default='dataset',
                      help='path to dataset')
  parser.add_argument('--order-dir', default='cifar10-cscores-orig-order.npz',
                      help='path to train val idx')
  parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                      help='model architecture: (default: resnet18)')
  parser.add_argument('--dataset', default='cifar10', type=str,
                      help='dataset')
  parser.add_argument('--printfreq', default=10, type=int,
                      help='print frequency (default: 10)')
  parser.add_argument('--workers', default=4, type=int,
                      help='number of data loading workers (default: 4)')
  parser.add_argument('--epochs', default=100, type=int,
                      help='number of total epochs to run')
  parser.add_argument('-b', '--batchsize', default=128, type=int,
                      help='mini-batch size (default: 256), this is the total')
  parser.add_argument('--optimizer', default="sgd", type=str,
                      help='optimizer')
  parser.add_argument('--scheduler', default="cosine", type=str,
                      help='lr scheduler')
  parser.add_argument('--lr', default=0.1, type=float,
                      help='initial learning rate', dest='lr')
  parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                      help='momentum')
  parser.add_argument('--wd', default=5e-4, type=float,
                      help='weight decay (default: 1e-4)')
  parser.add_argument('--seed', default=None, type=int,
                      help='seed for initializing training. ')
  # curriculum params
  parser.add_argument("--pacing-f", default="linear", type=str, help="which pacing function to take")
  parser.add_argument('--pacing-a', default=1., type=float,
                      help='weight decay (default: 1e-4)')
  parser.add_argument('--pacing-b', default=1., type=float,
                      help='weight decay (default: 1e-4)')
  parser.add_argument("--ordering", default="curr", type=str, help="which test case to use. supports: standard, curriculum, anti and random")
  parser.add_argument('--rand-fraction', default=0., type=float,
                    help='label curruption (default:0)')
  parser.add_argument("--cometKey", type=str)
  parser.add_argument("--cometWs", type=str)
  parser.add_argument("--cometName", type=str)

  return parser.parse_args()

# Comet Experiments
def setup_comet(args, resume_experiment_key=''):
    api_key = args.cometKey 
    workspace = args.cometWs 
    project_name = args.cometName
    enabled = bool(api_key) and bool(workspace)
    disabled = not enabled
    print(f"Setting up comet logging using: {{api_key={api_key}, workspace={workspace}, enabled={enabled}}}")

    if resume_experiment_key:
        experiment = ExistingExperiment(api_key=api_key, previous_experiment=resume_experiment_key)
        return experiment

    experiment_name = get_prefix(args)

    experiment = Experiment(api_key=api_key, parse_args=False, project_name=project_name,
                            workspace=workspace, disabled=disabled)
    if experiment_name:
      experiment.set_name(experiment_name)
    return experiment

def get_prefix(args):
    return "_".join([str(w) for k,w in vars(args).items() if "comet" not in k])

def run_cmd(cmd_str, prev_sp=None):
  """
  This function runs the linux command cmr_str as a subprocess after waiting
  for prev_sp subprocess to finish
  """
  if prev_sp is not None:
    prev_sp.wait()
  return subprocess.Popen(cmd_str, shell=True)#, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))


def get_model(model_name, nchannels=3, imsize=32, nclasses=10, half=False):

  ngpus = torch.cuda.device_count()

  print("=> Creating model '{}'".format(model_name))
  if imsize < 128 and model_name in models.__dict__:
    model = models.__dict__[model_name](num_classes=nclasses, nchannels=nchannels)
  model = nn.DataParallel(model).cuda()
  cudnn.benchmark = True
  return model

def get_optimizer(optimizer_name, parameters, lr, momentum=0, weight_decay=0):
  if optimizer_name == 'sgd':
    return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay)
  elif optimizer_name == 'nesterov_sgd':
    return optim.SGD(parameters, lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
  elif optimizer_name == 'rmsprop':
    return optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
  elif optimizer_name == 'adagrad':
    return optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
  elif optimizer_name == 'adam':
    return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

def get_scheduler(scheduler_name, optimizer, num_epochs, **kwargs):
  if scheduler_name == 'constant':
    return lr_scheduler.StepLR(optimizer, num_epochs, gamma=1, **kwargs)

  elif scheduler_name == 'step2':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 2), gamma=0.1, **kwargs)
  elif scheduler_name == 'step3':
    return lr_scheduler.StepLR(optimizer, round(num_epochs / 3), gamma=0.1, **kwargs)
  elif scheduler_name == 'exponential':
    return lr_scheduler.ExponentialLR(optimizer, (1e-3) ** (1 / num_epochs), **kwargs)
  elif scheduler_name == 'cosine':
    return lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, **kwargs)
  elif scheduler_name == 'step-more':
    return lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2, **kwargs)


def run_cmd(cmd_str, prev_sp=None):
  """
  This function runs the linux command cmr_str as a subprocess after waiting
  for prev_sp subprocess to finish
  """
  if prev_sp is not None:
    prev_sp.wait()
  return subprocess.Popen(cmd_str, shell=True)#, stdout=open(os.devnull, 'w'), stderr=open(os.devnull, 'w'))

def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                    'This will turn on the CUDNN deterministic setting, '
                    'which can slow down your training considerably! '
                    'You may see unexpected behavior when restarting '
                    'from checkpoints.')

class LossTracker(object):
  def __init__(self, num, prefix='', print_freq=1):
    self.print_freq=print_freq
    self.batch_time = AverageMeter('Time', ':6.3f')
    self.losses = AverageMeter('Loss', ':.4f')
    self.top1 = AverageMeter('Acc@1', ':6.2f')
    self.top5 = AverageMeter('Acc@5', ':6.2f')
    self.progress = ProgressMeter( num, [self.batch_time, self.losses, self.top1, self.top5], prefix=prefix)
    self.end = time.time()

  def update(self, loss, output, target):
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    self.losses.update(loss.item(), output.size(0))
    self.top1.update(acc1[0], output.size(0))
    self.top5.update(acc5[0], output.size(0))

  def display(self, step):
    self.batch_time.update(time.time() - self.end)
    self.end = time.time()
    if step % self.print_freq == 0:
      self.progress.display(step)
    

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def balance_order(order, dataset, num_classes=10):
    size_each_class = len(dataset) // num_classes
   
    class_orders = collections.defaultdict(list)
    for i in range(len(order)):
        class_orders[dataset.targets[order[i]]].append(i)
    # take each group containing the next easiest image for each class,
    # and putting them according to diffuclt-level in the new order
    length = []
    for cls in range(num_classes):
        length.append(len(class_orders[cls]))
    print ("minmax", min(length), max(length))  
    new_order = []
    
    for group_idx in range(min(length)):            
        group = sorted([class_orders[cls][group_idx] for cls in range(num_classes)])
        new_order.extend([order[idx] for idx in group])
        
    for group_idx in range(min(length), max(length)):
        cls_idx = [cls for cls in range(num_classes) if group_idx<length[cls]]
        group = sorted([class_orders[cls][group_idx] for cls in cls_idx])
        new_order.extend([order[idx] for idx in group])        
    assert len(new_order) == len(order)
    return new_order


def balance_order_val(order, dataset, num_classes=10,valp=0.1):
    size_each_class = len(dataset) // num_classes
    print (" size_each_class ", size_each_class )
    class_orders = collections.defaultdict(list)
    for i in range(len(order)):
        class_orders[dataset.targets[order[i]]].append(i)
    # take each group containing the next easiest image for each class,
    # and putting them according to diffuclt-level in the new order
    length = []
    new_order_val = []
    class_orders_new = collections.defaultdict(list)
    for cls in range(num_classes):
        np.random.seed(cls)
        tmp_id = np.array(class_orders[cls])
        random_id = np.random.choice(len(tmp_id),int(len(tmp_id)*valp),replace=False)
        tmp_id_val = [np.array(tmp_id)[ID] for ID in random_id ]  
        new_order_val.extend([order[idx] for idx in tmp_id_val ])
        
        class_orders_new[cls].extend([x for x in class_orders[cls] if x not in tmp_id_val])
        length.append(len(class_orders_new[cls]))
        
    new_order = []
    for group_idx in range(min(length)):            
        group = sorted([class_orders_new[cls][group_idx] for cls in range(num_classes)])
        new_order.extend([order[idx] for idx in group])
        
    for group_idx in range(min(length), max(length)):
        cls_idx = [cls for cls in range(num_classes) if group_idx<length[cls]]
        group = sorted([class_orders_new[cls][group_idx] for cls in cls_idx])
        new_order.extend([order[idx] for idx in group])  
           
    assert np.sum(new_order)+np.sum(new_order_val) == np.sum(order)
    
    return new_order,new_order_val

        
def get_pacing_function(total_step, total_data, args):
    """Return a  pacing function  w.r.t. step.
    input:
    a:[0,large-value] percentage of total step when reaching to the full data. This is an ending point (a*total_step, total_data)) 
    b:[0,1]  percentatge of total data at the begining of the training. Thia is a starting point (0,b*total_data))
    """
    a = args.pacing_a
    b = args.pacing_b 
    index_start = b*total_data
    if args.pacing_f == 'linear':
      rate = (total_data - index_start)/(a*total_step)
      def _linear_function(step):
        return int(rate *step + index_start)
      return _linear_function
    
    elif args.pacing_f == 'quad':
      rate = (total_data-index_start)/(a*total_step)**2  
      def _quad_function(step):
        return int(rate*step**2 + index_start)
      return _quad_function
    
    elif args.pacing_f == 'root':
      rate = (total_data-index_start)/(a*total_step)**0.5
      def _root_function(step):
        return int(rate *step**0.5 + index_start)
      return _root_function
    
    elif args.pacing_f == 'step':
      threshold = a*total_step
      def _step_function(step):
        return int( total_data*(step//threshold) +index_start)
      return _step_function      

    elif args.pacing_f == 'exp':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      rate =  (total_data-tilde_b)/(np.exp(c)-1)
      constant = c/tilde_a
      def _exp_function(step):
        if not np.isinf(np.exp(step *constant)):
            return int(rate*(np.exp(step*constant)-1) + tilde_b )
        else:
            return total_data
      return _exp_function

    elif args.pacing_f == 'log':
      c = 10
      tilde_b  = index_start
      tilde_a  = a*total_step
      ec = np.exp(-c)
      N_b = (total_data-tilde_b)
      def _log_function(step):
        return int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b )
      return _log_function
