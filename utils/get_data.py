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

import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset
from .cifar_label import *
import collections
import wget

def get_order(args):
  if 'cscores-orig-order.npz' in args.order_dir:
    temp_path = os.path.join("orders",args.dataset+'-cscores-orig-order.npz')
    if not os.path.isfile(temp_path):        
        print ('Downloading the data cifar10-cscores-orig-order.npz and cifar100-cscores-orig-order.npz to folder orders')
        if 'cifar100' == args.dataset:
            url = 'https://pluskid.github.io/structural-regularity/cscores/cifar100-cscores-orig-order.npz'
        elif 'cifar10' == args.dataset:
            url = 'https://pluskid.github.io/structural-regularity/cscores/cifar10-cscores-orig-order.npz'
        elif 'imagenet' == args.dataset:
            url = "https://pluskid.github.io/structural-regularity/cscores/imagenet-cscores-with-filename.npz"
        wget.download(url, './orders')
    temp_x = np.load(temp_path)['scores']
    ordering = collections.defaultdict(list)
    list(map(lambda a, b: ordering[a].append(b), np.arange(len(temp_x)),temp_x))
    order = [k for k, v in sorted(ordering.items(), key=lambda item: -1*item[1][0])]
  else:
    print ('Please check if the files %s in your folder -- orders. See ./orders/README.md for instructions on how to create the folder' %(args.order_dir))
    order = [x for x in list(torch.load(os.path.join("orders",args.order_dir)).keys())]
  return order
 
def get_curr_dl(args, train_set, order, start, end):
  return torch.utils.data.DataLoader(Subset(train_set, list(order[start:max(end,256)])),\
                                                           batch_size=args.batchsize,\
                                                           shuffle=True, num_workers=args.workers, pin_memory=True)
     
def get_dataset(args, split, clean=False, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  dataset_name = args.dataset
  data_dir = args.data_dir
  rand_fraction = args.rand_fraction
  if split == "val" and dataset_name == "cifar100N":
    pass
  if dataset_name in [ 'cifar10', 'cifar100']:
    dataset = globals()[f'get_{dataset_name}'](dataset_name, data_dir, split, transform=imsize, imsize=imsize, bucket=bucket, **kwargs)    
  elif dataset_name in ['cifar100N']:
    dataset = globals()[f'get_{dataset_name}'](dataset_name, data_dir, split,rand_fraction= rand_fraction,transform=imsize, imsize=imsize, bucket=bucket,**kwargs)
  item = dataset.__getitem__(0)[0]
  print (item.size(0))
  dataset.nchannels = item.size(0)
  dataset.imsize = item.size(1)
  return dataset


def get_aug(split, imsize=None, aug='large'):
  if aug == 'large':
    imsize = imsize if imsize is not None else 224
    if split == 'train':
      return [transforms.RandomResizedCrop(imsize, scale=(0.2, 1.0)),transforms.RandomHorizontalFlip()]
      #return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]
    else:
      return [transforms.Resize(round(imsize * 1.143)), transforms.CenterCrop(imsize)]
  else:
    imsize = imsize if imsize is not None else 32
    if split == 'train':
        train_transform = []
      #return [transforms.RandomCrop(imsize, padding=round(imsize / 8))]
        train_transform.append(transforms.RandomCrop(32, padding=4))
        train_transform.append(transforms.RandomHorizontalFlip())
        return train_transform
    else:
      return [transforms.Resize(imsize), transforms.CenterCrop(imsize)]


def get_transform(split, normalize=None, transform=None, imsize=None, aug='large'):
  if transform is None:
    if normalize is None:
        if aug == 'large':
          normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
          normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])  
    transform = transforms.Compose(get_aug(split, imsize=imsize, aug=aug)
                                   + [transforms.ToTensor(), normalize])
  return transform


def get_cifar10(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  return datasets.CIFAR10(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)


def get_cifar100(dataset_name, data_dir, split, transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  return datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)

def get_cifar100N(dataset_name, data_dir, split, rand_fraction=None,transform=None, imsize=None, bucket='pytorch-data', **kwargs):
  transform = get_transform(split, transform=transform, imsize=imsize, aug='small')
  if split=='train':
    return CIFAR100N(root=data_dir, train=(split=='train'), transform=transform, download=True, rand_fraction=rand_fraction)
  else:
    datasets.CIFAR100(data_dir, train=(split=='train'), transform=transform, download=True, **kwargs)        

