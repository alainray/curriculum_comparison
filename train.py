from utils import  LossTracker
import torch

def cuda_transfer(images, target):
    images = images.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    return images, target

def train(args, train_loader, model, criterion, optimizer,scheduler, epoch, iterations):
  # switch to train mode
  model.train()
  tracker = LossTracker(len(train_loader), f'Epoch: [{epoch}]', args.printfreq)
  for i, (images, target) in enumerate(train_loader):
    iterations += 1
    images, target = cuda_transfer(images, target)
    output = model(images)
    loss = criterion(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    tracker.update(loss, output, target)
    tracker.display(i)
    scheduler.step()
  return tracker.losses.avg, tracker.top1.avg,  iterations

def validate(args, val_loader, model, criterion):
  # switch to evaluate mode
  model.eval()
  with torch.no_grad():
    tracker = LossTracker(len(val_loader), f'val', args.printfreq)
    for i, (images, target) in enumerate(val_loader):
      images, target = cuda_transfer(images, target)
      output = model(images)
      loss = criterion(output, target)
      tracker.update(loss, output, target)
      tracker.display(i)
  return tracker.losses.avg, tracker.top1.avg