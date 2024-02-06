import torch
import numpy as np
from tqdm import tqdm

from util.train_utils import adjust_learning_rate, AverageMeter


def train_palm(args, train_loader, model, criterion, optimizer, epoch, scaler=None):
    model.train()

    losses = AverageMeter()
    sub_loss = {}

    for step, (images, labels) in enumerate((train_loader), start=epoch * len(train_loader)):
        if (len(images)) == 2:
            twocrop = True
            images = torch.cat([images[0], images[1]], dim=0)
        else:
            twocrop = False
            
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # compute loss
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                if args.fine_tune:
                    features = model.fine_tune_forward(images)
                else:
                    features = model(images)
                if twocrop:
                    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                else:
                    features = features.unsqueeze(1)
                loss, l_dict = criterion(features, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            old_scale = scaler.get_scale()
            scaler.update()    
            new_scale = scaler.get_scale()   

        losses.update(loss.item(), bsz)
        
        if new_scale >= old_scale:
            adjust_learning_rate(args, optimizer, train_loader, step)
            
        if step%len(train_loader) == 0:
            for k in l_dict.keys():
                sub_loss[k] = []
                
        for k in l_dict.keys():
            sub_loss[k].append(l_dict[k])
            
    for k in sub_loss.keys():
        sub_loss[k] = np.mean(sub_loss[k])
    

    return losses.avg, sub_loss


def train_supervised(args, train_loader, model, criterion, optimizer, epoch, warmup_schedular=None, schedular=None, scaler=None, index=None, index_map=None, k=1):
    model.train()

    losses = AverageMeter()
    sub_loss = {}

    for step, (images, labels) in enumerate(train_loader, start=epoch * len(train_loader)):

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # warm-up learning rate
        # warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        bsz = labels.shape[0]
        # compute loss
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                features = model(images)
                loss = criterion(features, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()    
        else:
            features = model(images)
            loss = criterion(features, labels)
            # SGD
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), bsz)

        # adjust_learning_rate(args, optimizer, train_loader, step, warmup_schedular=warmup_schedular, schedular=schedular)
            
        if step%len(train_loader) == 0:
            sub_loss['train'] = []
                
        for k in sub_loss.keys():
            sub_loss[k].append(loss.item())
            
    for k in sub_loss.keys():
        sub_loss[k] = np.mean(sub_loss[k])
    

    return losses.avg, sub_loss

def get_trainer(args):
    arch = args.method
    
    if "palm" in arch:
            trainer = train_palm
    else:
        trainer = train_supervised
    return trainer 