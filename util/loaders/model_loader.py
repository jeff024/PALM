import torch
import torch.nn as nn
from util.loss_functions import *
from torch.optim.lr_scheduler import MultiStepLR
from models.resnet import  SupCEResNet, PALMResNet

def get_model(args, num_classes, load_ckpt=True):
    method = args.method
    if 'palm' in method:
        model = PALMResNet(name=args.backbone)
    else:
        model = SupCEResNet(name=args.backbone, num_classes=num_classes)
        
    if load_ckpt:  
        checkpoint = torch.load(args.save_path, map_location="cuda:0")
        model.load_state_dict(checkpoint)
        print(f"ckpt loaded from {args.save_path}")

    model.eval()
    
    # get the number of model parameters
    print(f'{args.backbone}-{args.method}: Number of model parameters: {sum([p.data.nelement() for p in model.parameters()])}')
    return model


def get_criterion(args, num_classes, model):
    arch = args.method
    if 'palm' in arch:
        cri = PALM(args, temp=args.temp, num_classes=num_classes, proto_m=args.proto_m, n_protos=num_classes*args.cache_size,  k=args.k, lambda_pcon=args.lambda_pcon)
    else:
        cri = nn.CrossEntropyLoss()

    return cri
    
def get_encoder_dim(model):
    dummy_input = torch.zeros((1, 3, 32, 32))
    features = model.encoder(dummy_input)
    featdims = features.shape[1]
    return featdims
    
def set_model(args, num_classes, load_ckpt=True, load_epoch=None):
    model = get_model(args, num_classes, load_ckpt)
    criterion = get_criterion(args, num_classes, model)
    return model, criterion

def set_schedular(args, optimizer):
    scheduler = MultiStepLR(optimizer, milestones=[50,75,90], gamma=0.1)
    return scheduler