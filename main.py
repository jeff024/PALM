import torch
import os
import torch.backends.cudnn as cudnn
from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_loader_in
from util.loaders.model_loader import set_model
from util.train_utils import get_optimizer
from trainer import get_trainer
import numpy as np
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def main():

    train_loader, num_classes = get_loader_in(args, split='train')

    model, criterion = set_model(args, num_classes, load_ckpt=False)
    model.to(device)
    model.encoder.to(device)
    criterion.to(device)


    # build optimizer
    optimizer = get_optimizer(args, model, criterion)
    loss_min = np.Inf

    # tensorboard
    t = datetime.now().strftime("%d-%B-%Y-%H:%M:%S")
    logger = SummaryWriter(log_dir=f"runs/{args.backbone}-{args.method}/{t}")

    # get trainer and scaler
    trainer = get_trainer(args)
    scaler = torch.cuda.amp.GradScaler()
                
    for epoch in tqdm(range(args.epochs)):
        loss = trainer(args, train_loader, model, criterion, optimizer, epoch, scaler=scaler)
        
        if type(loss)==tuple:
            loss, l_dict = loss
            logger.add_scalar('Loss/train', loss, epoch)
            for k in l_dict.keys():
                logger.add_scalar(f'Loss/{k}', l_dict[k], epoch)
        else:
            logger.add_scalar('Loss/train', loss, epoch)
        logger.add_scalar('Lr/train', optimizer.param_groups[0]['lr'], epoch)

        if loss < loss_min:
            loss_min = loss
            torch.save(model.state_dict(), args.save_path)

if __name__ == "__main__":

    FORCE_RUN = True
    # FORCE_RUN=False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cudnn.benchmark = True

    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(args)
    args.save_epoch = 50

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # check if the model is trained
    if os.path.exists(args.save_path) and not FORCE_RUN:
        exit()

    main()
