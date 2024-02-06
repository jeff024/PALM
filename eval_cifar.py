import os
import time
from util.loaders.args_loader import get_args
import torch
import numpy as np
from util.evaluations.mahalanobis import *
from util.evaluations.metrics import compute_traditional_ood
import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

eval_maha(args)
cache_dir = os.path.join("cache", f"{args.backbone}-{args.method}")
shutil.rmtree(cache_dir)
