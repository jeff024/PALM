#! /usr/bin/env python3
import torch
import os
from util.loaders.args_loader import get_args
from util.loaders.data_loader import get_loader_in, get_loader_out
from util.loaders.model_loader import get_model
import numpy as np
import torch.nn.functional as F
import time
from tqdm import tqdm


device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = get_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
np.random.seed(args.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


trainloaderIn, num_classes = get_loader_in(args, split='train', mode='eval')
testloaderIn, _ = get_loader_in(args, split='val', mode='eval')
model = get_model(args, num_classes, load_ckpt=True)
model.to(device)
model.eval()

batch_size = args.batch_size

FORCE_RUN = True

dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
features = model.encoder(dummy_input)
featdims = features.shape[1]

begin = time.time()

for split, in_loader in [('train', trainloaderIn), ('val', testloaderIn), ]:
    in_save_dir = os.path.join("cache", f"{args.backbone}-{args.method}", args.in_dataset)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)
    cache_name = os.path.join(
        in_save_dir, f"{split}_{args.backbone}-{args.method}_features.npy")
    label_cache_name = os.path.join(
        in_save_dir, f"{split}_{args.backbone}-{args.method}_labels.npy")
    if FORCE_RUN or not os.path.exists(cache_name):
        print(f"Processing in-distribution {args.in_dataset} images")
        t0 = time.time()
        ########################################
        features, labels = [], []
        total = 0

        model.eval()

        for index, (img, label) in enumerate(tqdm(in_loader)):

            img = img.cuda()

            features += list(model.encoder(img).data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

            total += len(img)

        feat_log, label_log = np.array(features), np.array(labels)
        ########################################
        np.save(cache_name, feat_log)
        np.save(label_cache_name, label_log)
        print(
            f"{total} images processed, {time.time()-t0} seconds used\n")

for ood_dataset in args.out_datasets:
    # print(f"OOD Dataset: {ood_dataset}")
    out_loader = get_loader_out(args, dataset=ood_dataset, split=('val'), mode='eval')

    out_save_dir = os.path.join(in_save_dir, ood_dataset)
    if not os.path.exists(out_save_dir):
        os.makedirs(out_save_dir)
    cache_name = os.path.join(out_save_dir, f"{args.backbone}-{args.method}_features.npy")
    label_cache_name = os.path.join(out_save_dir, f"{args.backbone}-{args.method}_labels.npy")

    if FORCE_RUN or not os.path.exists(cache_name):
        t0 = time.time()
        print(f"Processing out-of-distribution {ood_dataset} images")

        ########################################
        features, labels = [], []
        total = 0

        model.eval()

        for index, (img, label) in enumerate(tqdm(out_loader)):

            img, label = img.cuda(), label.cuda()

            features += list(model.encoder(img).data.cpu().numpy())
            labels += list(label.data.cpu().numpy())

            total += len(img)

        feat_log, label_log = np.array(features), np.array(labels)
        ########################################
        np.save(cache_name, feat_log)
        np.save(label_cache_name, label_log)
        print(f"{total} images processed, {time.time()-t0} seconds used\n")
        t0 = time.time()

print(time.time() - begin)
