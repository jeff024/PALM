import time
from util.evaluations import metrics
import torch
import numpy as np
from util.evaluations.write_to_csv import write_csv

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_maha(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    class_num = 10 if args.in_dataset == "CIFAR-10" else 100
    
    feat_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/train_{args.backbone}-{args.method}_features.npy", allow_pickle=True)
    label_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/train_{args.backbone}-{args.method}_labels.npy", allow_pickle=True)
    feat_log = feat_log.astype(np.float32)

    feat_log_val = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/val_{args.backbone}-{args.method}_features.npy", allow_pickle=True)
    label_log_val = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/val_{args.backbone}-{args.method}_labels.npy", allow_pickle=True)
    feat_log_val = feat_log_val.astype(np.float32)

    ood_feat_log_all = {}
    for ood_dataset in args.out_datasets:
        ood_feat_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/{ood_dataset}/{args.backbone}-{args.method}_features.npy", allow_pickle=True)
        ood_label_log = np.load(f"cache/{args.backbone}-{args.method}/{args.in_dataset}/{ood_dataset}/{args.backbone}-{args.method}_labels.npy", allow_pickle=True)
        ood_feat_log = ood_feat_log.astype(np.float32)
        ood_feat_log_all[ood_dataset] = ood_feat_log

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))# Last Layer only

    ftrain = prepos_feat(feat_log)
    ftest = prepos_feat(feat_log_val)
    food_all = {}
    for ood_dataset in args.out_datasets:
        food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])


# #################### SSD+ score OOD detection #################
    begin = time.time()
    mean_feat = ftrain.mean(0)
    std_feat = ftrain.std(0)
    prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
    ftrain_ssd = prepos_feat_ssd(ftrain)
    ftest_ssd = prepos_feat_ssd(ftest)
    food_ssd_all = {}
    for ood_dataset in args.out_datasets:
        food_ssd_all[ood_dataset] = prepos_feat_ssd(food_all[ood_dataset])
    
    cov = lambda x: np.cov(x.T, bias=True)
    
    def maha_score(X):
        z = X-mean_feat
        inv_sigma = np.linalg.pinv(cov(ftrain_ssd))
        return -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)

    dtest = maha_score(ftest_ssd)
    all_results = []
    for name, food in food_ssd_all.items():
        print(f"Evaluating {name}")
        dood = maha_score(food)
        results = metrics.cal_metric(dtest, dood)
        all_results.append(results)
    
    metrics.print_all_results(all_results, args.out_datasets, 'SSD+')
    args.score = "mahalanobis"
    write_csv(args, all_results)
    print(time.time() - begin)
    
    