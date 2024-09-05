import torch
import numpy as np
import random
import os
import torch.nn as nn

def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files


def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_accuracy(outputs, labels, binary=False):
    #if multi-label classification
    if len(labels.size())>1:
        outputs = (outputs>0.0).float()
        correct = ((outputs==labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()
    if binary:
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)
    c = (predictions == labels).float().squeeze()
    accuracy = torch.mean(c)
    return accuracy.item()


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


def make_log_name(args):
    log_name = args.model

    if args.mode == 'eval':
        log_name = args.modelpath.split('/')[-1]
        # remove .pt from name
        log_name = log_name[:-3]

    else:
        if args.pretrained:
            log_name += '_pretrained'
        log_name += '_seed{}_epochs{}_bs{}_lr{}'.format(args.seed, args.epochs, args.batch_size, args.lr)

        if args.method == 'adv_debiasing':
            log_name += '_advlamb{}_eta{}'.format(args.adv_lambda, args.eta)

        elif args.method == 'scratch_mmd' or args.method == 'kd_mfd':
            log_name += '_{}'.format(args.kernel)
            log_name += '_sigma{}'.format(args.sigma) if args.kernel == 'rbf' else ''
            log_name += '_jointfeature' if args.jointfeature else ''
            log_name += '_lambf{}'.format(args.lambf) if args.method == 'scratch_mmd' else ''

        if args.labelwise:
            log_name += '_labelwise'

        if args.teacher_path is not None:
            log_name += '_temp{}'.format(args.kd_temp)
            log_name += '_lambh{}_lambf{}'.format(args.lambh, args.lambf)

            if args.no_annealing:
                log_name += '_fixedlamb'
        if args.dataset == 'celeba' and args.target != 'Attractive':
            log_name += '_{}'.format(args.target)
               
    return log_name

def allignments(model):
    layer_list = []
    outputs = model(torch.rand([1,3,176,176]), get_inter = True)
    teacher_feature_size = outputs[0].size(1)
    student_feature_size = outputs[0].size(1)
    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
    layer_list.append(nn.Linear(student_feature_size, teacher_feature_size))
    model.adaptation_layers = nn.ModuleList(layer_list)
    model.adaptation_layers.cuda()
    return model

def calculate_accuracy(preds, targets):
    if len(preds) == 0:  # To handle any case where there might be no data for a combination
        return None
    return np.mean(preds == targets) * 100