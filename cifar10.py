#!/usr/bin/env python

"""
    cifar10.py
    
    Train preactivation ResNet18 on CIFAR10 w/ linear learning rate annealing

"""

# for fast run for debug: --epochs=1 --extra=1 --burnout=1 --repeat_same_aug=1
##--load_csv=output/stage0/training_stats.csv

from __future__ import division, print_function


from alu_transforms import *
from model import ResNet18
import random
import matplotlib as mpl
mpl.use('Agg')


from lr import LRSchedule
from helpers import set_seeds

import torch
torch.backends.cudnn.benchmark = True
import os
from torchvision import transforms, datasets
from datetime import datetime
from my_aug import *

OUTPUT_DIR = os.path.join('.','output')
USE_CUDA = True if torch.cuda.is_available() else False
time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
out_dir = os.path.join(OUTPUT_DIR, time_stamp)
if not os.path.exists(out_dir):
    os.mkdir(out_dir)
csv_path = os.path.join(out_dir ,'result_stats.csv')
training_csv_path = os.path.join(out_dir, 'training_stats.csv')

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--extra', type=int, default=15)
    parser.add_argument('--burnout', type=int, default=15)
    parser.add_argument('--lr-schedule', type=str, default='linear_cycle')
    parser.add_argument('--lr-init', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=64) #TODO try other values
    parser.add_argument('--seed', type=int, default=789)
    parser.add_argument('--download', action="store_true")
    parser.add_argument('--lr_resume', type=float, default=0.0015)
    parser.add_argument('--out_dir', type=str, default=OUTPUT_DIR, help='Path to save model and results')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file')
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--use_half', default=True, type=lambda x: (str(x).lower() == 'true'),   help='use half ..')



    return parser.parse_args()

args = parse_args()

set_seeds(args.seed)



# --
# IO

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

print('cifar10.py: making dataloaders...', file=sys.stderr)

cifar10_stats = {
    "mean" : (0.4914, 0.4822, 0.4465),
    "std"  : (0.24705882352941178, 0.24352941176470588, 0.2615686274509804),
}


def create_my_sub_aug_list():
    sub_aug_list = []
    sub_aug_list.append(sub0())
    sub_aug_list.append(sub1())
    sub_aug_list.append(sub2())
    sub_aug_list.append(sub3())
    sub_aug_list.append(sub4())
    sub_aug_list.append(sub5())
    sub_aug_list.append(sub6())
    sub_aug_list.append(sub7())
    sub_aug_list.append(sub8())
    return sub_aug_list


def do_sub_aug_best_policies(img):


    policie = random.choice(list_of_policies)

    sub_aug_to_do = random.choice(policie)

    img = sub_aug_to_do.aug(img)


    return img



def get_transforms():

    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: np.asarray(x)),
        transforms.Lambda(lambda x: Image.fromarray(x)),

        ###choosen augmantation####
        transforms.Lambda(lambda x: do_sub_aug_best_policies(x)),
        ###########################
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_stats['mean'], cifar10_stats['std']),
    ])

    return transform_train , transform_test


def get_dataloaders(transform_train , transform_test):

    try:
        trainset = datasets.CIFAR10(root='./data', train=True, download=args.download, transform=transform_train)
        testset  = datasets.CIFAR10(root='./data', train=False, download=args.download, transform=transform_test)
    except:
        raise Exception('cifar10.py: error loading data -- try rerunning w/ `--download` flag')

    random_seed = 1000
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)



    train_idx = indices

    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )


    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=512,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    dataloaders = {
        "train" : trainloader,
        "test"  : testloader,
    }

    return dataloaders



checkpoint_path = os.path.join(args.out_dir, 'checkpoint.pth.tar')

print('cifar10.py: initializing model...', file=sys.stderr)

def get_model():

    if (USE_CUDA):
        if args.use_half:
            model = ResNet18(use_half=args.use_half).cuda().half()
        else:
            model = ResNet18(use_half=args.use_half).cuda()
    else:
        if args.use_half:
            model = ResNet18(use_half=args.use_half).half()
        else:
            model = ResNet18(use_half=args.use_half)



    if args.resume:
        checkpoint_file = args.resume
        if os.path.isfile(checkpoint_file):
            print("loading checkpoint {}".format(args.resume))
            checkpoint = torch.load(checkpoint_file)
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded checkpoint {} )".format(checkpoint_file ))


    # --
    # Initialize optimizer

    print('cifar10.py: initializing optimizer...', file=sys.stderr)



    lr_scheduler = getattr(LRSchedule, args.lr_schedule)(lr_init=args.lr_init, epochs=args.epochs, extra=args.extra)
    model.init_optimizer(
        opt=torch.optim.SGD,
        params=model.parameters(),
        lr_scheduler=lr_scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True    )

    return model



# --
# Train
print('cifar10.py: training...', file=sys.stderr)

with open(csv_path, 'w') as f:
    f.write('epoch,augmentation,aug_prob,aug_mag_scale,train_loss,test_acc,dur\n')


##best 3 policies choosen by FastAutoAugment
policie1 = [sub8(),sub0(),sub7(),sub4(),sub1()]
policie2 = [sub8(),sub5(),sub3(),sub4(),sub6()]
policie3 = [sub8(),sub1(),sub2(),sub3(),sub7()]
list_of_policies = [policie1,policie2,policie3]


transform_train, transform_test = get_transforms()
dataloaders = get_dataloaders(transform_train, transform_test)


model = get_model()
t = time()
for epoch in range(args.epochs + args.extra + args.burnout):
    train = model.train_epoch(dataloaders, mode='train')

    with open(training_csv_path, 'a') as f:
        f.write('{},{},{},{} \n'.format(epoch, model.lr, float(train['loss']),  time() - t))

    print(json.dumps({
        "epoch"     : int(epoch),
        "lr"        : model.lr,
        "train_loss": float(train['loss']),
        "time"      : time() - t,
    }))
    sys.stdout.flush()

    if epoch > args.epochs - 5:

        test = model.eval_epoch(dataloaders, 'test')
        print(json.dumps({
            "test_acc"  : float(test['acc']),
            "loss"      : float(test['loss']),
            "time": time() - t,
        }))


        with open(csv_path, 'a') as f: #use for new file  with open(csv_path, 'w') as f:
            f.write('{},{},{},{} \n'.format(epoch, float(train['loss']), float(test['acc']), time() - t))




torch.save({'state_dict': model.state_dict()}, checkpoint_path)


