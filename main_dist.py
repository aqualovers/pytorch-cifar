'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.multiprocessing as mp
import torch.distributed as dist

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging
from tqdm import tqdm

import models
from utils import set_logger

from torch.cuda.amp import GradScaler, autocast


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=512, type=int, help='batch size')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--output_dir', required=True, type=str, help='output directory')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--workers', default=4, type=int,
                    help='number of workers for dataloader')

parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--amp", action='store_true', default=False)

    
best_acc = 0  # best test accuracy
    
def main():
    args = parser.parse_args()

    ngpus_per_node = torch.cuda.device_count()

    if args.dist:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(None, ngpus_per_node, args)

    
def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    
    if args.amp:
        args.scaler = GradScaler()
    
    if args.dist:
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        
        torch.cuda.set_device(args.gpu)
        
        if args.rank == 0:
            if not os.path.isdir(args.output_dir):
                os.makedirs(args.output_dir)

        torch.distributed.barrier()
        
    else:
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)
        
    set_logger(os.path.join(args.output_dir, 'train.log'))


    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)


    if args.dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        batch_size = int(args.batch_size / args.world_size)
        
        if args.batch_size % args.world_size > 0:
            logging.info("Batch size {} is not a multiple of number of total GPUS {}.".format(args.batch_size, args.world_size))
            logging.info("Batch size {} per GPU and total {} will be applied instead.".format(batch_size, batch_size * args.world_size))
        
    else:
        train_sampler = None
        batch_size = args.batch_size

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.workers
    )

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # Model
    print('==> Building model..')
    net = models.ResNet152(args.amp)
    net.cuda()
    #net.to(device)

    if args.dist:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[args.gpu]
        )
    else:
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        logging.info('Epoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        with tqdm(total=len(trainloader)) as t:
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.cuda(), targets.cuda()
                #inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                
                if args.amp:
                    with autocast():
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)

                        args.scaler.scale(loss).backward()
                        args.scaler.step(optimizer)
                        args.scaler.update()
                else:
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                t.set_postfix(loss='{:05.3f}'.format(train_loss/(batch_idx+1)), acc='{:05.3f}'.format(100.*correct/total))
                t.update()

        logging.info('Train Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            with tqdm(total=len(testloader)) as t:
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.cuda(), targets.cuda()
                    #inputs, targets = inputs.to(device), targets.to(device)
                    
                    
                    if args.amp:
                        with autocast():
                            outputs = net(inputs)
                            loss = criterion(outputs, targets)
                    else:
                        outputs = net(inputs)
                        loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    t.set_postfix(loss='{:05.3f}'.format(test_loss/(batch_idx+1)), acc='{:05.3f}'.format(100.*correct/total))
                    t.update()

        logging.info('Eval Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            logging.info("- Found new best accuracy")

            if not args.dist or args.rank == 0:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'acc': acc,
                    'epoch': epoch,
                }
                torch.save(state, os.path.join(args.output_dir, 'ckpt.pth'))

            best_acc = acc



    logging.info("Start training...")

    for epoch in range(start_epoch, args.epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
        
        
if __name__ == '__main__':
    main()
