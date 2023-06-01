from util.misc import AverageMeter
from util.eval import accuracy
from util.datasets import load_dataset
from util.augment import *
import argparse
import logging
import os
import random
import sys
import time

import torch.backends.cudnn as cudnn
import torch.utils.data.dataloader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from model import alexnet
from torch.autograd import Variable
sys.path.append('./util')


def check_p(value: float):
    if value < 0 or value > 1:
        raise argparse.ArgumentTypeError("value must be 0-1")
    return value


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='PyTorch CIFAR Classifier')
parser.add_argument('--batch_size', type=int, default=128,
                    help="Every train dataset size.")
parser.add_argument('--lr', type=float, default=0.0001,
                    help="starting lr")
parser.add_argument('--epochs', type=int, default=200, help="Train loop")
parser.add_argument('--phase', type=str, default='train',
                    help="train or eval? Default:`train`")
parser.add_argument('--model_path', type=str,
                    default="./checkpoints/CIFAR100_baseline_epoch_201.pth", help="load model path.")
parser.add_argument('--augment', type=str, default='baseline',
                    help="augment method? choices: baseline, mixup, cutout, cutmix Default:`baseline`", choices=['baseline', 'mixup', 'cutout', 'cutmix'])
parser.add_argument('--alpha', type=int, default=1.0,
                    help='alpha for mixup or cutmix')
parser.add_argument('--p', type=check_p, default=0.5,
                    help="probability of cutting out or cutmix")
parser.add_argument('--maskout_size', type=int, default=8,
                    help="maskout_size for cutout")
args = parser.parse_args()
try:
    os.makedirs("./checkpoints")
except OSError:
    pass

# set random seed
manualSeed = random.randint(1, 10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

cudnn.benchmark = True

# setup gpu driver
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load datasets
train_dataloader, test_dataloader = load_dataset(args.batch_size)

# Load model
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(alexnet())
else:
    model = alexnet()
model.to(device)

# Loss function
criterion = torch.nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# log地址
logging.basicConfig(filename='./log/'+args.augment + 'logging.log', filemode='a+',
                    format="%(message)s", level=logging.INFO)
writer = SummaryWriter('./'+args.augment+'_log')


def train(model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        # get the inputs; data is a list of [inputs, labels]
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)
        if args.augment == 'mixup':
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,
                                                           args.alpha)
            inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))
            # compute output
            output = model(inputs)
            loss = mixup_criterion(
                criterion, output, targets_a, targets_b, lam)
            prec1_a, prec5_a = accuracy(output, targets_a, topk=(1, 5))
            prec1_b, prec5_b = accuracy(output, targets_b, topk=(1, 5))
            prec1, prec5 = lam*prec1_a + \
                (1-lam)*prec1_b, lam*prec5_a+(1-lam)*prec5_b
        elif args.augment == 'baseline':
            output = model(inputs)
            loss = criterion(output, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        elif args.augment == 'cutout':
            CutOut = cutout(args.maskout_size, args.p, False)
            inputs = CutOut(inputs)
            output = model(inputs)
            loss = criterion(output, targets)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        elif args.augment == 'cutmix':
            inputs, targets_a, targets_b, lam = cutmix(
                args.alpha, args.p, inputs, targets)
            output = model(inputs)
            loss = criterion(output, targets_a) * lam + \
                criterion(output, targets_b) * (1. - lam)
            prec1, prec5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradients in a backward pass
        optimizer.zero_grad()
        loss.backward()

        # Call step of optimizer to update model params
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    logging.info(f"Epoch [{epoch + 1}] [{batch_idx}/{len(train_dataloader)}]\t"
                 f"Time {data_time.val:.3f} ({data_time.avg:.3f})\t"
                 f"Loss {losses.avg:.4f}\t"
                 f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                 f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})")
    if epoch % 50 == 0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"./checkpoints/CIFAR100_{args.augment}_epoch_{epoch + 1}.pth")
    return losses.avg


def test(model):
    # switch to evaluate mode
    model.eval()
    # init value
    total = 0.
    correct = 0.
    with torch.no_grad():
        for _, data in enumerate(test_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def run():
    best_prec1 = 0.
    avg_loss = []
    prec = []
    for epoch in tqdm(range(args.epochs)):
        # train for one epoch
        logging.info(f"Begin Training Epoch {epoch + 1}")
        loss = train(model, criterion, optimizer, epoch)
        avg_loss.append(loss)
        # evaluate on validation set
        logging.info(f"Begin Validation @ Epoch {epoch + 1}")
        prec1 = test(model)
        prec.append(prec1)
        # remember best prec@1 and save checkpoint if desired
        best_prec1 = max(prec1, best_prec1)

        logging.info("Epoch Summary: ")
        logging.info(f"\tEpoch Accuracy: {prec1}")
        logging.info(f"\tBest Accuracy: {best_prec1}")

        writer.add_scalar('loss', loss, epoch)
        writer.add_scalar('prec', prec1, epoch)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f"./checkpoints/CIFAR100_{args.augment}_epoch_{args.epochs + 1}.pth")
    logging.info(avg_loss)
    logging.info(prec)


if __name__ == '__main__':
    if args.phase == "train":
        run()
    elif args.phase == "eval":
        if args.model_path != "":
            print("Loading model...")
            checkpoint = torch.load(
                args.model_path, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model_state_dict"])
            print("Loading model successful!")
            acc = test(model)
            print(
                f"Accuracy of the network on the 10000 test images: {acc:.2f}%.")
        else:
            print(
                "WARNING: You want use eval pattern, so you should add --model_path MODEL_PATH")
    else:
        print(args)
