from __future__ import division
from __future__ import print_function

import argparse
import os
import random

import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torch.nn import CTCLoss
from torch.utils.data import DataLoader

import dataset
from conf import params
from models import crnn
from utils import utils
from utils.augmentation import GridDistortion, ImgAugTransform
from utils.error_rates import cer, jaccard_similarity
from utils.utils import collect_alphabet

parser = argparse.ArgumentParser()
parser.add_argument('-train', '--train_root', required=True, help='path to train dataset')
parser.add_argument('-val', '--val_root', required=True, help='path to val dataset')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--display_interval', type=int, default=500, help='Display loss after every N iterate')
parser.add_argument('--val_interval', type=int, default=1000, help='Run validating after every N iterate')
parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs')
parser.add_argument('--model_out', type=str, required=True, help='path to model file for saving')
parser.add_argument('--alphabet_out', type=str, required=True, help='path to alphabet file for saving')
args = parser.parse_args()


# ensure everytime the random is the same
random.seed(params.manual_seed)
np.random.seed(params.manual_seed)
torch.manual_seed(params.manual_seed)

cudnn.benchmark = True


def data_loader():
    # train
    transform = torchvision.transforms.Compose([ImgAugTransform(), GridDistortion(prob=0.65)])
    train_dataset = dataset.LmdbDataset(root=args.train_root, transform=transform)
    assert train_dataset
    if not params.random_sample:
        sampler = dataset.RandomSequentialSampler(train_dataset, params.batchSize)
    else:
        sampler = None
    _train_loader = DataLoader(train_dataset, batch_size=params.batchSize, shuffle=True, sampler=sampler,
                               num_workers=int(params.workers),
                               collate_fn=dataset.AlignCollate(img_h=params.img_h, img_w=params.img_w))
    # val
    transform = torchvision.transforms.Compose([dataset.ResizeNormalize((params.img_w, params.img_h))])
    val_dataset = dataset.LmdbDataset(root=args.val_root, transform=transform)
    _val_loader = DataLoader(val_dataset, shuffle=True, batch_size=params.batchSize, num_workers=int(params.workers))
    
    return _train_loader, _val_loader


train_loader, val_loader = data_loader()
alphabet = collect_alphabet([args.train_root, args.val_root])
dl_model = crnn.net_init(alphabet, args.pretrained_model)
print(dl_model)

# -----------------------------------------------
"""
In this block
    Init some utils defined in utils.py
"""
# Compute average for `torch.Variable` and `torch.Tensor`.
loss_avg = utils.Averager()

# Convert between str and label.
converter = utils.StrLabelConverter(alphabet)

# -----------------------------------------------
"""
In this block
    criterion define
"""
criterion = CTCLoss()

# -----------------------------------------------
"""
In this block
    Init some tensor
    Put tensor and net on cuda
    NOTE:
        image, text, length is used by both val and train
        because train and val will never use it at the same time.
"""
image = torch.FloatTensor(params.batchSize, 3, params.img_h, params.img_h)
text = torch.LongTensor(params.batchSize * 5)
length = torch.LongTensor(params.batchSize)

if params.cuda and torch.cuda.is_available():
    criterion = criterion.cuda()
    image = image.cuda()
    text = text.cuda()

    dl_model = dl_model.cuda()
    if params.multi_gpu:
        dl_model = torch.nn.DataParallel(dl_model, device_ids=range(params.ngpu))

image = Variable(image)
text = Variable(text)
length = Variable(length)

# -----------------------------------------------
"""
In this block
    Setup optimizer
"""
if params.adam:
    optimizer = optim.Adam(dl_model.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
elif params.adadelta:
    optimizer = optim.Adadelta(dl_model.parameters())
else:
    optimizer = optim.RMSprop(dl_model.parameters(), lr=params.lr)

# -----------------------------------------------
"""
In this block
    Dealwith lossnan
    NOTE:
        I use different way to dealwith loss nan according to the torch version. 
"""
if params.dealwith_lossnan:
    if torch.__version__ >= '1.1.0':
        """
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.
        Pytorch add this param after v1.1.0 
        """
        criterion = CTCLoss(zero_infinity=True)
    else:
        """
        only when
            torch.__version__ < '1.1.0'
        we use this way to change the inf to zero
        """
        dl_model.register_backward_hook(dl_model.backward_hook)

# -----------------------------------------------


def val(model, criterion):
    print('Start val')
    model.eval()
    val_iter = iter(val_loader)

    n_correct = 0
    similarity = 0
    distances = 0
    count = 0.0
    val_loss_avg = utils.Averager()

    max_iter = len(val_loader)
    all_predicts = []
    for it in range(max_iter):
        data = val_iter.next()
        it += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.load_data(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.load_data(text, t)
        utils.load_data(length, l)

        preds = model(image)
        preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
        val_cost = criterion(preds, text, preds_size, length) / batch_size
        val_loss_avg.add(val_cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        cpu_texts_decode = []
        for t in cpu_texts:
            cpu_texts_decode.append(t.decode('utf-8', 'strict'))
        for pred, target in zip(sim_preds, cpu_texts_decode):
            if pred == target:
                n_correct += 1
            simr = jaccard_similarity([x for x in pred], [x for x in target])
            distance = cer(pred, target)
            all_predicts.append({'pred': pred, 'actual': target, 'similarity': simr, 'distant': distance})
            similarity += simr
            distances += distance
            count += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_val_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts_decode):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / count
    similarity = similarity / count
    distance = distances / count
    print('Val loss: %f, accuracy: %f, similarity: %f, distance: %f' % (val_loss_avg.val(), accuracy, similarity, distance))
    return accuracy, all_predicts


def train(model, criterion, optimizer, train_iter):
    model.train()

    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.load_data(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.load_data(text, t)
    utils.load_data(length, l)
    
    optimizer.zero_grad()
    preds = model(image)
    preds_size = Variable(torch.LongTensor([preds.size(0)] * batch_size))
    train_cost = criterion(preds, text, preds_size, length) / batch_size
    train_cost.backward()
    optimizer.step()
    return train_cost


if __name__ == "__main__":
    best_acc = 0
    if not os.path.exists(os.path.dirname(args.model_out)):
        os.makedirs(os.path.dirname(args.model_out))
    for epoch in range(args.n_epochs):
        train_iter = iter(train_loader)
        i = 0
        while i < len(train_loader):
            with torch.set_grad_enabled(True):
                cost = train(dl_model, criterion, optimizer, train_iter)
            loss_avg.add(cost)
            i += 1

            if i % args.display_interval == 0:
                print('[%d/%d][%d/%d] Loss: %f' % (epoch, args.n_epochs, i, len(train_loader), loss_avg.val()))
                loss_avg.reset()

            if i % args.val_interval == 0:
                with torch.no_grad():
                    acc, predicts = val(dl_model, criterion)
                if best_acc < acc:
                    best_acc = acc
                    torch.save(dl_model.state_dict(), args.model_out)
                    with open(args.alphabet_out, 'w', encoding='utf8') as f:
                        json.dump({'alphabet': alphabet}, f, ensure_ascii=False)

    print('Final acc: ' + str(best_acc))
