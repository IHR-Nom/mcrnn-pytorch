import argparse
import json

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

import dataset
import models.crnn as crnn
from conf import params
from utils import utils

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_path', type = str, required = True, help = 'crnn model path')
parser.add_argument('-a', '--alphabet_path', type = str, required = True, help = 'alphabet file')
parser.add_argument('-i', '--image_path', type = str, required = True, help = 'demo image path')
args = parser.parse_args()

model_path = args.model_path
image_path = args.image_path

# net init

with open(args.alphabet_path, encoding='utf-8') as f:
    alphabet = json.load(f)['alphabet']
nclass = len(alphabet) + 1
model = crnn.CRNN(params.img_h, params.nc, nclass, params.nh)
if torch.cuda.is_available():
    model = model.cuda()

# load model
print('loading pretrained model from %s' % model_path)
if params.multi_gpu:
    model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(model_path))

converter = utils.StrLabelConverter(alphabet)

transformer = dataset.ResizeNormalize((params.img_w, params.img_h))
image = Image.open(image_path).convert('RGB')
image = utils.format_image(np.array(image), width=params.img_w, height=params.img_h)
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

model.eval()
preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = Variable(torch.LongTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
