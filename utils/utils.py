#!/usr/bin/python
# encoding: utf-8
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class StrLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    
    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.
        """

        length = []
        result = []
        for item in text:            
            item = item.decode('utf-8','strict')
            length.append(len(item))
            r = []
            for char in item:
                index = self.dict[char]
                # result.append(index)
                r.append(index)
            result.append(r)
        
        max_len = 0
        for r in result:
            if len(r) > max_len:
                max_len = len(r)
        
        result_temp = []
        for r in result:
            for i in range(max_len - len(r)):
                r.append(0)
            result_temp.append(r)

        text = result_temp
        return torch.LongTensor(text), torch.LongTensor(length)

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.LongTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.LongTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.LongTensor([l]), raw=raw))
                index += l
            return texts


class Averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res


def one_hot(v, v_length, nc):
    batchSize = v_length.size(0)
    maxLength = v_length.max()
    v_onehot = torch.FloatTensor(batchSize, maxLength, nc).fill_(0)
    acc = 0
    for i in range(batchSize):
        length = v_length[i]
        label = v[acc:acc + length].view(-1, 1).long()
        v_onehot[i, :length].scatter_(1, label, 1.0)
        acc += length
    return v_onehot


def load_data(v, data):
    with torch.no_grad():
        v.resize_(data.size()).copy_(data)


def pretty_print(v):
    print('Size {0}, Type: {1}'.format(str(v.size()), v.data.type()))
    print('| Max: %f | Min: %f | Mean: %f' % (v.max().data[0], v.min().data[0],
                                              v.mean().data[0]))


def assure_ratio(img):
    """Ensure imgH <= imgW."""
    b, c, h, w = img.size()
    if h > w:
        main = nn.UpsamplingBilinear2d(size=(h, h), scale_factor=None)
        img = main(img)
    return img


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def collect_alphabet(dataset_paths):
    chars = []
    for dataset_path in dataset_paths:
        env = lmdb.open(dataset_path, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        if not env:
            print('cannot creat lmdb from %s' % dataset_path)
            sys.exit(0)
        with env.begin(write=False) as txn:
            n_samples = int(txn.get('num-samples'.encode('utf-8')))
            for i in range(n_samples):
                label_key = 'label-%09d' % (i + 1)
                label = txn.get(label_key.encode('utf-8'))
                label = label.decode('utf-8', 'strict')
                for char in label:
                    if char not in chars:
                        chars.append(char)
    return ''.join(chars)


def format_image(image, width, height):
    image = resize_filling(image, (height, width))
    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def resize_filling(image, new_size, color=None):
    n_width, n_height = new_size
    height, width = image.shape[:2]
    if width > n_width:
        ratio = n_width / width
        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
    blank_image = np.zeros((n_height, n_width, 3), np.uint8)
    if color is None:
        color = bincount_app(image)
    lower = np.array([color[0] - 20, color[1] - 20, color[2] - 20])
    upper = np.array([color[0] + 20, color[1] + 20, color[2] + 20])
    mask = cv2.inRange(image, lower, upper)
    masked_image = np.copy(image)
    masked_image[mask != 0] = color
    blank_image[:] = color

    x_offset, y_offset = int((n_width - width) / 2), 10
    # Here, y_offset+height <= blank_image.shape[0] and x_offset+width <= blank_image.shape[1]
    blank_image[y_offset:y_offset + height, x_offset:x_offset + width] = masked_image.copy()
    return blank_image


def bincount_app(a):
    a2D = a.reshape(-1,a.shape[-1])
    col_range = (256, 256, 256) # generically : a2D.max(0)+1
    a1D = np.ravel_multi_index(a2D.T, col_range)
    return np.unravel_index(np.bincount(a1D).argmax(), col_range)
