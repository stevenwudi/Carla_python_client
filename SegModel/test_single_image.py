#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import json
import sys
import cv2

import torch
from torchvision import transforms
from torch.autograd import Variable

from SegModel import drn
from SegModel.DRN import DRNSeg
from SegModel.city_scape_info import CITYSCAPE_PALETTE

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
try:
    from modules import batchnormsync
except ImportError:
    pass


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-cmd', default='train', help='train or test')
    parser.add_argument('-d', '--data-dir', default='/home/public/CITYSCAPE')
    parser.add_argument('-c', '--classes', default=19, type=int)
    parser.add_argument('-s', '--crop-size', default=896, type=int)
    parser.add_argument('--step', type=int, default=200)
    parser.add_argument('--arch', default='drn_d_38', help='architecture')
    parser.add_argument('--batch-size', type=int, default=6, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr-mode', type=str, default='step', help='step or poly')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained',
                        default='./SegModel/drn_d_38_cityscapes.pth', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=16)
    parser.add_argument('--load-release', dest='load_rel', default=None)
    parser.add_argument('--phase', default='val')
    parser.add_argument('--random-scale', default=2, type=float)
    parser.add_argument('--random-rotate', default=10, type=int)
    parser.add_argument('--bn-sync', action='store_true')
    parser.add_argument('--ms', action='store_true',
                        help='Turn on multi-scale testing')
    parser.add_argument('--with-gt', action='store_true')
    parser.add_argument('--test-suffix', default='', type=str)
    args = parser.parse_args()

    assert args.data_dir is not None
    assert args.classes > 0

    print(' '.join(sys.argv))
    print(args)

    if args.bn_sync:
        drn.BatchNorm = batchnormsync.BatchNormSync

    return args


def test_single_image(image, MINI_WINDOW_HEIGHT, MINI_WINDOW_WIDTH):
    # argument parsing
    args = parse_args()

    # Information contains mean and std
    info = json.load(open('./SegModel/info.json', 'r'))

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=info['mean'], std=info['std'])
    ])

    image = data_transform(image)
    image = image.unsqueeze(0)

    # model definition
    single_model = DRNSeg(args.arch, args.classes, pretrained_model=None, pretrained=False)

    # Load pretrained weights
    checkpoint = torch.load(args.pretrained)
    single_model.load_state_dict(checkpoint)

    model = torch.nn.DataParallel(single_model).cuda()
    model.eval()

    image_var = Variable(image, requires_grad=False, volatile=True)
    final = model(image_var)[0]
    _, pred = torch.max(final, 1)
    pred = pred.cpu().data.numpy()

    pred_resize = cv2.resize(pred[0].astype('uint8'), (MINI_WINDOW_WIDTH, MINI_WINDOW_HEIGHT), interpolation=cv2.INTER_NEAREST)

    pred_color = CITYSCAPE_PALETTE[pred_resize.squeeze()]
    return pred[0], pred_color

