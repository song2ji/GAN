# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Entry point for testing AttGAN network."""

import argparse
import json
import os
from os.path import join
import torchsnooper
import torch
import torch.utils.data as data
import torchvision.utils as vutils

from attgan import AttGAN
from data import check_attribute_conflict
from helpers import Progressbar
from utils import find_model
from dataset import MultiCelebADataset

def parse(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', dest='experiment_name', required=True)
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--num_test', dest='num_test', type=int)
    parser.add_argument('--load_epoch', dest='load_epoch', type=str, default='latest')
    parser.add_argument('--custom_img', action='store_true')
    parser.add_argument('--custom_data', type=str, default='./data/custom')
    parser.add_argument('--custom_attr', type=str, default='./data/list_attr_custom.txt')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--multi_gpu', action='store_true')
    return parser.parse_args(args)

args_ = parse()
print(args_)

with open(join('output', args_.experiment_name, 'setting.txt'), 'r') as f:
    args = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

args.test_int = args_.test_int
args.num_test = args_.num_test
args.gpu = args_.gpu
args.load_epoch = args_.load_epoch
args.multi_gpu = args_.multi_gpu
args.custom_img = args_.custom_img
args.custom_data = args_.custom_data
args.custom_attr = args_.custom_attr
args.n_attrs = len(args.attrs)
args.betas = (args.beta1, args.beta2)

print(args)
output_path = join('output', args.experiment_name, 'sample_testing')
if args.data == 'CelebA':
    from data import CelebA
    test_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'test', args.attrs)
os.makedirs(output_path, exist_ok=True)
test_dataloader = data.DataLoader(
    test_dataset, batch_size=32, num_workers=args.num_workers,
    shuffle=False, drop_last=False
)

if args.num_test is None:
    print('Testing images:', len(test_dataset))
else:
    print('Testing images:', min(len(test_dataset), args.num_test))


attgan = AttGAN(args)
attgan.load(find_model(join('output', args.experiment_name, 'checkpoint'), args.load_epoch))
progressbar = Progressbar()

attgan.eval()

attrs_default = [
        'Bangs', 'Mouth_Slightly_Open',
        'Eyeglasses', 'Bushy_Eyebrows'
    ]
    
dataset = MultiCelebADataset(attrs_default)
#@torchsnooper.snoop()
for i in range(len(attrs_default)):
    img_a, att_a = next(dataset.gen(0, False))
    img_b, att_b = next(dataset.gen(0, True))
    img_a = img_a.cuda() if args.gpu else img_a
    att_a = att_a.cuda() if args.gpu else att_a
    att_a = att_a.type(torch.float)
    img_b = img_b.cuda() if args.gpu else img_b
    att_b = att_b.cuda() if args.gpu else att_b
    #att_b = att_b.type(torch.float)
    if 3 == 3:
        with torch.no_grad():
            for j in range(32):
                #ide = torch.randperm(len(att_a))
                #att_b = att_b[ide].contiguous()
                #img_b = img_b[ide]
                tar_j = img_b[j]
                tar_j = torch.unsqueeze(tar_j, 0)
                tar_j = tar_j.repeat(32, 1,  1, 1)
                tar_j = tar_j.type(torch.float)
                print(tar_j.shape)
                _, zs_a = attgan.G(img_a, mode='enc')
                _, zs_b = attgan.G(tar_j, mode='enc')
                h1_a, h2_a = torch.split(zs_a, 256, dim=1)
                h1_b, h2_b = torch.split(zs_b, 256, dim=1)
                z_b, g_b = attgan.classify(zs_b, att_b)
                z_a, g_a = attgan.classify(zs_a, att_a)

                gen2_b = attgan.diffatt(g_a, g_b, att_a, att_b, 2)
                gen2_a = attgan.diffatt(g_b, g_a, att_b, att_a, 2)
                h_a1b2 = torch.cat([h1_a, gen2_b], dim=1)
                h_b1a2 = torch.cat([h1_b, gen2_a], dim=1)
                img_fake_a = attgan.G(h_a1b2, mode='dec')
                img_fake_b = attgan.G(h_b1a2, mode='dec')
                viz_images = torch.stack([img_a,tar_j,img_fake_a], dim=1)
                viz_images = viz_images.view(-1, *list(img_a.size())[1:])
                vutils.save_image(viz_images,
                                  '%s/niter_%03d_%03d.png' % (output_path, i, j),
                                  nrow=3 * 4,
                                  normalize=True)


