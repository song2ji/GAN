# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""AttGAN, generator, and discriminator."""

import torch
import torch.nn as nn
from nn import LinearBlock, Conv2dBlock, ConvTranspose2dBlock, MiddleBlock, DownBlock
from torchsummary import summary
import numpy as np
import torchsnooper
from torchvision import models
from loss import StyleLoss, PerceptualLoss
# This architecture is for images of 128x128
# In the original AttGAN, slim.conv2d uses padding 'same'
MAX_DIM = 64 * 16  # 1024

class VGG16FeatureExtractor(nn.Module):#vgg特征提取网络
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False#固定编码器

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class Generator(nn.Module):#生成器
    def __init__(self, enc_dim=64, enc_layers=3, enc_norm_fn='batchnorm', enc_acti_fn='lrelu',
                 dec_dim=64, dec_layers=3, dec_norm_fn='batchnorm', dec_acti_fn='relu',
                 n_attrs=4, shortcut_layers=1, inject_layers=0, img_size=128):
        super(Generator, self).__init__()
        self.shortcut_layers = min(shortcut_layers, dec_layers - 1)#?
        self.inject_layers = min(inject_layers, dec_layers - 1)#?
   #     self.f_size = img_size // 2**(enc_layers+1)  # f_size = 4 for 128x128
        self.f_size = 4
        print(self.f_size)
        layers = []
        n_in = 3
        for i in range(enc_layers):#设置编码层
            n_out = min(enc_dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=enc_norm_fn, acti_fn=enc_acti_fn
            )]
            n_in = n_out
        self.enc_layers = nn.ModuleList(layers)
        self.up_att = nn.Upsample(scale_factor=16, mode='bicubic', align_corners=True)#上采样
        self.re_att = nn.ReLU() #F.interpolate(z,size=[128,128], mode='nearest')#relu层

        layers = []
        n_in = n_in #n_attrs  # 1024 + 13
        for i in range(dec_layers):
            if i < dec_layers - 1:
                n_out = min(dec_dim * 2**(dec_layers-i-2), MAX_DIM)
                layers += [ConvTranspose2dBlock(
                    n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=dec_norm_fn, acti_fn=dec_acti_fn
                )]
                n_in = n_out
                #n_in = n_in + n_in//2 if self.shortcut_layers > i else n_in
                #n_in = n_in + n_attrs if self.inject_layers > i else n_in
            else:
                layers += [ConvTranspose2dBlock(
                    n_in, 3, (4, 4), stride=2, padding=1, norm_fn='none', acti_fn='tanh'
                )]
        self.dec_layers = nn.ModuleList(layers)
        self.F = Extractors()
        self.T = Translator()

    @torchsnooper.snoop()
    def encode(self, x):
        # x.size=[32,3,64,64]
        z = x
        # print('img size ')
        # print(x.shape)
        zs = []
        for layer in self.enc_layers:
            z = layer(z)
            # print('encode size of each layer')
            # print(z.shape)
            zs.append(z)
        # h2.size=[32,256,4,4]

        # 4-attr
        """
        box = []
        for i in range(4):#4个属性？
            num_chs = h2.size(1)
            per_chs = float(num_chs) / 4
            start = int(np.rint(per_chs * i))
            end = int(np.rint(per_chs * (i + 1)))
            temp = h2.narrow(1, start, end-start)#提取维度1的数据
            # temp.size=[32,64,4,4]

            att_mean = temp.reshape(-1, 64, 8, 8).mean(1)#64*8*8
            # att_mean.size=[8,8,8]
            segmap = att_mean.view(-1, 1, 8, 8).repeat(1, 3, 1, 1)
            # segmap.size=[8,3,8,8]
            segmap = self.re_att(segmap)
            # segmap = torch.sign(segmap)
            segmap = F.interpolate(segmap,size=[128,128], mode='nearest')
            # F.interpolate利用插值方法，对输入的张量数组进行上\下采样操作, size为输出空间大小
            box.append(segmap)
        re = torch.cat(box, dim=1)
        """


        # 1-attr
        re=1
        return re, z#编码到
    #@torchsnooper.snoop()
    def decode(self, z):
        #a_tile = a.view(a.size(0), -1, 1, 1).repeat(1, 1, self.f_size, self.f_size)
        #z = zs[-1]#torch.cat([zs[-1], a_tile], dim=1)
            
        for i, layer in enumerate(self.dec_layers):
            z = layer(z)
            #if self.shortcut_layers > i:  # Concat 1024 with 512
                #z = torch.cat([z, zs[len(self.dec_layers) - 2 - i]], dim=1)
        return z

    def extract(self, x):
        return self.F(x)

    def translate(self, gen2, s):
        return self.T(gen2, s)
    
    def forward(self, x, a=None, mode='enc'):
        if mode == 'enc':
            return self.encode(x)
        if mode == 'dec':
            #assert a is not None, 'No given attribute.'
            z = self.decode(x)
            return z
        raise Exception('Unrecognized mode: ' + mode)




class Discriminators1(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=3, img_size=128):
        super(Discriminators1, self).__init__()
    #    self.f_size = img_size // 2**n_layers #4
        self.f_size = 4
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2**i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        print(self.conv)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none') #对抗
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            # 4-attr
            # LinearBlock(fc_dim, 4, 'none', 'none')#判别
            # 1-attr
            LinearBlock(fc_dim, 1, 'none', 'none')  # 判别
        )
    
    def forward(self,img_m):
        img_z_m = self.conv(img_m)
        # size[32,1024,4,4]
        img_z_m = img_z_m.view(img_z_m.size(0), -1)
        # size[32,16384]s
        return self.fc_adv(img_z_m), self.fc_cls(img_z_m)

class Discriminators2(nn.Module):
    # No instancenorm in fcs in source code, which is different from paper.
    def __init__(self, dim=64, norm_fn='instancenorm', acti_fn='lrelu',
                 fc_dim=1024, fc_norm_fn='none', fc_acti_fn='lrelu', n_layers=3, img_size=128):
        super(Discriminators2, self).__init__()
        #self.f_size = img_size // 2 ** n_layers  # 4
        self.f_size=4
        layers = []
        n_in = 3
        for i in range(n_layers):
            n_out = min(dim * 2 ** i, MAX_DIM)
            layers += [Conv2dBlock(
                n_in, n_out, (4, 4), stride=2, padding=1, norm_fn=norm_fn, acti_fn=acti_fn
            )]
            n_in = n_out
        self.conv = nn.Sequential(*layers)
        self.fc_adv = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            LinearBlock(fc_dim, 1, 'none', 'none')  # 对抗
        )
        self.fc_cls = nn.Sequential(
            LinearBlock(1024 * self.f_size * self.f_size, fc_dim, fc_norm_fn, fc_acti_fn),
            # 4-attr
            # LinearBlock(fc_dim, 4, 'none', 'none')#判别
            # 1-attr
            LinearBlock(fc_dim, 1, 'none', 'none')  # 判别
        )

    def forward(self,img):
        img = self.conv(img)
        img = img.view(img.size(0), -1)
        return self.fc_adv(img)


class Extractors(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [64, 128, 256, 512, 1024, 2048]
        self.style_dim=256
        # extractors:
        # # No normalization (Tag-specific)
        # channels: [64, 128, 256, 512, 1024, 2048]
        self.model = nn.Sequential(
            nn.Conv2d(3, self.channels[0], 1, 1, 0),
            *[DownBlock(self.channels[i], self.channels[i + 1]) for i in range(len(self.channels) - 1)],
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels[-1], self.style_dim, 1, 1, 0),  # 通道数为风格*标签种类
        )
    def forward(self, x):
        s = self.model(x).view(x.size(0), 1, -1)
        return s[:,0]


class Translator(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels = [128, 128, 128, 128, 128, 128, 128, 128]
        # translators:
        # # Adaptive Instance Normalization (Tag-specific)
        # channels: [64, 64, 64, 64, 64, 64, 64, 64]
        self.gen2channel=128
        self.style_dim =256
        self.model = nn.Sequential(
            nn.Conv2d(self.gen2channel, self.channels[0], 1, 1, 0),
            *[MiddleBlock(self.channels[i], self.channels[i + 1]) for i in range(len(self.channels) - 1)]
        )
        # 从编码器最后一个通道维数开始卷积

        self.style_to_params = nn.Linear(self.style_dim, self.get_num_adain_params(self.model))
        print(self.style_to_params)
        # 风格维数————需要自适应归一化的维数，从self.model块所需要的获取

        # self.features = nn.Sequential(
        #     nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
        # )
        #
        # self.masks = nn.Sequential(
        #     nn.Conv2d(channels[-1], hyperparameters['decoder']['channels'][0], 1, 1, 0),
        #     nn.Sigmoid()
        # )

    def forward(self, gen2,s):
        # print('s:')
        # print(s.shape)
       #  print(gen2.shape)
       # print(s.shape)
        p = self.style_to_params(s)  # 将风格维数转化为自适应输入的参数？

        self.assign_adain_params(p, self.model)

        gen2_add_style = self.model(gen2)

        return gen2_add_style

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                m.bias = adain_params[:, :m.num_features].contiguous().view(-1, m.num_features, 1)
                m.weight = adain_params[:, m.num_features:2 * m.num_features].contiguous().view(-1, m.num_features, 1) + 1
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ in ["AdaptiveInstanceNorm2d"]:
                num_adain_params += 2 * m.num_features
        return num_adain_params

import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim


# multilabel_soft_margin_loss = sigmoid + binary_cross_entropy

class AttGAN():
    def __init__(self, args):
        self.mode = args.mode
        self.gpu = args.gpu
        self.multi_gpu = args.multi_gpu if 'multi_gpu' in args else False
        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2
        self.lambda_3 = args.lambda_3
        self.lambda_gp = args.lambda_gp
        
        self.G = Generator(
            args.enc_dim, args.enc_layers, args.enc_norm, args.enc_acti,
            args.dec_dim, args.dec_layers, args.dec_norm, args.dec_acti,
            args.n_attrs, args.shortcut_layers, args.inject_layers, args.img_size
        )
        self.G.train()
        if self.gpu: self.G.cuda()
        #  summary(self.G, [(3, args.img_size, args.img_size), (args.n_attrs, 1, 1)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        self.D1 = Discriminators1(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D2 = Discriminators2(
            args.dis_dim, args.dis_norm, args.dis_acti,
            args.dis_fc_dim, args.dis_fc_norm, args.dis_fc_acti, args.dis_layers, args.img_size
        )
        self.D1.train()
        self.D2.train()
        if self.gpu: self.D1.cuda()
        if self.gpu: self.D2.cuda()
    #    summary(self.D, [(3, args.img_size, args.img_size)], batch_size=4, device='cuda' if args.gpu else 'cpu')
        
        # if self.multi_gpu:
        #     self.G = nn.DataParallel(self.G)
        #     self.D = nn.DataParallel(self.D)
        
        self.optim_G = optim.Adam(self.G.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D1 = optim.Adam(self.D1.parameters(), lr=args.lr, betas=args.betas)
        self.optim_D2 = optim.Adam(self.D2.parameters(), lr=args.lr, betas=args.betas)
    
    def set_lr(self, lr):
        for g in self.optim_G.param_groups:
            g['lr'] = lr
        for g in self.optim_D1.param_groups:
            g['lr'] = lr
        for g in self.optim_D2.param_groups:
            g['lr'] = lr

    def classify(self, zs, a):#打标签
        h1, h2 = torch.split(zs, 128, dim=1)
        # box = []

        # 4-attri
        """
        for i in range(a.size(1)):
            # a.size=[32,4]
            # h2.size=[32,256,4,4]
            num_chs = h2.size(1)
            per_chs = float(num_chs) / 4
            start = int(np.rint(per_chs * i))
            end = int(np.rint(per_chs * (i + 1)))
            temp = h2.narrow(1, start, end-start) #temp=[32,64,4,4]
            # x.narrow(0, 1, 3):沿第0轴方向的第1个元素开始切片（第0轴维度大小为5）取3个元素
            av = a.view(a.size(0), -1, 1, 1) #a.size=[32,4]
            # av.size=[32,4,1,1]
            ai =av[:,i,:,:]
            # ai.size=[32,1,1,1]
            ai = torch.unsqueeze(ai, 1)
            # ai.size=[32,1,1,1,1]
            tar_i =ai.repeat(1, 64,  h2.size(2), h2.size(2)) #h2.size(2)=4
            # tar_i.shape=[32,64,4,4]
            box.append(torch.mul(tar_i, temp))#打标签
            # torch.mul(a, b)是矩阵a和b对应位相乘，a和b的维度必须相等
        re = torch.cat(box, dim=1)
        # cat是将两个张量（tensor）拼接在一起，cat是concatenate的意思, dim=0按行拼接，dim=1按列拼接
        """

        # 1-attri
        ai = a.view(len(a), 1, 1, 1)
        # ai.size = [32, 1, 1, 1]
        # h2.size = [32, 256, 4, 4]
        tar_i = ai.repeat(1, 128, 16, 16)
        # tar_i.size = [32, 256, 4, 4]

        re = torch.mul(tar_i, h2)
        z = torch.cat([h1, re], dim=1)
        return z,re

    def diffatt(self, re_a, re_b, att_a, att_b, index):
        num_chs = re_a.size(1)
        per_chs = float(num_chs) / 4
        box = []
        for i in range(att_a.size(1)):
            start = int(np.rint(per_chs * i))
            end = int(np.rint(per_chs * (i + 1)))
            temp_a = re_a.narrow(1, start, end-start)
            temp_b = re_b.narrow(1, start, end-start)
            if i != index :
                #box.append(torch.zeros_like(temp_b))
                box.append(temp_a)
            else:
                box.append(temp_b)
        z = torch.cat(box, dim=1)		
        return z

    def trainG(self, img_a_m, img_b_m, img_a, img_b,att_a, att_a_, att_b, att_b_, mask):
        for p in self.D1.parameters():#关闭判别器
            p.requires_grad = False
        # for p in self.D2.parameters():  # 关闭判别器
        #     p.requires_grad = False

        _, zs_a = self.G(img_a_m, mode='enc')#编码
        _, zs_b = self.G(img_b_m, mode='enc')#编码
        h1_a, h2_a = torch.split(zs_a, 128, dim=1)
        h1_b, h2_b = torch.split(zs_b, 128, dim=1)
        # z_b, gen2_b = self.classify(zs_b, att_b)#z新生成的码，gen2_为标签处理后的块
        # z_a, gen2_a = self.classify(zs_a, att_a)

        # s_a = self.G.extract(img_a_m)
        # s_b = self.G.extract(img_b_m)

        # s_a_att = self.classify(s_a, att_a)
        # s_b_att = self.classify(s_b, att_b)

        z_a, gen2_a = self.classify(zs_a, att_a)
        z_b, gen2_b = self.classify(zs_b, att_b)

        # 重构
        # gen2_a=self.G.translate(h2_a,s_a_att)
        # gen2_b=self.G.translate(h2_b,s_b_att)



        h_a1a2 = torch.cat([h1_a, gen2_a], dim=1)
        h_b1b2 = torch.cat([h1_b, gen2_b], dim=1)

        img_recon_b_m  = self.G(h_a1a2, mode='dec')
        img_recon_a_m  = self.G(h_b1b2, mode='dec')

        # 迁移
        h_a1b2 = torch.cat([h1_a, gen2_b], dim=1)
        h_b1a2 = torch.cat([h1_b, gen2_a], dim=1)

        img_fake_a_m = self.G(h_a1b2, mode='dec')
        img_fake_b_m = self.G(h_b1a2, mode='dec')

        img_fake_a = img_fake_a_m + img_a-  mask * img_a
        img_fake_b = img_fake_b_m + img_b - mask * img_b

        d_a_fake, dc_a_fake = self.D1(img_fake_a_m)
        d_b_fake, dc_b_fake = self.D1(img_fake_b_m)

        # d_a_fake_total = self.D2(img_fake_a)
        # d_b_fake_total = self.D2(img_fake_b)

        s_a_fake = self.G.extract(img_fake_a_m)
        s_b_fake = self.G.extract(img_fake_b_m)

        dc_a_fake = torch.squeeze(dc_a_fake, 1)
        dc_b_fake = torch.squeeze(dc_b_fake, 1)

        if self.mode == 'wgan':
            gf_loss = -d_a_fake.mean() - d_b_fake.mean()
            # gf_total_loss= -d_a_fake_total.mean()-d_b_fake_total.mean()

        gc_loss = F.binary_cross_entropy_with_logits(dc_a_fake, att_b) + F.binary_cross_entropy_with_logits(dc_b_fake, att_a)
        gr_loss = F.l1_loss(img_recon_a_m, img_a_m)+  F.l1_loss(img_recon_b_m, img_b_m)#+ F.l1_loss(img_recon_b, img_b)
        # gs_loss = F.l1_loss(s_a_fake, s_a)+F.l1_loss(s_b_fake, s_b)

        '''
        if True:
            vgg_style = StyleLoss().cuda()
            vgg_content = PerceptualLoss().cuda()
            vgg_loss_style = vgg_style(img_recon_a, img_a)*250  
            vgg_loss_content = vgg_content(img_recon_a, img_a)*5
            vgg_loss = vgg_loss_style + vgg_loss_content
        else:
            vgg_loss = 0
        loss_app_per = 0.0
        vgg_output = VGG16FeatureExtractor().cuda()
        vgg_gt = VGG16FeatureExtractor().cuda()
        feat_output = vgg_output(img_recon_a)
        feat_gt = vgg_gt(img_a) 
        for i in range(3):
            loss_app_per += F.l1_loss(feat_output[i], feat_gt[i])
        '''
        # g_loss = gf_loss+ gf_total_loss + 200 * gr_loss + self.lambda_2 * gc_loss
        g_loss = gf_loss + 200 * gr_loss + self.lambda_2 * gc_loss
        
        self.optim_G.zero_grad()
        g_loss.backward()
        self.optim_G.step()
        errG = {
            'g_loss': g_loss.item(), 'gf_loss': gf_loss.item(),
            'gc_loss': gc_loss.item(), 'gr_loss': gr_loss.item()
            #'vgg_loss': vgg_loss.item()
        }
        return errG


    def trainD1(self, img_a_m, img_b_m, img_a, img_b,att_a, att_a_, att_b, att_b_, mask):
        for p in self.D1.parameters():  # 打开判别器c
            p.requires_grad = True

        _, zs_a = self.G(img_a_m, mode='enc')  # 编码
        _, zs_b = self.G(img_b_m, mode='enc')  # 编码

        h1_a, h2_a = torch.split(zs_a, 128, dim=1)#属性分割
        h1_b, h2_b = torch.split(zs_b, 128, dim=1)#属性风格

        # s_a = self.G.extract(img_a_m)
        # s_b = self.G.extract(img_b_m)

        z_a, gen2_a = self.classify(zs_a, att_a)
        z_b, gen2_b = self.classify(zs_b, att_b)

        # gen2_a = self.G.translate(h2_a, s_a_att)
        # gen2_b = self.G.translate(h2_b, s_b_att)

        h_a1b2 = torch.cat([h1_a, gen2_b], dim=1)
        h_b1a2 = torch.cat([h1_b, gen2_a], dim=1)

        img_fake_a_m = self.G(h_a1b2, mode='dec')
        img_fake_b_m = self.G(h_b1a2, mode='dec')

        img_fake_a = img_fake_a_m + img_a - mask * img_a
        img_fake_b = img_fake_b_m + img_b - mask * img_b

        d_a_fake, dc_a_fake = self.D1(img_fake_a_m)
        d_b_fake, dc_b_fake = self.D1(img_fake_b_m)

        d_real_a, dc_real_a = self.D1(img_a_m)
        d_real_b, dc_real_b = self.D1(img_b_m)

        def gradient_penalty(f, real, fake=None):
            def interpolate(a, b=None):
                if b is None:  # interpolation in DRAGAN
                    beta = torch.rand_like(a)
                    b = a + 0.5 * a.var().sqrt() * beta
                alpha = torch.rand(a.size(0), 1, 1, 1)
                alpha = alpha.cuda() if self.gpu else alpha
                inter = a + alpha * (b - a)
                return inter
            x = interpolate(real, fake).requires_grad_(True)
            pred = f(x)
            if isinstance(pred, tuple):
                pred = pred[0]
            grad = autograd.grad(
                outputs=pred, inputs=x,
                grad_outputs=torch.ones_like(pred),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            grad = grad.view(grad.size(0), -1)
            norm = grad.norm(2, dim=1)
            gp = ((norm - 1.0) ** 2).mean()
            return gp
        
        if self.mode == 'wgan':
            wd = d_real_a.mean() + d_real_b.mean() - d_a_fake.mean() - d_b_fake.mean()#对抗
            df_loss = -wd
            df_gp = gradient_penalty(self.D1, img_a_m, img_fake_a_m) + gradient_penalty(self.D1,img_b_m, img_fake_b_m)

        dc_real_a = torch.squeeze(dc_real_a, 1)
        dc_real_b = torch.squeeze(dc_real_b, 1)
        # print(dc_real_a.shape)
        # print(att_a.shape)
        dc_loss = F.binary_cross_entropy_with_logits(dc_real_a, att_a) + F.binary_cross_entropy_with_logits(dc_real_b, att_b)
        d_loss = df_loss + self.lambda_gp * df_gp + self.lambda_3 * dc_loss



        self.optim_D1.zero_grad()
        d_loss.backward()
        self.optim_D1.step()
        
        errD1 = {
            'd_loss': d_loss.item(), 'df_loss': df_loss.item(), 
            'df_gp': df_gp.item(), 'dc_loss': dc_loss.item()
        }
        return errD1


    """
    
    def trainD2(self, img_a_m, img_b_m, img_a, img_b, att_a, att_a_, att_b, att_b_, mask):
        for p in self.D2.parameters():  # 打开判别器
            p.requires_grad = True

        _, zs_a = self.G(img_a_m, mode='enc')  # 编码
        _, zs_b = self.G(img_b_m, mode='enc')  # 编码

        h1_a, h2_a = torch.split(zs_a, 128, dim=1)  # 属性分割
        h1_b, h2_b = torch.split(zs_b, 128, dim=1)  # 属性风格

        z_a, gen2_a = self.classify(zs_a, att_a)
        z_b, gen2_b = self.classify(zs_b, att_b)
        # gen2_a = self.G.translate(h2_a, s_a_att)
        # gen2_b = self.G.translate(h2_b, s_b_att)

        h_a1b2 = torch.cat([h1_a, gen2_b], dim=1)
        h_b1a2 = torch.cat([h1_b, gen2_a], dim=1)

        img_fake_a_m = self.G(h_a1b2, mode='dec')
        img_fake_b_m = self.G(h_b1a2, mode='dec')

        img_fake_a = img_fake_a_m + img_a - mask * img_a
        img_fake_b = img_fake_b_m + img_b - mask * img_b

        d_a_fake = self.D2(img_fake_a)
        d_b_fake = self.D2(img_fake_b)

        d_real_a= self.D2(img_a)
        d_real_b= self.D2(img_b)

        if self.mode == 'wgan':
            wd = d_real_a.mean() + d_real_b.mean() - d_a_fake.mean() - d_b_fake.mean()  # 对抗
            df_loss = -wd

        # print(dc_real_a.shape)
        # print(att_a.shape)
        d_loss = 100*df_loss

        self.optim_D2.zero_grad()
        d_loss.backward()
        self.optim_D2.step()

        errD2 = {
            'df_all_loss': df_loss.item()
        }
        return errD2
    """
    
    def train(self):
        self.G.train()
        self.D1.train()
        # self.D2.train()
    
    def eval(self):
        self.G.eval()
        self.D1.eval()
        # self.D2.eval()
    
    def save(self, path):
        states = {
            'G': self.G.state_dict(),
            'D1': self.D1.state_dict(),
            # 'D2': self.D2.state_dict(),
            'optim_G': self.optim_G.state_dict(),
            'optim_D1': self.optim_D1.state_dict(),
            # 'optim_D2': self.optim_D2.state_dict()
        }
        torch.save(states, path)
    
    def load(self, path):
        states = torch.load(path, map_location=lambda storage, loc: storage)
        if 'G' in states:
            self.G.load_state_dict(states['G'])
        if 'D1' in states:
            self.D1.load_state_dict(states['D1'])
        # if 'D2' in states:
        #    self.D2.load_state_dict(states['D2'])
        if 'optim_G' in states:
            self.optim_G.load_state_dict(states['optim_G'])
        if 'optim_D1' in states:
            self.optim_D1.load_state_dict(states['optim_D1'])
        # if 'optim_D1' in states:
        #     self.optim_D2.load_state_dict(states['optim_D2'])
    
    def saveG(self, path):
        states = {
            'G': self.G.state_dict()
        }
        torch.save(states, path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', dest='img_size', type=int, default=128)
    parser.add_argument('--shortcut_layers', dest='shortcut_layers', type=int, default=1)
    parser.add_argument('--inject_layers', dest='inject_layers', type=int, default=0)
    parser.add_argument('--enc_dim', dest='enc_dim', type=int, default=64)
    parser.add_argument('--dec_dim', dest='dec_dim', type=int, default=64)
    parser.add_argument('--dis_dim', dest='dis_dim', type=int, default=64)
    parser.add_argument('--dis_fc_dim', dest='dis_fc_dim', type=int, default=1024)
    parser.add_argument('--enc_layers', dest='enc_layers', type=int, default=5)
    parser.add_argument('--dec_layers', dest='dec_layers', type=int, default=5)
    parser.add_argument('--dis_layers', dest='dis_layers', type=int, default=5)
    parser.add_argument('--enc_norm', dest='enc_norm', type=str, default='batchnorm')
    parser.add_argument('--dec_norm', dest='dec_norm', type=str, default='batchnorm')
    parser.add_argument('--dis_norm', dest='dis_norm', type=str, default='instancenorm')
    parser.add_argument('--dis_fc_norm', dest='dis_fc_norm', type=str, default='none')
    parser.add_argument('--enc_acti', dest='enc_acti', type=str, default='lrelu')
    parser.add_argument('--dec_acti', dest='dec_acti', type=str, default='relu')
    parser.add_argument('--dis_acti', dest='dis_acti', type=str, default='lrelu')
    parser.add_argument('--dis_fc_acti', dest='dis_fc_acti', type=str, default='relu')
    parser.add_argument('--lambda_1', dest='lambda_1', type=float, default=100.0)
    parser.add_argument('--lambda_2', dest='lambda_2', type=float, default=10.0)
    parser.add_argument('--lambda_3', dest='lambda_3', type=float, default=1.0)
    parser.add_argument('--lambda_gp', dest='lambda_gp', type=float, default=10.0)
    parser.add_argument('--mode', dest='mode', default='wgan', choices=['wgan', 'lsgan', 'dcgan'])
    parser.add_argument('--lr', dest='lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5)
    parser.add_argument('--beta2', dest='beta2', type=float, default=0.999)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    args.n_attrs = 13
    args.betas = (args.beta1, args.beta2)
    attgan = AttGAN(args)
