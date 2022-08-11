import numpy as np
from collections import OrderedDict
import torchvision.models as models
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config
from .Swin_transformer import SwinTransformer
config = Config()






class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(inp_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(inp_dim / 2))
        self.conv2 = Conv(int(inp_dim / 2), int(inp_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(inp_dim / 2))
        self.conv3 = Conv(int(inp_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CBamSpatialAttention(nn.Module):
    def __init__(self,reduction = 2):
        super(CBamSpatialAttention,self).__init__()
        kernel_size = 7
        self.att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1)//2),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        out = self._PoolAlongChannel(x)
        out = self.att(out)
        out = torch.sigmoid(out)
        return x*out

    def _PoolAlongChannel(self,x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)

class Fusion_block(nn.Module):
    def __init__(self, ch_1, ch_2, reduction, ch_int, ch_out, drop_rate=0.):
        super(Fusion_block, self).__init__()

        # spatial attention for swin-transformer branch
        self.fc1 = nn.Conv2d(ch_2, ch_2 // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(ch_2 // reduction, ch_2, kernel_size=1)
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))


        # spatial attention for CNN branch
        self.spatial = CBamSpatialAttention()

        # bi-linear modelling for both
        self.W_c = Conv(ch_1, ch_int, 1, bn=True, relu=False)
        self.W_s = Conv(ch_2, ch_int, 1, bn=True, relu=False)
        self.W = Conv(ch_int, ch_int, 3, bn=True, relu=True)

        self.residual = Residual(ch_1 + ch_2 + ch_int, ch_out)
        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate

        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))

    def forward(self, c, s):

        # bilinear pooling
        W_c = self.W_c(c)
        W_s = self.W_s(s)
        bp = self.W(W_c * W_s)

        # spatial attention for cnn branch   spatial
        c = self.spatial(c)

        # channel attetion for swin-transformer branch
        s_h, s_w = self.pool_h(s), self.pool_w(s)  # .permute(0, 1, 3, 2)
        s = torch.matmul(s_h, s_w)
        s = self.relu(self.fc1(s))
        s = self.fc2(s)

        fuse = self.residual(torch.cat([c, s, bp], 1))

        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse







class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels,
            use_batchnorm=True,
    ):
        super().__init__()

        self.conv1 = Conv(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            bn=use_batchnorm
        )
        self.conv2 = Conv(
            out_channels,
            out_channels,
            kernel_size=1,
            bn=use_batchnorm
        )
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x, skip):

        x = self.upsampling(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class DecoderCup(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = DecoderBlock(in_channels=256, out_channels=64,skip_channels = 64)

        self.Blocks = nn.ModuleList()

        for i in range(3):
            #in_channels = int(128 * 2 ** (3-i))
            block = DecoderBlock(in_channels=256,out_channels=256,skip_channels=256)
            self.Blocks.append(block)

    def forward(self,features,x0):
        x = features[0]

        for i, block in enumerate(self.Blocks):
            skip = features[i+1]
            x = block(x, skip)
        x = self.block(x, x0)
        return x


class Swin_transfusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.drop_rate = config.drop_rate
        self.reduction = [1,2,2,4]
        self.resnet = models.resnet101(pretrained=True)
        self.initial = Conv(inp_dim=3, out_dim=32, bn=True)

        self.SwinTransformer = SwinTransformer(img_size=768,
                                patch_size=4,
                                in_chans=3,
                                num_classes=config.num_classes,
                                embed_dim=config.embed_dim,
                                depths=config.depths,
                                num_heads=config.num_heads,
                                window_size=config.window_size,
                                mlp_ratio=config.mlp_ratio,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=config.drop_rate,
                                drop_path_rate=config.drop_path_rate,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=True)

        self.Fuse = nn.ModuleList()

        # ResNet channel, change with the selected depth
        C1 = 256
        self.channel = [C1,C1*2,C1*4,C1*8]

        for i_layer in range(len(Config.depths)):
            self.dim = int(config.embed_dim * 2 ** i_layer)
            ch_int = 256                                     # int(config.embed_dim * 2 ** i_layer)
            ch_out = 256                                      # int(128* 2 ** i_layer)
            fuse = Fusion_block(ch_1=self.channel[i_layer], ch_2=self.dim, reduction=self.reduction[i_layer],
                                       ch_int=ch_int, ch_out=ch_out, drop_rate=self.drop_rate)
            self.Fuse.append(fuse)

        self.decoder = DecoderCup()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear')


        self.last = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)),
            ('norm', nn.BatchNorm2d(96)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv1', nn.Conv2d(96, config.num_classes, kernel_size=1))
        ]))


    def forward(self, x):

        inputs = x
        initial = self.initial(inputs)

        # swin transformer
        Trans_features = self.SwinTransformer(inputs)

        # resnet
        x0 = self.resnet.conv1(inputs)
        x0 = self.resnet.bn1(x0)
        x0 = self.resnet.relu(x0)
        c0 = x0
        x0 = self.resnet.maxpool(x0)
        c1 = self.resnet.layer1(x0)
        c2 = self.resnet.layer2(c1)
        c3 = self.resnet.layer3(c2)
        c4 = self.resnet.layer4(c3)
        features = [c1, c2, c3, c4]

        # 特征融合
        fuse_features = []
        for i,fuse in enumerate(self.Fuse):
            fuse_feature= fuse(features[i],Trans_features[i])
            fuse_features.append(fuse_feature)
        fuse_features = fuse_features[::-1]

        # decoder and upsampling
        x = self.decoder(fuse_features,c0)
        x = self.upsampling(x)
        x = torch.cat([x, initial], dim=1)
        x = self.last(x)
        return x




if __name__ == '__main__':
    with torch.no_grad():
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cuda0 = torch.device('cuda:0')
        x = torch.rand((1, 3, 768, 768), device=cuda0)
        model = TransResNet()
        model.cuda()
        output = model(x)
        print('output:', output.shape)