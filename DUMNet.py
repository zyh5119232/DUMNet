from module.modules_for_CCBANet import SELayer
from module.gsmodule import GumbelSoftmax2D
from Res2Net import res2net50_v1b_26w_4s

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.BasicBlock import BasicConv2d
from module.CA import channel_attention
from module.SA import spatial_attention, spatial_attention_2input
from torch.nn.parameter import Parameter

class ChannelAdapter(nn.Module):
    # def __init__(self, num_features, reduction=4, reduce_to=64):
    def __init__(self, num_features, reduction=2, reduce_to=64):
        super(ChannelAdapter, self).__init__()
        self.n = reduction
        self.reduce = num_features > 20
        self.conv = nn.Sequential(nn.Conv2d(num_features,
                                            reduce_to, kernel_size=3, padding=1, bias=True),
                                  nn.LeakyReLU(inplace=True))
        self.ca = channel_attention(num_features)

    def forward(self, x):
        # reduce dimension
        if self.reduce:
            batch, c, w, h = x.size()
            # x = x.view(batch, -1, self.n, w, h)
            # x = torch.max(x, dim=2).values
            channel_attention(c),
            # x = CA_Block(in_dim=c)(x)
            # x = self.ca(x)
        # conv
        xn = self.conv(x)
        return xn

    def initialize(self):
        weight_init(self)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)

class SKConv(nn.Module):
    def __init__(self, channel, M=4, G=32, r=8, stride=1, L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(channel / r), L)
        self.M = M
        self.features = channel
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3 + i * 2, stride=stride, padding=1 + i, groups=G),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Conv2d(channel, d, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(d),
                                nn.ReLU(inplace=True))
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, channel, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        batch_size = x.shape[0]

        feats = [conv(x) for conv in self.convs]
        feats = torch.cat(feats, dim=1)
        feats = feats.view(batch_size, self.M, self.features, feats.shape[2], feats.shape[3])

        feats_U = torch.sum(feats, dim=1)
        feats_S = self.gap(feats_U)
        feats_Z = self.fc(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.M, self.features, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(feats * attention_vectors, dim=1)

        return feats_V


class ASPP(nn.Module):
    def __init__(self, in_channel, depth=32):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)

        atrous_block6 = self.atrous_block6(x)

        atrous_block12 = self.atrous_block12(x)

        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class Guidence(nn.Module):
    def __init__(self, channel):
        super(Guidence, self).__init__()
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.x3_4 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                                  nn.LeakyReLU(inplace=True))
        self.x2_3 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                                  nn.LeakyReLU(inplace=True))
        self.x1_2 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                                  nn.LeakyReLU(inplace=True))
        # self.guideout = nn.Sequential(channel_attention(channel * 4), nn.Conv2d(channel * 4, channel, 3, 1, 1),
        #                               nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True))
        self.guideout = nn.Sequential(channel_attention(channel), nn.Conv2d(channel, channel, 3, 1, 1),
                                      nn.BatchNorm2d(channel), nn.LeakyReLU(inplace=True))
        self.conv3_4 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                                     nn.LeakyReLU(inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                                     nn.LeakyReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                                     nn.LeakyReLU(inplace=True))
        self.ASPP = ASPP(channel*4)

    def forward(self, x1, x2, x3, x4):
        x1_2 = self.x1_2(abs(x1 - self.ret(x2, x1))) + x1
        x1_2 = self.conv1_2(x1_2)
        x2_3 = self.x2_3(abs(x2 - self.ret(x3, x2))) + x2
        x2_3 = self.conv2_3(x2_3)
        x3_4 = self.x3_4(abs(x3 - self.ret(x4, x3))) + x3
        x3_4 = self.conv3_4(x3_4)
        guide = torch.cat((self.ret(x4, x1_2), self.ret(x3_4, x1_2), self.ret(x2_3, x1_2), x1_2), dim=1)
        guide = self.ASPP(guide)
        guideout = self.guideout(guide)
        return guideout

    def initialize(self):
        weight_init(self)


class DECODER(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.guide_code = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1), nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Conv2d(channel, 3, 1)
        self.GS = GumbelSoftmax2D(hard=True)
        self.conv0 = nn.Conv2d(channel, channel, 1, padding=0)
        self.conv1 = nn.Conv2d(channel, channel, 1, padding=0)
        self.conv2 = nn.Conv2d(channel, channel, 1, padding=0)
        self.enhance0 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                        nn.LeakyReLU(inplace=True))
        self.enhance1 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                        nn.LeakyReLU(inplace=True))
        self.enhance2 = nn.Sequential(nn.Conv2d(channel, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                        nn.LeakyReLU(inplace=True))
        self.output = nn.Sequential(nn.Conv2d(channel*3, channel, 3, 1, 1), nn.BatchNorm2d(channel),
                      nn.LeakyReLU(inplace=True))
        self.origin = nn.Sequential(nn.BatchNorm2d(channel, eps=1e-3), nn.PReLU(channel), nn.Conv2d(channel, channel, 1, padding=0))
        self.se = SELayer(channel)
        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True))

    def forward(self, guide, curr_feature, gumbel=False):
        origin_feature = self.origin(curr_feature)
        guide_feature = self.guide_code(guide)
        gate = self.fc(guide_feature)
        guide_feature = self.GS(gate, gumbel=gumbel) * torch.mean(guide_feature, dim=1, keepdim=True)
        attention_map = [guide_feature[:, 0, :, :].unsqueeze(1), guide_feature[:, 1, :, :].unsqueeze(1), guide_feature[:, 2, :, :].unsqueeze(1)]
        guide_feature = self.ret(guide_feature, curr_feature)
        x0 = self.conv0(guide_feature[:, 0, :, :].unsqueeze(1) * curr_feature)
        x1 = self.conv1(guide_feature[:, 1, :, :].unsqueeze(1) * curr_feature)
        x2 = self.conv2(guide_feature[:, 2, :, :].unsqueeze(1) * curr_feature)
        x0 = self.enhance0(x0)
        x1 = self.enhance1(x1)
        x2 = self.enhance2(x2)

        x = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).to(guide.device)
        y = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).to(guide.device)
        z = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=True).to(guide.device)
        # out = self.se(x1)
        out = x*x0 + y*x1 + z*x2
        out = self.out_conv(out)
        output = origin_feature + out
        return output, attention_map

    def initialize(self):
        weight_init(self)

class Res2Net(nn.Module):
    def __init__(self):
        super(Res2Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.se1 = SELayer(64)
        self.se2 = SELayer(256)
        self.se3 = SELayer(512)
        self.se4 = SELayer(1024)
        self.se5 = SELayer(2048)
    def forward(self, x):
        feature = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)      # bs, 64, 88, 88
        x2 = self.resnet.layer1(x1)      # bs, 256, 88, 88
        x3 = self.resnet.layer2(x2)     # bs, 512, 44, 44
        x4 = self.resnet.layer3(x3)     # bs, 1024, 22, 22
        x5 = self.resnet.layer4(x4)     # bs, 2048, 11, 11\
        feature.append(x2)
        feature.append(x3)
        feature.append(x4)
        feature.append(x5)
        return feature


class ZYHNet(nn.Module):
    def __init__(self, base="res2net", Channel=32):
        super(ZYHNet, self).__init__()
        if base == "res2net":
            self.backbone = Res2Net()
            self.backbone_dims = [256, 512, 1024, 2048]
        self.adapter = nn.ModuleList(
            [ChannelAdapter(num_features=dims, reduce_to=Channel) for dims in self.backbone_dims])
        self.e1 = nn.Sequential(nn.Conv2d(self.backbone_dims[0], Channel, kernel_size=3, padding=1), nn.BatchNorm2d(Channel),
                                nn.ReLU(inplace=True))
        self.e2 = nn.Sequential(nn.Conv2d(self.backbone_dims[1], Channel, kernel_size=3, padding=1), nn.BatchNorm2d(Channel),
                                nn.ReLU(inplace=True))
        self.e3 = nn.Sequential(nn.Conv2d(self.backbone_dims[2], Channel, kernel_size=3, padding=1), nn.BatchNorm2d(Channel),
                                nn.ReLU(inplace=True))
        self.e4 = nn.Sequential(nn.Conv2d(self.backbone_dims[3], Channel, kernel_size=3, padding=1), nn.BatchNorm2d(Channel),
                                nn.ReLU(inplace=True))
        self.guide = Guidence(channel=Channel)
        self.multi_kernel_output = SKConv(channel=Channel, M=4, G=32, r=8)
        self.guideout = nn.Sequential(nn.Conv2d(Channel, 1, kernel_size=3, padding=1))
        self.decoder4 = DECODER(channel=Channel)
        self.decoder3 = DECODER(channel=Channel)
        self.decoder2 = DECODER(channel=Channel)
        self.guideprocess1 = nn.Sequential(nn.Conv2d(Channel, Channel, 3, padding=1), nn.BatchNorm2d(Channel), nn.LeakyReLU(True))
        self.guideprocess2 = nn.Sequential(nn.Conv2d(Channel, Channel, 3, padding=1), nn.BatchNorm2d(Channel), nn.LeakyReLU(True))
        self.guideprocess3 = nn.Sequential(nn.Conv2d(Channel, Channel, 3, padding=1), nn.BatchNorm2d(Channel), nn.LeakyReLU(True))
        # self.channel_change1 = channel_change(filters[1], Channel)
        # self.channel_change2 = channel_change(filters[2], Channel)
        # self.channel_change3 = channel_change(filters[3], Channel)
        # self.channel_change4 = channel_change(filters[4], Channel)

        self.sa1 = spatial_attention_2input(channel=Channel)
        self.sa2 = spatial_attention_2input(channel=Channel)
        self.sa3 = spatial_attention_2input(channel=Channel)
        self.sa4 = spatial_attention_2input(channel=Channel)

        self.out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, kernel_size=1, padding=0)
        )

        self.out1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(1, 1, kernel_size=1, padding=0)
        )
        self.relu = nn.LeakyReLU(inplace=True)
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(Channel, Channel // 2, kernel_size=1, padding=0)
        )

        self.out1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(Channel // 2, 1, kernel_size=1, padding=0)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(Channel*2, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(Channel, Channel, 3, padding=1)
        self.pre = nn.Sequential(
            nn.Conv2d(64, 1, 3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, padding=1)
        )
        self.conv_x4 = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.conv_x3 = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.conv_x1 = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.conv_out = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            self.relu,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            self.relu,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(Channel, Channel, 3, padding=1),
            self.relu,
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(Channel*3, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(Channel*3, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(Channel*3, Channel, 3, padding=1),
            nn.BatchNorm2d(Channel),
            self.relu,
        )
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.initialize()

    def forward(self, image):
        features = []
        x0 = image
        x = self.backbone(x0)
        x1 = x[-4]
        x2 = x[-3]
        x3 = x[-2]
        x4 = x[-1]
        x1, x2, x3, x4 = self.e1(x1), self.e2(x2), self.e3(x3), self.e4(x4)
        guide = self.guide(x1, x2, x3, x4)
        guideout = guide
        guideout = self.multi_kernel_output(guideout)
        guideout = self.guideout(guideout)
        x4 = self.up3(x4)
        x4, feature = self.decoder4(guide, x4)
        x4 = self.conv_x4(x4)

        guide = self.sa4(self.ret(x4, guide) - guide, guide)
        guide = self.guideprocess3(guide)

        x3 = self.conv3(torch.cat((x3, x4, self.ret(guide, x3)), 1))
        x3 = self.up2(x3)
        x3, feature = self.decoder3(guide, x3)
        x3 = self.conv_x3(x3)
        guide = self.sa3(self.ret(x3, guide) - guide, guide)
        guide = self.guideprocess2(guide)
        x2 = self.conv2(torch.cat((x2, x3, self.ret(guide, x2)), 1))
        x2 = self.up1(x2)
        x2, feature = self.decoder2(guide, x2)
        x2 = self.conv_x2(x2)
        guide = self.sa2(self.ret(x2, guide) - guide, guide)
        guide = self.guideprocess1(guide)

        x1 = self.conv1(torch.cat((x1, x2, self.ret(guide, x1)), 1))

        x1 = self.conv_x1(x1)
        y = self.conv_out(x1)
        y = self.out(y)
        y = self.out1(y)
        return y, guideout

    def initialize(self):
        weight_init(self)


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (
        nn.ReLU, nn.LeakyReLU, nn.ReLU6, nn.GELU, nn.Upsample, Parameter, nn.ModuleList, nn.AvgPool2d,
        nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.Sigmoid)):
            pass
        else:
            # m.initialize()
            pass
        # Dnc_stage1 = self.bn_3(self.up(Enc_final))  # 1/4
        # stage1_confidence = torch.max(nn.Softmax2d()(Dnc_stage1), dim=1)[0]
        # b, c, h, w = Dnc_stage1.size()
        # # TH = torch.mean(torch.median(stage1_confidence.view(b,-1),dim=1)[0])
        #
        # stage1_gate = (1-stage1_confidence).unsqueeze(1).expand(b, c, h, w)
        #
        # Dnc_stage2_0 = self.level2_C(output2)  # 2h 2w
        # Dnc_stage2 = self.bn_2(self.up(Dnc_stage2_0 * stage1_gate + (Dnc_stage1)))  # 4h 4w


if __name__ == "__main__":
    model = ZYHNet()
    indata = torch.ones(3, 3, 224, 224)
    output = model(indata)
    print(output[0].shape)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameters: %.2fM" % (total / 1e6))
