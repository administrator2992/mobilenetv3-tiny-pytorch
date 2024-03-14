import torch
import torch.nn as nn

from nets.mobilenet_v3 import mobilenet_v3
from nets.attention import cbam_block, eca_block, se_block, CA_Block

attention_block = [se_block, cbam_block, eca_block, CA_Block]

class MobileNetV3Large(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV3Large, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained, mb_type='large')

    def forward(self, x):
        x = self.model.features[:7](x)
        out4 = self.model.features[7:13](x)
        out5 = self.model.features[13:16](out4)
        return out4, out5

class MobileNetV3Small(nn.Module):
    def __init__(self, pretrained=False):
        super(MobileNetV3Small, self).__init__()
        self.model = mobilenet_v3(pretrained=pretrained, mb_type='small')

    def forward(self, x):
        x = self.model.features[:4](x)
        out4 = self.model.features[4:7](x)
        out5 = self.model.features[7:12](out4)
        return out4, out5

#-------------------------------------------------#
#   卷积块 -> 卷积 + 标准化 + 激活函数
#   Conv2d + BatchNormalization + ReLU6
#-------------------------------------------------#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicConv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

def conv_dw(filter_in, filter_out, stride = 1):
    return nn.Sequential(
        nn.Conv2d(filter_in, filter_in, 3, stride, 1, groups=filter_in, bias=False),
        nn.BatchNorm2d(filter_in),
        nn.ReLU6(inplace=True),

        nn.Conv2d(filter_in, filter_out, 1, 1, 0, bias=False),
        nn.BatchNorm2d(filter_out),
        nn.ReLU6(inplace=True),
    )

#---------------------------------------------------#
#   卷积 + 上采样
#---------------------------------------------------#
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            BasicConv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x

#---------------------------------------------------#
#   最后获得yolov4的输出
#---------------------------------------------------#
def yolo_head(filters_list, in_filters):
    m = nn.Sequential(
        conv_dw(in_filters, filters_list[0]),
        
        nn.Conv2d(filters_list[0], filters_list[1], 1),
    )
    return m
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi=0, pretrained=False, backbone="mobilenetv3large"):
        super(YoloBody, self).__init__()
        self.phi            = phi
        if backbone == "mobilenetv3large":
            self.backbone       = MobileNetV3Large(pretrained=pretrained)
            in_filter           = 160
        elif backbone == "mobilenetv3small":
            self.backbone       = MobileNetV3Small(pretrained=pretrained)
            in_filter           = 96

        self.conv_for_P5    = BasicConv(in_filter,256,1)
        self.yolo_headP5    = yolo_head([512, len(anchors_mask[0]) * (5 + num_classes)],256)

        self.upsample       = Upsample(256,128)
        self.yolo_headP4    = yolo_head([256, len(anchors_mask[1]) * (5 + num_classes)],384)

        if 1 <= self.phi and self.phi <= 4:
            self.feat1_att      = attention_block[self.phi - 1](256)
            self.feat2_att      = attention_block[self.phi - 1](512)
            self.upsample_att   = attention_block[self.phi - 1](128)

    def forward(self, x):
        #---------------------------------------------------#
        #   生成CSPdarknet53_tiny的主干模型
        #   feat1的shape为26,26,256
        #   feat2的shape为13,13,512
        #   
        #   MobileNetV3-Large
        #   feat1: 26,26,112
        #   feat2: 13,13,160
        # 
        #   MobileNetV3-Small
        #   feat1: 26,26,40
        #   feat2: 13,13,96
        #---------------------------------------------------#
        feat1, feat2 = self.backbone(x)
        if 1 <= self.phi and self.phi <= 4:
            feat1 = self.feat1_att(feat1)
            feat2 = self.feat2_att(feat2)

        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,255
        out0 = self.yolo_headP5(P5)

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample(P5)
        print(P5_Upsample.size())
        # 26,26,256 + 26,26,128 -> 26,26,384
        if 1 <= self.phi and self.phi <= 4:
            P5_Upsample = self.upsample_att(P5_Upsample)
        P4 = torch.cat([P5_Upsample,feat1],axis=1)

        # 26,26,384 -> 26,26,256 -> 26,26,255
        out1 = self.yolo_headP4(P4)
        
        return out0, out1

