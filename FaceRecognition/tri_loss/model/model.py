import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .resnet_stn import resnet50, resnet152, resnet101
import time
from ..utils.candidate_parts import generate_parts
import math
from torch.autograd import Variable
from .layer import *


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
        x: pytorch Variable
    Returns:
        x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x




class Model(nn.Module):
    def __init__(self, last_conv_stride=11, num_classes=13164, is_RPP=True, basenet='Resnet50', loss='Cosface', is_pool=True):
        super(Model, self).__init__()
        self.basenet = basenet

        self.pool_type= last_conv_stride//10
        self.is_pool = is_pool

        if basenet == 'Resnet50':
            self.base = resnet50(pretrained=True, last_conv_stride=last_conv_stride%10)
        elif basenet == 'Resnet152':
            self.base = resnet152(pretrained=True, last_conv_stride=last_conv_stride%10)
        elif basenet == 'Resnet101':
            self.base = resnet101(pretrained=True, last_conv_stride=last_conv_stride%10)
        else:
            print("Unknown Base Network")
        self.is_Graph = is_RPP

        # self.base = inception_v3(pretrained=True)
        if self.basenet == 'Resnet50' or self.basenet == 'Resnet152' or self.basenet == 'Resnet101':
            self.feature_in_dim = 2048
        elif self.basenet == 'Densenet201':
            self.feature_in_dim = 1920
        elif self.basenet == 'Densenet121':
            self.feature_in_dim = 1024
        elif self.basenet == 'Densenet169':
            self.feature_in_dim = 1664
        elif self.basenet == 'Densenet161':
            self.feature_in_dim = 2208
        elif basenet.startswith('efficient'):
            self.feature_in_dim=self.base.out_channels


        self.global_bn1=nn.BatchNorm2d(self.feature_in_dim).apply(weights_init_kaiming)
        self.global_bn3=nn.BatchNorm2d(self.feature_in_dim).apply(weights_init_kaiming)
        self.loss = loss
       
        if self.loss == "Softmax":
            self.id_classifier1 = nn.Linear(self.feature_in_dim, num_classes, bias=False)#.apply(weights_init_kaiming)
            self.id_classifier3 = nn.Linear(self.feature_in_dim, num_classes, bias=False)#.apply(weights_init_kaiming)
            weights_init_classifier(self.id_classifier1)
            weights_init_classifier(self.id_classifier3)
        elif self.loss == "Cosface":
            self.id_classifier1 = MarginCosineProduct(self.feature_in_dim, num_classes)#.apply(weights_init_kaiming) 
            self.id_classifier3 = MarginCosineProduct(self.feature_in_dim, num_classes)#.apply(weights_init_kaiming) 
            weights_init_classifier(self.id_classifier1)
            weights_init_classifier(self.id_classifier3)
        elif self.loss == "AL":
            self.id_classifier1 = AngleLinear(self.feature_in_dim, num_classes) 
            weights_init_classifier(self.id_classifier1)
            

    def forward(self, img, target):
        x, x_hard = self.base(img)
        
        if self.is_pool == True:
            id_feat1=F.max_pool2d(x, x.size()[2:])
            id_feat3=F.max_pool2d(x_hard, x_hard.size()[2:])
        else:
            id_feat1=self.fc(x.view(x.size(0),-1))
            id_feat3=self.fc(x_hard.view(x_hard.size(0),-1))
        
        if self.training==True:
            if self.loss == "Softmax":
                id_logit1=self.id_classifier1(self.global_bn1(id_feat1).view(id_feat1.size(0),-1))
                id_logit3=self.id_classifier3(self.global_bn3(id_feat3).view(id_feat3.size(0),-1))
            else:
                if self.is_pool == False:
                    id_logit1=self.id_classifier1(id_feat1.view(id_feat1.size(0),-1), target)
                    id_logit3=self.id_classifier3(id_feat3.view(id_feat3.size(0),-1), target)
                else:
                    id_logit1=self.id_classifier1(self.global_bn1(id_feat1).view(id_feat1.size(0),-1), target)
                    id_logit3=self.id_classifier3(self.global_bn3(id_feat3).view(id_feat3.size(0),-1), target)
        else:
            id_logit1 = []
            id_logit3 = []

        if self.training==True or self.is_pool == False:
            id_feat1 = id_feat1.view(id_feat1.size(0), -1)
            id_feat3 = id_feat3.view(id_feat3.size(0), -1)
        else:
            id_feat1=self.global_bn1(id_feat1).view(id_feat1.size(0),-1)
            id_feat3=self.global_bn3(id_feat3).view(id_feat3.size(0),-1)
        
        id_feat = torch.cat((id_feat1, id_feat3),1)

        return id_feat, id_logit1, id_logit3
