import torch
from torch import nn, autograd, optim
import torchvision as tv
import matplotlib.pyplot as plt

import data

DEFAULT_CONTENT_LAYERS = [ 'conv_4' ]
DEFAULT_STYLE_LAYERS = [ 'conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5' ]

## Additional Layer Types
class GramMatrix(nn.Module):

    def forward(self, ins):
        n, c, h, w = ins.size()  ## tensor is NCHW format
        features = ins.view(n * c, h * w)
        gram_prod = torch.mm(features, features.t())
        return gram_prod.div(n * c * h * w)
        

## Losses
class ContentLoss(nn.Module):

    def __init__(self, target, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detatch() * weight
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, ins):
        self.loss = self.criterion(ins * self.weight, self.target)
        self.output = ins.clone()
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detatch() * weight
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, ins):
        self.output = ins.clone()
        self.gram_mtx = self.gram(ins)
        self.gram_mtx.mul_(self.weight)
        self.loss = self.criterion(self.gram_mtx, self.target)
        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss

def load_vgg19():
    vgg = tv.models.vgg19(pretrained=True).features
    if data.HAS_CUDA:
        vgg = vgg.cuda()
    return vgg

def build_model(style_img, content_img, style_weight=1000, content_weight=1, content_layers=DEFAULT_CONTENT_LAYERS, style_layers=DEFAULT_STYLE_LAYERS):
    vgg = load_vgg19()
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    gram = GramMatrix()

    if data.HAS_CUDA:
        model.cuda()
        gram.cuda()

    i = 1
    for layer in list(vgg):
        if isinstance(layer, nn.Conv2d):
        elif isinstance(layer, nn.ReLU):
        elif isinstance(layer, nn.MaxPool2d):
                

vgg = load_vgg19()
for layer in list(vgg):
    print(repr(layer))
