import torch
from torch import nn, autograd, optim
import matplotlib.pyplot as plt

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
