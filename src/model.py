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
        self.target = target.detach() * weight
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
        self.target = target.detach() * weight
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

## Load and build model
def load_vgg19():
    vgg = tv.models.vgg19(pretrained=True).features
    if data.HAS_CUDA:
        vgg = vgg.cuda()
    return vgg

def build_model(content_img, style_img, content_weight=1, style_weight=1000, content_layers=DEFAULT_CONTENT_LAYERS, style_layers=DEFAULT_STYLE_LAYERS):
    vgg = load_vgg19()
    content_losses = []
    style_losses = []
    model = nn.Sequential()
    gram = GramMatrix()

    if data.HAS_CUDA:
        model.cuda()
        gram.cuda()

    i, conv_i, relu_i, pool_i = 1, 1, 1, 1
    for layer in list(vgg):
        if isinstance(layer, nn.Conv2d):
            layer_name = 'conv_' + str(conv_i)
            model.add_module(layer_name, layer)
            conv_i += 1
            
            if layer_name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_losses.append(content_loss)
                i += 1

            if layer_name in style_layers:
                target = model(style_img).clone()
                target_gram = gram(target)
                style_loss = StyleLoss(target_gram, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_losses.append(style_loss)
                i += 1
                
        elif isinstance(layer, nn.ReLU):
            layer_name = 'relu_' + str(relu_i)
            model.add_module(layer_name, layer)
            relu_i +=1

            if layer_name in content_layers:
                target = model(content_img).clone()
                content_loss = ContentLoss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_losses.append(content_loss)
                i += 1

            if layer_name in style_layers:
                target = model(style_img).clone()
                target_gram = gram(target)
                style_loss = StyleLoss(target_gram, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_losses.append(style_loss)
                i += 1
                
        elif isinstance(layer, nn.MaxPool2d):
            layer_name = 'pool_' + str(pool_i)
            model.add_module(layer_name, layer)
            pool_i += 1

    return model, content_losses, style_losses

def build_optimizer(output_size):
    input_img = autograd.Variable(torch.randn(output_size)).type(data.DTYPE)
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer
    
## Style transfer
def stylize(content_img, style_img, nsteps=500, content_weight=1, style_weight=1000):
    print('Setting threads...')
    torch.set_num_threads(32)
    print('Number threads set to: %f' % torch.get_num_threads())
    
    print('Building model...')

    model, content_losses, style_losses = build_model(content_img, style_img, content_weight=content_weight, style_weight=style_weight)
    input_param, optimizer = build_optimizer(content_img.data.size())

    print('Done!')
    print('Transferring style...')

    i = [0]
    while i[0] < nsteps:
        
        def closure():
            input_param.data.clamp_(0, 1)
            optimizer.zero_grad()
            style_score, content_score = 0, 0
            model(input_param)

            for loss in content_losses:
                content_score += loss.backward()
            for loss in style_losses:
                style_score += loss.backward()

            i[0] += 1

            if i[0] % 50 == 0: 
                print("run {}:".format(i))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(style_score.data[0], content_score.data[0]))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)
    return input_param


content_img = data.load_content_image()
style_img = data.load_style_image()

#data.display_tensor(content_img, title='Content Image')
#data.display_tensor(style_img, title='Style Image')

output_img = stylize(content_img, style_img, nsteps=250)

data.display_tensor(output_img, title='Output Image')
