import torch
from torch import autograd
import torchvision as tv
from PIL import Image
import matplotlib.pyplot as plt

import os


HAS_CUDA = torch.cuda.is_available()
DTYPE = torch.cuda.FloatTensor if HAS_CUDA else torch.FloatTensor
IMSIZE = 512 #if HAS_CUDA else 128

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CONTENT_PATH = os.path.join(BASE_PATH, '../data/images/content.jpg')
STYLE_PATH = os.path.join(BASE_PATH, '../data/images/style.jpg')

def load_content_image():
    return load_image(CONTENT_PATH)

def load_style_image():
    return load_image(STYLE_PATH)

def load_image(image_path):
    loader = tv.transforms.Compose([tv.transforms.Resize((IMSIZE, IMSIZE)), tv.transforms.ToTensor()])
    pil_image = Image.open(image_path)
    image = autograd.Variable(loader(pil_image)).unsqueeze(0)
    return image.type(DTYPE)

def display_tensor(tensor, title=None):
    to_tensor = tv.transforms.ToPILImage()
    tensor = tensor.data
    tensor = tensor.clone().cpu()
    tensor = tensor.view(3, IMSIZE, IMSIZE)
    image = to_tensor(tensor)

    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.show()

def save_tensor(tensor, title=None):
    to_tensor = tv.transforms.ToPILImage()
    tensor = tensor.data
    tensor = tensor.clone().cpu()
    tensor = tensor.view(3, IMSIZE, IMSIZE)
    image = to_tensor(tensor)

    plt.figure()
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.savefig(title)
