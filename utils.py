import os

import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

import numpy as np

from io import BytesIO
import base64

def lastest_arverage_value(values, length=100):
    if len(values) < length:
        length = len(values)
    return sum(values[-length:])/length

class ImageFolder(torch.utils.data.Dataset):
    def __init__(self, root_path, imsize=None, cropsize=None, cencrop=False):
        super(ImageFolder, self).__init__()

        self.file_names = sorted(os.listdir(root_path))
        self.root_path = root_path
        self.transform = _transformer(imsize, cropsize, cencrop)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root_path + self.file_names[index])).convert("RGB")
        return self.transform(image)

def _normalizer(denormalize=False):
    # set Mean and Std of RGB channels of IMAGENET to use pre-trained VGG net
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]    
    
    if denormalize:
        MEAN = [-mean/std for mean, std in zip(MEAN, STD)]
        STD = [1/std for std in STD]
    
    return transforms.Normalize(mean=MEAN, std=STD)

def _transformer(imsize=None, cropsize=None, cencrop=False):
    normalize = _normalizer()
    transformer = []
    if imsize:
        transformer.append(transforms.Resize(imsize))
    if cropsize:
        if cencrop:
            transformer.append(transforms.CenterCrop(cropsize))
        else:
            transformer.append(transforms.RandomCrop(cropsize))

    transformer.append(transforms.ToTensor())
    transformer.append(normalize)
    return transforms.Compose(transformer)

def imsave(tensor, path):
    denormalize = _normalizer(denormalize=True)
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = torchvision.utils.make_grid(tensor)
    torchvision.utils.save_image(denormalize(tensor).clamp_(0.0, 1.0), path)
    return None

def imload(path, imsize=None, cropsize=None, cencrop=False):
    transformer = _transformer(imsize, cropsize, cencrop)
    return transformer(Image.open(path).convert("RGB")).unsqueeze(0)

def maskingload(path):
    path_list = path.split('base64,')
    image_string = path_list[1]
    im = Image.open(BytesIO(base64.b64decode(image_string))).convert("RGBA").convert("L")
    print('maskingsum', np.sum(im))

    return im

def imload_web(path, imsize=None, cropsize=None, cencrop=False, bbox='undefined', scale=False):
    transformer = _transformer(imsize, cropsize, cencrop)
    path_list = path.split('base64,')
    image_string = path_list[1]
    # data = {}
    # data['img'] = image_string
    im = Image.open(BytesIO(base64.b64decode(image_string))).convert("RGB")
    if bbox!='undefined':
        im = im.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
    if scale!=False and scale!=1:
        w, h = im.size
        n_w = int(scale*w)
        n_h = int(scale*h)
        if n_w==0:
            n_w=1
        if n_h==0:
            n_h=1
        im = im.resize((n_w, n_h))
    return transformer(im).unsqueeze(0)

def result_to_web(result, masking):
    im = imshow(result)
    print(type(im))
    masking_img = maskingload(masking)

    w, h = masking.size
    tr = Image.new('RGBA', (w, h), (255, 0, 0, 0))
    
    im = Image.composite(tr ,im, masking_img)

    buffered = BytesIO()
    im.save(buffered, format="png")
    img_str = base64.b64encode(buffered.getvalue())
    img_str=img_str.decode('utf-8')
    print(type(img_str))
    return img_str


def imshow(tensor):
    denormalize = _normalizer(denormalize=True)    
    if tensor.is_cuda:
        tensor = tensor.cpu()    
    tensor = torchvision.utils.make_grid(denormalize(tensor.squeeze(0)))
    image = transforms.functional.to_pil_image(tensor.clamp_(0.0, 1.0))
    print(type(image))
    return image

def maskload(path):
    mask = Image.open(path).convert('L')
    return transforms.functional.to_tensor(mask).unsqueeze(0)
