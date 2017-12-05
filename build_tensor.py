import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr
from torchvision import models

from matplotlib import pyplot as plt
from PIL import Image
import os, random, math

image_root = '../roadway_intel/data/image'
tensor_root = '../roadway_intel/data/tensor'
root_dir = '../roadway_intel/data'
preprocess = tr.Compose([tr.Scale((224, 224)), tr.ToTensor()])
process = lambda img, l: preprocess(img.crop(tuple(l)))

def get_data(root_dir, mk, md, y, img_name):
    ret = [None] * 9                 
    img_file = os.path.join(root_dir, 'image', mk, md, y, img_name)
    label_name = img_name.replace('.jpg', '.txt')
    label_file = os.path.join(root_dir, 'label', mk, md, y, label_name)

    try:
        ret[:4] = [os.path.abspath(img_file), int(mk), int(md), int(y)]
        with open(label_file, 'r') as f:
            for i, l in enumerate(f):
                if i == 0: ret[4] = int(l.strip())
                if i == 2: ret[5:] = [int(x)-1 for x in l.strip().split(' ')]
        return ret
    except:
        return None

cnt = 0
for mk in os.listdir(image_root):
    for md in os.listdir(os.path.join(image_root, mk)):
        for y in os.listdir(os.path.join(image_root, mk, md)):
            for img_name in os.listdir(os.path.join(image_root, mk, md, y)):
                data = get_data(root_dir, mk, md, y, img_name)
                if data is None: continue
                
                with Image.open(data[0]) as img_pil:
                    img_tensor = process(img_pil, data[5:])
                    tensor_fname = data[0].replace('.jpg', '.pt').replace('image', 'tensor')
                    with open(tensor_fname, 'wb') as f:
                        torch.save(img_tensor, f)
                
                cnt += 1
                if cnt % 500 == 0: print cnt
