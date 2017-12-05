import PIL, torch
from PIL import Image
import numpy as np
import IPython.display
from io import BytesIO
import torchvision.transforms as transforms

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
    
def pil2tensor(img):
    return transforms.ToTensor()(img)

def tensor2pil(tensor):
    return transforms.ToPILImage()(tensor)

def show_image(input_image):
    f = BytesIO()
    if type(input_image) == torch.Tensor:
        input_image = np.uint8(input_image.mul(255).numpy().transpose(1, 2, 0)) 
        Image.fromarray(input_image).save(f, 'png')
    else:
        input_image.save(f, 'png')
    IPython.display.display(IPython.display.Image(data = f.getvalue()))