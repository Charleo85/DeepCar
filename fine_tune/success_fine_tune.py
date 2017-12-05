import torch
from torch import nn, optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr
from torchvision import models

from matplotlib import pyplot as plt
from PIL import Image
import os, random, math

from tqdm import tqdm as tqdm

plt.switch_backend('agg')

image_root = '../../download/data/image'

info_to_idx = {}; idx_to_info = []; cnt = 0
for mk in os.listdir(image_root):
    for md in os.listdir(os.path.join(image_root, mk)):
        num = 0
        for y in os.listdir(os.path.join(image_root, mk, md)):
            num += len(os.listdir(os.path.join(image_root, mk, md, y)))
        if num < 125: continue
        try: tup = (int(mk), int(md)) #int(y)
        except: continue
        if tup not in info_to_idx:
            idx_to_info.append(tup); info_to_idx[tup] = cnt; cnt += 1

"""
@Parameters:
"root_dir": root directory of the dataset
"crop": whether each input image is cropped by the given bounding box
"shuffle": whether the dataset is shuffled after created
"set_split": if true, generates a file of how to split training and testing data,
             otherwise, use an existing file named "split.txt" to split the data
"test_ratio": if "set_split" is true, the splitted (# train data) : (# test data) 
              would be close to "test_ratio" : 1

@Returns: (image, label)
"image": a 3D tensor of size 3x224x224, representing the input image
"label": a list of format: [make, model, year, view_angle, 
                            bbox_left, bbox_upper, bbox_right, bbox_lower]
"""
class CompCars(Dataset):
    def __init__(self, root_dir, preprocess, crop=False, shuffle=False, train=True, set_split=True, test_ratio=1):
        super(CompCars, self).__init__()
        self.data_list = []
        #self.preprocess = tr.Compose([tr.Scale((224, 224)), tr.ToTensor()])
        image_root = os.path.join(root_dir, 'image')
        def process_crop(img, l):
#             print("called")
            cropped_img = img.crop(tuple(l))
            w, h = cropped_img.size
            if w == 0 or h == 0: return preprocess(img)
            return preprocess(cropped_img)
                
        if crop:
            self.process = process_crop
        else:
            self.process = lambda img, l: preprocess(img)
        
        if not set_split and 'split.txt' not in os.listdir(root_dir):
            raise ValueError('No split criterion found, but set_split is False')
        if set_split:
            split_f = open(os.path.join(root_dir, 'split.txt'), 'w')
        else:
            split_f = open(os.path.join(root_dir, 'split.txt'), 'r')
        
        for mk in os.listdir(image_root):
            for md in os.listdir(os.path.join(image_root, mk)):
                num = 0
                for y in os.listdir(os.path.join(image_root, mk, md)):
                    num += len(os.listdir(os.path.join(image_root, mk, md, y)))
                if num < 125: continue
        
                for y in os.listdir(os.path.join(image_root, mk, md)):
                    names = os.listdir(os.path.join(image_root, mk, md, y))
                    if set_split:
                        cnt = len(names) / (test_ratio+1)
                        tests = []
                        if cnt == 0: tests = [-1]
                        else: tests = random.sample(xrange(len(names)), cnt)
                        split_f.write(' '.join([str(x) for x in tests]) + '\n')
                    else:
                        teststr = split_f.readline().strip().split(' ')
                        tests = [int(x) for x in teststr]

                    for i, img_name in enumerate(names):
                        if train and i in tests: continue
                        if not train and i not in tests: continue
                        data = self._get_data(root_dir, mk, md, y, img_name)
                        if data is not None: self.data_list.append(data)
                        
        split_f.close()
        if shuffle: random.shuffle(self.data_list)
                      
    def _get_data(self, root_dir, mk, md, y, img_name):
        ret = [None] * 9
        image_file = os.path.join(root_dir, 'image', mk, md, y, img_name)
        label_name = img_name.replace('.jpg', '.txt')
        label_file = os.path.join(root_dir, 'label', mk, md, y, label_name)

        try:
            ret[:4] = [os.path.abspath(image_file), int(mk), int(md), int(y)]
            with open(label_file, 'r') as f:
                for i, l in enumerate(f):
                    if i == 0: ret[4] = int(l.strip())
                    if i == 2: 
                        ret[5:] = [int(x)-1 for x in l.strip().split(' ')]
            return ret
        except:
            return None

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        with Image.open(data[0]) as img:
            return self.process(img, data[5:]), info_to_idx[tuple(data[1:3])]

train_preprocess = tr.Compose([tr.Resize((256)),
                            tr.RandomHorizontalFlip(), 
                            tr.RandomCrop(224),
                            tr.ColorJitter(
                                brightness=0.5,
                                contrast=0.4,
                                saturation=0.4,
                                hue=0.1
                            ),
                            tr.ToTensor(),
                            tr.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                       ])
val_preprocess = tr.Compose([tr.Resize((256)),
                            tr.CenterCrop(224), 
                            tr.ToTensor(),
                            tr.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                       ])
print 'building training set'
trainset = CompCars('../../download/data', train_preprocess, crop=False, train=True, set_split=True, test_ratio=15)
print 'building validation set'
valset = CompCars('../../download/data', val_preprocess, crop=False, train=False, set_split=False)

print len(trainset), len(valset), len(idx_to_info)

trainLoader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=16)
valLoader = DataLoader(valset, batch_size=64, shuffle=True, num_workers=16)

def train_model(network, criterion, optimizer, scheduler, trainLoader, valLoader, n_epochs = 10, model_name='.'):
    network = network.cuda()
    criterion = criterion.cuda()
        
    train_loss_arr = [None] * n_epochs
    train_top5_arr = [None] * n_epochs
    train_top1_arr = [None] * n_epochs
    valid_loss_arr = [None] * n_epochs
    valid_top5_arr = [None] * n_epochs
    valid_top1_arr = [None] * n_epochs
        
    for epoch in range(0, n_epochs):
        correct1 = 0.0
        correct5 = 0.0
        cum_loss = 0.0
        counter = 0

        # Make a pass over the training data.
        t = tqdm(trainLoader, desc = 'Training epoch %d' % epoch)
        network.train()  # This is important to call before training!
        for (i, (inputs, labels)) in enumerate(t):
            
            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # Backward pass:
            optimizer.zero_grad()
            # Loss is a variable, and calling backward on a Variable will
            # compute all the gradients that lead to that Variable taking on its
            # current value.
            loss.backward() 

            # Weight and bias updates.
            optimizer.step()

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.topk(5, dim=1)
            for j in xrange(5):
                if j==0: correct1 += (max_labels[:,j] == labels.data).sum()
                correct5 += (max_labels[:,j] == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), top5 = 100 * correct5 / counter, top1 = 100 * correct1 / counter)
                                
            train_loss_arr[epoch] = cum_loss / (1 + i)
            train_top5_arr[epoch] = 100 * correct5 / counter
            train_top1_arr[epoch] = 100 * correct1 / counter

        # Make a pass over the validation data.
        correct1 = 0.0
        correct5 = 0.0
        cum_loss = 0.0
        counter = 0
        t = tqdm(valLoader, desc = 'Validation epoch %d' % epoch)
        network.eval()  # This is important to call before evaluating!
        for (i, (inputs, labels)) in enumerate(t):

            # Wrap inputs, and targets into torch.autograd.Variable types.
            inputs = Variable(inputs)
            labels = Variable(labels)
            
            inputs = inputs.cuda()
            labels = labels.cuda()

            # Forward pass:
            outputs = network(inputs)
            loss = criterion(outputs, labels)

            # logging information.
            cum_loss += loss.data[0]
            max_scores, max_labels = outputs.data.topk(5, dim=1)
            for j in xrange(5):
                if j==0: correct1 += (max_labels[:,j] == labels.data).sum()
                correct5 += (max_labels[:,j] == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), top5 = 100 * correct5 / counter, top1 = 100 * correct1 / counter)
                    
            valid_loss_arr[epoch] = cum_loss / (1 + i)
            valid_top5_arr[epoch] = 100 * correct5 / counter
            valid_top1_arr[epoch] = 100 * correct1 / counter
        
        scheduler.step()
        if (epoch+1) % 20 == 0:
            with open('%s/epoch%d.pt' % (model_name, epoch+1,), 'wb') as f:
                torch.save(network, f)
                print 'Successfully saved model at epoch %d' % (epoch+1,)
        
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('loss scores')
    #     axes = plt.gca()
    #     axes.set_ylim([1.7,2.0])
    plt.plot(xrange(n_epochs), train_loss_arr)
    plt.plot(xrange(n_epochs), valid_loss_arr)

    plt.savefig(model_name+'/loss.png')

    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('accuracy scores')
    plt.plot(xrange(n_epochs), train_top1_arr)
    plt.plot(xrange(n_epochs), valid_top1_arr)

    plt.savefig(model_name+'/acctop1.png')
    
    plt.figure()
    plt.xlabel('epoch')
    plt.ylabel('accuracy scores')
    plt.plot(xrange(n_epochs), train_top5_arr)
    plt.plot(xrange(n_epochs), valid_top5_arr)

    plt.savefig(model_name+'/acctop5.png')
    
    
# vgg16 = models.vgg16(pretrained = True)
# for param in vgg16.parameters():
#     param.requires_grad = False
# vgg16.classifier = nn.Sequential(
#     nn.Linear(25088, 4096), 
#     nn.ReLU(), 
#     nn.Dropout(0.5),
#     nn.Linear(4096, 4096),
#     nn.ReLU(), 
#     nn.Dropout(0.5),
#     nn.Linear(4096, len(idx_to_info))
# )

# optimizer = optim.SGD(vgg16.classifier.parameters(), lr = 0.001)
# scheduler = LambdaLR(optimizer, lambda e: 1 if e < 200/2 else 0.1)
# criterion = nn.CrossEntropyLoss()

# print 'VGG16 models loaded'
# train_model(vgg16, criterion, optimizer, scheduler, trainLoader, valLoader, n_epochs = 200, model_name='vgg16')


# resnet18 = models.resnet18(pretrained=True)
# for param in resnet18.parameters():
#     param.requires_grad = False
# resnet18.fc = nn.Sequential(
#     nn.Linear(512, 512), 
#     nn.ReLU(), 
#     nn.Dropout(0.5),
#     nn.Linear(512, 512),
#     nn.ReLU(), 
#     nn.Dropout(0.5),
#     nn.Linear(512, len(idx_to_info))
# )
# optimizer = optim.SGD(resnet18.fc.parameters(), lr = 0.001)
# scheduler = LambdaLR(optimizer, lambda e: 1 if e < 200/2 else 0.1)
# criterion = nn.CrossEntropyLoss()

# print 'resnet18 models loaded'
# train_model(resnet18, criterion, optimizer, scheduler, trainLoader, valLoader, n_epochs = 200, model_name='resnet18')

resnet152 = models.resnet152(pretrained=True)
for param in resnet152.parameters():
    param.requires_grad = False
resnet152.fc = nn.Sequential(
    nn.Linear(2048, 2048), 
    nn.ReLU(), 
    nn.Dropout(0.5),
    nn.Linear(2048, 512),
    nn.ReLU(), 
    nn.Dropout(0.5),
    nn.Linear(512, len(idx_to_info))
)
optimizer = optim.SGD(resnet152.fc.parameters(), lr = 0.001)
scheduler = LambdaLR(optimizer, lambda e: 1 if e < 200/2 else 0.1)
criterion = nn.CrossEntropyLoss()

print 'resnet152 models loaded'
train_model(resnet152, criterion, optimizer, scheduler, trainLoader, valLoader, n_epochs = 200, model_name='resnet152')

