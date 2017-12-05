import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr
from torchvision import models

from matplotlib import pyplot as plt
from PIL import Image
import os, random, math

from tqdm import tqdm


def build_dict(image_root):
    global info_to_idx, idx_to_info
    info_to_idx = {}
    idx_to_info = []
    cnt = 0

    for mk in os.listdir(image_root):
        for md in os.listdir(os.path.join(image_root, mk)):
            for y in os.listdir(os.path.join(image_root, mk, md)):
                try: tup = (int(mk), int(md), int(y))
                except: continue
                if tup not in info_to_idx:
                    idx_to_info.append(tup)
                    info_to_idx[tup] = cnt
                    cnt += 1


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
    def __init__(self, root_dir, crop=False, shuffle=False, train=True, set_split=True, test_ratio=1):
        super(CompCars, self).__init__()
        self.data_list = []
        self.preprocess = tr.Compose([tr.Scale((224, 224)), tr.ToTensor()])
        image_root = os.path.join(root_dir, 'image')
        
        if crop:
            self.process = lambda img, l: self.preprocess(img.crop(tuple(l)))
        else:
            self.process = lambda img, l: self.preprocess(img)
        
        if not set_split and 'split.txt' not in os.listdir(root_dir):
            raise ValueError('No split criterion found, but set_split is False')
        if set_split:
            split_f = open(os.path.join(root_dir, 'split.txt'), 'w')
        else:
            split_f = open(os.path.join(root_dir, 'split.txt'), 'r')
        
        for mk in os.listdir(image_root):
            for md in os.listdir(os.path.join(image_root, mk)):
                for y in os.listdir(os.path.join(image_root, mk, md)):
                    names = os.listdir(os.path.join(image_root, mk, md, y))
                    if set_split:
                        cnt = len(names) / (test_ratio+1)
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
                    if i == 2: ret[5:] = [int(x)-1 for x in l.strip().split(' ')]
            return ret
        except:
            return None

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        with Image.open(data[0]) as img:
            return self.process(img, data[5:]), info_to_idx[tuple(data[1:4])]


def train_model(network, criterion, optimizer, trainLoader, valLoader, n_epochs = 10):
    network = network.cuda()
    criterion = criterion.cuda()
        
    for epoch in range(0, n_epochs):
        correct = 0.0
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
                correct += (max_labels[:,j] == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
            
            del inputs, labels
            
        # Make a pass over the validation data.
        correct = 0.0
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
                correct += (max_labels[:,j] == labels.data).sum()
            counter += inputs.size(0)
            t.set_postfix(loss = cum_loss / (1 + i), accuracy = 100 * correct / counter)
            
            del inputs, labels


def main():
    build_dict('/home/ubuntu/download/data/image')
    
    trainset = CompCars('/home/ubuntu/download/data', crop=True, train=True, set_split=False, test_ratio=9)
    valset = CompCars('/home/ubuntu/download/data', crop=True, train=False, set_split=False)
    print '%d training samples' % len(trainset)
    print '%d validation samples' % len(valset)
    print

    trainLoader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16)
    valLoader = DataLoader(valset, batch_size=128, shuffle=True, num_workers=16)

    vgg16 = models.vgg16(pretrained = True)
    for param in vgg16.parameters():
        param.requires_grad = False
    vgg16.classifier = nn.Sequential(
        nn.Linear(25088, 4096), 
        nn.ReLU(), 
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(), 
        nn.Dropout(0.5),
        nn.Linear(4096, len(idx_to_info))
    )

    optimizer = optim.SGD(vgg16.classifier.parameters(), lr = 0.0001)
    criterion = nn.CrossEntropyLoss()

    print 'finished initializing network'
    print

    train_model(vgg16, criterion, optimizer, trainLoader, valLoader, n_epochs=20)


if __name__ == '__main__': main()
