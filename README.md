# DeepCar
Fine-grained detection on Vehicle Model/Make

## Dataset
Training dataset consisted of 163/1,716 vehicle make/models from [CompCars dataset](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)[1]


## Fine-tune VGG
#### Architecture
A [VGG16](http://arxiv.org/pdf/1409.1556.pdf) model pre-trained on ImageNet was fine-tuned with CompCars dataset (16970/776 train/valid images - 115 vehicles/classes)

#### Results
Accuracy: **93.2% top-5** in **200 epochs**
Base learning rate of 0.001 and batch size of 64 were used. 

## RA-CNN Look closer to see better
#### Architecture
#### Results


### References
[1] Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. A Large-Scale Car Dataset for Fine-Grained Categorization and Verification, In Computer Vision and Pattern Recognition (CVPR), 2015.
