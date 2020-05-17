# Optical Character Recognition Problem for Conditional Random Fields with Convolutions

### Description
We will explore Conditional Random Fields (CRFs), but we will use additional image level features such as convolutions to aid the training. We use PyTorch to implement our CRF model and convolutions.

### Dataset
The dataset is the same as the previous dataset from assignment 1. The original dataset is downloaded from http://www.seas.upenn.edu/∼taskar/ocr. 

### Environments and required packages
```bash
$ python3 --version
Python 3.7.4

$ pip3 install torch
$ pip3 install numpy
```

### How to run
To run the following programs, you will need to move into the folder named "code" which can be done with the following command
```bash
cd /path/to/assign/code
```

### Assignment 2
##### 3(a)
This section included the implementation of the convolution layer to combine with each CRF model provided below.
```bash
python3 conv_test.py
```
The results will be printed out to the terminal

##### 4(a)
This section is the implementation of the previous CRF model utilizing pytorch. This section is not designed to be run individually.

##### 4(b)
This section runs the following model(s):
  1. CNN-CRF
You will need to open the train.py file. Line 14, you will need to modify it to be the following:
```
batch_size = 64
```
You will need to open the crf.py file. Line 32, you will need to modify it to be the following:
```
self.cnn = conv.Conv(kernel_size=(5, 5), padding = False, stride = 1) 
```
After the above is complete, save and run the following:
 ```bash
 python3 train.py
 ```
 ##### 4(c)
 This section runs the following model(s):
  1. CNN-CNN-CRF
 You will need to open the train.py file. 
 Line 14, you will need to modify it to be the following:
 ```
batch_size = 64
```
You will need to open the crf.py file. 
Line 32, you will need to modify it to be the following:
```
self.cnn1 = conv.Conv(kernel_size=(5, 5), padding = False, stride = 1) 
self.cnn2 = conv.Conv(kernel_size=(3, 3), padding = False, stride = 1) 
```
Line 162, you will need to modify it to be the following:
```
middle_features = self.cnn1.forward(X)
return self.cnn2.forward(middle_features)
```
After the above is complete, save and run the following:
```bash
python3 train.py
```
##### 4(d)
This section repeats 4b and 4c respectively, with utilization of GPU. Additional instructions to run thus will not be included.

##### GoogLeNet

The model of GoogLeNet presents in 'googlenet.py'. To train the model:
```bash
python3 main_GN.py	
```
For ADAM optimizer:
```
optimizer = optim.Adam(net.parameters(), lr = 0.1, eps = 1e-08)
```
