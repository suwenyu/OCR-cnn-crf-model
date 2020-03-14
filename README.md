# aml2020s_assign2

### Description
In this project we will continue to explore Conditional Random Fields (CRFs), but we will use additional image level features such as convolutions to aid the training. We will use PyTorch to implement our CRF model and convolutions.

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
3a included the implementation of the convolution layer to combine with each CRF model provided below.
```bash
python3 conv_test.py
```
The results will be printed out to the terminal

##### 4(a)
This section is the implementation of the previous CRF model utilizing pytorch. This section is not designed to be run individually.

##### 4(b)
This section runs the following model(s):
  1. CNN-CRF
  
You will need to open the crf.py file. Line 32, you will need to modify it to be the following:
```
self.cnn = conv.Conv(kernel_size=(5, 5), padding = False, stride = 1) 
```
After the above is complete, save and run the following:
 ```bash
 python3 train.py
 ```
 
