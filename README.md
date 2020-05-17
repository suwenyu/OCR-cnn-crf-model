# Optical Character Recognition Problem for Conditional Random Fields with Convolutions

### Description
We will explore Conditional Random Fields (CRFs), but we will use additional image level features such as convolutions to aid the training. We use PyTorch to implement our CRF model and convolutions.


### Dataset
The original dataset is downloaded from http://www.seas.upenn.edu/∼taskar/ocr. It contains the image and label of 6,877 words collected from 150 human subjects, with 52,152 letters in total. To simplify feature engineering, each letter image is encoded by a 128 (=16\*8) dimensional vector, whose entries are either 0 (black) or 1 (white).


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
cd /path/to/repo/code
```

### Assignment

##### 3(a)
This section included the implementation of the convolution layer to combine with each CRF model provided below.
```bash
python3 conv_test.py
```
The results will be printed out to the terminal

##### 4(a)
This section is the implementation of the previous CRF model utilizing pytorch. This section is not designed to be run individually.


##### 4(b)
  * CNN-CRF
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

##### Results




