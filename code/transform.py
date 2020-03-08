import cv2
import numpy as np
import utils,string
# import matplotlib.pyplot as mp
import os
from skimage.transform import rotate as rotatopotato

from scipy import ndimage, misc
import math
#, compute, check_grad

def read_data():
    train_data = utils.read_data_seq('../data/train_mini.txt')
    trainX, trainY = [], []
    for d in train_data:
        trainX.append(d[1])
        trainY.append(d[0])

    return trainX, trainY

#offset (tx, ty)
def translate(features, tx, ty):
    features = features.reshape(16, 8)
    
    lenX, lenY = features.shape
    #translated = np.zeros((lenX, lenY), features.dtype)

    M = np.float32([[1,0,tx],[0,1,ty]])
    dst = cv2.warpAffine(features,M,(lenY,lenX))

    # mp.imshow(dst, cmap='gray')
    # mp.show()
    return dst.flatten()


def rotate(features, alpha):

    X = features.reshape(16,8)

    # Y = misc.imrotate(X, 15)
    Y = rotatopotato(X, 15)
    lenx1, lenx2 = X.shape;
    leny1, leny2 = Y.shape;

    #  Trim the result back to the original size, around its center point.
    fromx = math.floor((leny1 + 1 - lenx1)/2);
    fromy = math.floor((leny2 + 1 - lenx2)/2);

    destination = Y[fromx:fromx+lenx1, fromy:fromy+lenx2]

    img_binary = cv2.threshold(destination, 128,255, cv2.THRESH_BINARY)[1]

    rows,col=np.where(img_binary == 255)
    img_binary[rows, col] = 1

    return img_binary.flatten()


def transform_data(train_data, n):
    f = open('../data/transform.txt', 'r')
    trans_list = []
    for line in f:
        line_list = line.strip().split(' ')
        # print(line_list)
        if line_list != []:
            trans_list.append([line_list[0], int(line_list[1]), line_list[2:]])
    f.close()
    # print(trans_list)
    for i in range(n):
        if trans_list[i][0] == 'r':
            _id = trans_list[i][1]

            new_img = [rotate(j, int(trans_list[i][2][0])) for j in train_data[_id-1][1]]
            train_data[_id-1][1] = np.array(new_img)

            # print("rotate")
            # rotate()
        
        elif trans_list[i][0] == 't': 
            _id = trans_list[i][1]

            # print(_id)
            # for j in train_data[_id-1][1]:
                # print(j)
            # print([ trans_list[i] for j in train_data[_id][1]])
            # print([ (int(trans_list[i][2][0]), int(trans_list[i][2][1])) for j in train_data[_id][1]])
            new_img = [translate(j, int(trans_list[i][2][0]), int(trans_list[i][2][1])) for j in train_data[_id-1][1]]
            train_data[_id-1][1] = np.array(new_img)


    return train_data
            # print("translate")
            # translate()
        
        


if __name__ == "__main__":
    
    train_data = utils.read_data_seq('../data/train.txt')


    train_data_new = transform_data(train_data, 1000)
    for i, j in zip(train_data_new, train_data):
        # print(i, j)
        for k, l in zip(i, j):
        #     # print(k)
        #     # print(k.shape)
            if k.shape != l.shape:
                print(k)
    # print(train_data_new)

    # X, Y = read_data()
    # #train_data = utils.read_data_seq('../data/train_mini.txt') 

    # #p, letter = train_data[1], train_data[0]
    # pic = X[0][1]

    # print(np.array(pic).shape)
    
    # pic_d = np.array(pic).reshape(16,8)
    # print(pic_d.shape)

    # #original figure
    # #mp.figure(3)
    # mp.imshow(pic_d, cmap = 'gray')
    # mp.show()
    # #translation
    # translate_data(pic_d, 3, 3)

    # #lists of transformations from translation.txt (haven't done yet)
    # f = open('../data/translation_mini.txt', 'r')

    # for line in f:
    #     offset = []
    #     content = line.rstrip().split(' ')
    #     flag, nth, offx, offy = content[0], int(content[1]), int(content[2]), int(content[3])
    #     #if(flag=='t'):
    #         #print('word count: ', nth, '(', offx, offy, ')')
    #         #print('origin data:\n', X[nth])
    #         #print(np.array(X[nth]).shape)
    #         #translate_data(pic[nth], offx, offy)


