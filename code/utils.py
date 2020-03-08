import numpy as np
import string
import torch


letter_dict = dict(zip(string.ascii_lowercase, range(0,26)))


def read_data_seq(filename):
    # read data from file, return list of tuples, 
    # and each tuple contains word(list of chars) and img features([128, 1]*len(word))
    # ex. (['a', 'k', 'e'], [[1, 0, 0 ... ], [0, 1, 0 ...], [0, 0, 0 ...]])
    data = []
    tmp_id = 1
    label = []
    features = []
    
    f = open(filename, 'r')
    for line in f:
        line_list = line.rstrip().split(' ')
        _id, letter, word_id = line_list[0], line_list[1], int(line_list[3])
        feature = line_list[5:]

        label.append(letter_dict[letter])
        features.append(np.array(feature, dtype=float))

        if int(line_list[2]) < 0:
            label = np.array(label, dtype=np.int)
            features = np.array(features, dtype=np.float)
            data.append([label, features])
            tmp_id = word_id
            label = []
            features = []
    
    return data


def read_data_struct():
    label, features = [], []
    
    f = open('../data/train.txt', 'r')
    for line in f:
        line_list = line.rstrip().split(' ')
        _id, letter, word_id = line_list[0], line_list[1], int(line_list[3])
        feature = line_list[5:]   

        features.append(feature)
        label.append([letter_dict[letter], word_id])

    features = np.array(features, dtype=np.float)
    return label, features


def load_decode_input():
    f = open('../data/decode_input.txt', 'r')
    inp_list = []
    for line in f:
        inp_list.append(line.rstrip())
    # print(inp_list)
    
    X = np.array(inp_list[:100 * 128], dtype=np.float64).reshape(100, 128)
    W = np.array(inp_list[100 * 128:100 * 128 + 26 * 128], dtype=np.float64).reshape(26, 128)

    T = np.array(inp_list[-26*26:], dtype=np.float64).reshape(26, 26)
    # original data is stored in column vector (swap by column and row)
    T = np.swapaxes(T, 0, 1)
    
    return X, W, T

def read_model():
#function to read model for 2a
    f = open('../data/model.txt', 'r')
    line_list = []    
    for line in f:
        line = line.rstrip()
        line_list.append(line)


    W = np.array(line_list[:26*128], dtype=float).reshape(26, 128)
    T = np.array(line_list[26*128:], dtype=float).reshape(26, 26)
    T = np.swapaxes(T, 0, 1)
    return W, T


def load_model_params(filename):
    file = open(filename, 'r')
    params = []

    for line in file:
        params.append(line.rstrip())

    return np.array(params, dtype=np.float)


def extract_w(params):
    return (params[:26*128]).reshape(26, 128)


def extract_t(params):

    t = np.array(params[26*128:]).reshape(26, 26)
    # print(t[25][25])
    # print(params[26*128:].shape)
    # for i in t:
    #     print(np.sum(i))
    t = np.swapaxes(t, 0, 1)
    return t

# def extract_w(params):
#     return params[:26*128].view(26, 128)


# def extract_t(params):

#     t = params[26*128:].view(26, 26)
#     # print(t[25][25])
#     # print(params[26*128:].shape)
#     # for i in t:
#     #     print(np.sum(i))
#     t = t.permute(0, 1)
#     # t = np.swapaxes(t, 0, 1)
#     return t

if __name__ == '__main__':
    train_data = read_data_seq('../data/train.txt')
    test_data = read_data_seq('../data/test.txt')
    # print(read_model())
    # datax, datay = read_data_struct()
    # print(datay)