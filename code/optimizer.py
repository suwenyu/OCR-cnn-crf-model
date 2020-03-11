"""
Author: Yeshu Li
The Python program has been tested under macOS Mojava Version 10.14.3 and Ubuntu 18.04.

The file paths are hard-coded in the code for my convenience. There are 4 features in crf.py file.

1. p2a function computes the required log-likelihood and stores the required gradients in gradients.txt.
2. p2b function computes the optimal parameter by using L-BFGS-B optimization method, outputs the final objective function value and stores the optimal parameter in solution.txt.
3. checkGrad function checks the gradients against finite differences.


"""

import time
import math
import numpy as np
from scipy.optimize import check_grad, minimize

K = 27
imgSize = 128
paraNum = K * K + K * imgSize

def readDataset(filePath):

    words = []

    with open(filePath, 'r') as f:
        label = []
        data = []
        for line in f.readlines():
            tokens = line.split()
            label.append(ord(tokens[1]) - ord('a'))
            data.append([int(x) for x in tokens[5:]])
            if tokens[2] == '-1':
                words.append([np.array(label), np.array(data)])
                label = []
                data = []

    return words

def readParameter(filePath):

    w = np.zeros((K, imgSize))
    T = np.zeros((K, K))

    with open(filePath, 'r') as f:
        lines = [float(line) for line in f.readlines()]
        for i in range(K):
            w[i] = np.array(lines[i * imgSize : (i + 1) * imgSize])
        offset = K * imgSize
        for i in range(K):
            for j in range(K):
                T[j, i] = lines[offset + i * K + j]

    return w, T

def computeAllDotProduct(w, word):
   
    label, data = word
    dots = np.dot(w, data.transpose())

    return dots

def logTrick(numbers):

    if len(numbers.shape) == 1:
        M = np.max(numbers)
        return M + np.log(np.sum(np.exp(numbers - M)))
    else:
        M = np.max(numbers, 1)
        return M + np.log(np.sum(np.exp((numbers.transpose() - M).transpose()), 1))

def logPYX(word, w, T, alpha, dots):

    label, data = word
    # print(label, data)
    m = len(label)
    res = sum([dots[label[i], i] for i in range(m)]) + sum([T[label[i], label[i + 1]] for i in range(m - 1)])
    logZ = logTrick(dots[:, m - 1] + alpha[m - 1, :])
    res -= logZ
    return res

def computeDP(word, w, T, dots):

    label, data = word
    m = len(label)
    alpha = np.zeros((m, K))
    for i in range(1, m):
        alpha[i] = logTrick(np.tile(dots[:, i - 1] + alpha[i - 1, :], (K, 1)) + T.transpose())
    beta = np.zeros((m, K))
    for i in range(m - 2, -1, -1):
        beta[i] = logTrick(np.tile(dots[:, i + 1] + beta[i + 1, :], (K, 1)) + T)

    return alpha, beta

def computeMarginal(word, w, T, alpha, beta, dots):

    label, data = word
    m = len(label)
    p1 = np.zeros((m, K))
    for i in range(m):
        p1[i] = alpha[i, :] + beta[i, :] + dots[:, i]
        p1[i] = np.exp(p1[i] - logTrick(p1[i]))
    p2 = np.zeros((m - 1, K, K))
    for i in range(m - 1):
        p2[i] = np.tile(alpha[i, :] + dots[:, i], (K, 1)).transpose() + np.tile(beta[i + 1, :] + dots[:, i + 1], (K, 1)) + T
        p2[i] = np.exp(p2[i] - logTrick(p2[i].flatten()))

    return p1, p2

def computeGradientWy(word, p1):

    label, data = word
    m = len(label)
    cof = np.zeros((K, m))
    for i in range(m):
        cof[label[i], i] = 1
    cof -= p1.transpose()
    res = np.dot(cof, data)

    return res

def computeGradientTij(word, p2):

    label, data = word
    m = len(label)
    res = np.zeros(p2.shape)
    for i in range(m - 1):
        res[i, label[i], label[i + 1]] = 1
    res -= p2
    res = np.sum(res, 0)
   
    return res

def crfFuncGrad(x, dataset, C):

    x = np.array(x)
    w = np.array(x[ : imgSize * K]).reshape(K, imgSize)
    T = np.array(x[imgSize * K : ]).reshape(K, K)

    meandw = np.zeros((K, imgSize))
    meandT = np.zeros((K, K))
    meanLogPYX = 0

    for word in dataset:
        # print(word)

        dots = computeAllDotProduct(w, word)
        alpha, beta = computeDP(word, w, T, dots)
        p1, p2 = computeMarginal(word, w, T, alpha, beta, dots)

        dw = computeGradientWy(word, p1)
        dT = computeGradientTij(word, p2)

        meanLogPYX += logPYX(word, w, T, alpha, dots)
        meandw += dw
        meandT += dT

    meanLogPYX /= len(dataset)
    meandw /= len(dataset)
    meandT /= len(dataset)

    meandw *= (-C)
    meandT *= (-C)

    meandw += w
    meandT += T

    gradients = np.concatenate((meandw.flatten(), meandT.flatten()))

    objValue = -C * meanLogPYX + 0.5 * np.sum(w ** 2) + 0.5 * np.sum(T ** 2)

    return [objValue, gradients]

def checkGradient(dataset, w, T):

    lossFunc = lambda x, *args: crfFuncGrad(x, *args)[0]
    gradFunc = lambda x, *args: crfFuncGrad(x, *args)[1]
    x0 = np.concatenate((w.flatten(), T.flatten()))

    print(check_grad(lossFunc, gradFunc, x0, dataset, 1))

def p2a(dataset, w, T, outputPath):

    meanLogPYX = 0
    meandw = np.zeros((K, imgSize))
    meandT = np.zeros((K, K))

    for word in dataset:

        dots = computeAllDotProduct(w, word)
        alpha, beta = computeDP(word, w, T, dots)
        p1, p2 = computeMarginal(word, w, T, alpha, beta, dots)

        meanLogPYX += logPYX(word, w, T, alpha, dots)
        dw = computeGradientWy(word, p1)
        dT = computeGradientTij(word, p2)
        meandw += dw
        meandT += dT

    meanLogPYX /= len(dataset)
    meandw /= len(dataset)
    meandT /= len(dataset)

    print('Mean log-likelihood: %lf' % meanLogPYX)

    res = np.concatenate((meandw.flatten(), meandT.transpose().flatten()))
    resStr = '\n'.join([str(num) for num in res])

    with open(outputPath, 'w') as f:
        f.write(resStr)

    return 0

def p2b(trainSet, testSet, solutionPath, C):

    paras = np.zeros(paraNum)

    optimalParas = minimize(crfFuncGrad, paras, args = (trainSet, C), method = 'L-BFGS-B', jac = True, options={'disp': 99})

    x = np.array(optimalParas.x)
    w = np.array(x[ : imgSize * K]).reshape(K, imgSize)
    T = np.array(x[imgSize * K : ]).reshape(K, K)

    res = np.concatenate((w.flatten(), T.transpose().flatten()))
    resStr = '\n'.join([str(num) for num in res])

    print('Optimal objective value: %lf' % optimalParas.fun)

    with open(solutionPath, 'w') as f:
        f.write(resStr)

    return optimalParas

def main():

    start = time.time()

    trainPath = '../data/train.txt'
    testPath = '../data/test.txt'
    modelParameterPath = '../data/model.txt'

    trainSet = readDataset(trainPath)
    testSet = readDataset(testPath)
    w, T = readParameter(modelParameterPath)

    # generate the answer for q2b
    c_list = [10**x for x in range(4, 5)]
    paths = ['results/parameterc' + str(10**x) + '.txt' for x in range(4, 5)]

    for iter, C in enumerate(c_list):  # generate optimized parameters and store them in text file.
        print(C)
        print(paths[iter])
        p2b(trainSet, testSet, paths[iter], C)

    print('Time elapsed: %lf' % (time.time() - start))

if __name__ == '__main__':

    main()
