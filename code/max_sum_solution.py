import numpy as np
import utils, compute


def max_sum(X, W, T):
    
    letter_size = 26
    seq_len = X.shape[0]
    l = np.zeros((seq_len, letter_size))
    tmp = [0] * (letter_size)
    y_ans = np.zeros(seq_len, dtype=np.int)

    #  li(yi) = max_{y_i-1}{ <wyi−1, xi−1> + Tyi−1,yi + li−1(yi−1) }
    #  i: i, j: y_{i}, k: y_{i-1}
    for i in range(1, seq_len):
        for j in range(letter_size):
            for k in range(letter_size):
                tmp[k] = np.dot(W[k], X[i-1]) + T[k, j] + l[i-1, k]
            l[i, j] = max(tmp)
    
    # recovery 
    # y∗_m = argmax_ym { <wym, xm> + lm(ym) } 
    for i in range(letter_size):
        tmp[i] = np.dot(W[i], X[-1]) + l[-1, i]

    y_ans[-1] = np.argmax(tmp)
    # print(tmp[y_ans[-1]])


    # y∗_i−1 = argmax_{yi−1}{ <wyi−1, xi−1> + T_{yi−1,y∗i} + li−1(yi−1) }
    for i in range(seq_len-1, 0, -1):
        for j in range(0, letter_size):
            tmp[j] = np.dot(W[j], X[i-1]) + T[j, y_ans[i]] + l[i-1, j]
        y_ans[i-1] = np.argmax(tmp)

    return y_ans



if __name__ == "__main__":
    X, W, T = utils.load_decode_input()

    decode_list = max_sum(X, W, T)
    # print(decode_list)

    f = open('../result/decode-output.txt', 'w')
    for i in decode_list:
        f.write(str(i+1) + "\n")
    f.close()
    