import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# For comparing cpu and gpu time
x = [i for i in range(0, 10)]
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)


y1 = [0.0, 294.29, 506.80, 672.45, 932.21, 1179.25, 1344.9, 1604.6, 1805.99, 1935.68]
y2 = [0.0, 1433.88, 2827.28, 3836.08, 5386.23, 6674.76, 7368.88, 7869.45, 9254.52, 10573.84]

fig, ax = plt.subplots()
line1, = ax.plot(x, y1, label='GPU')
line2, = ax.plot(x, y2, label='CPU')

ax.set(xlabel='iterator (s)', ylabel='second (s)',
       title='Comparing the runtime of CPU and GPU')
ax.grid()

ax.legend()

fig.savefig("comp_cup_gpu.png")
plt.show()




# plot the nn result
x = [i for i in range(0, 40, 5)]
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)


y1 = [0.614792, 0.675470, 0.702501, 0.681748, 0.681044, 0.646903, 0.680885, 0.670873]
y2 = [0.078125, 0.191406, 0.183594, 0.167969, 0.187500, 0.214844, 0.152344, 0.167969]

fig, ax = plt.subplots()
line1, = ax.plot(x, y1, label='letter_wise')
line2, = ax.plot(x, y2, label='word_wise')

ax.set(xlabel='iterator (s)', ylabel='accuracy',
       title='Test Accuracy of CRF + CNN with 256 batch size, kernel (5 * 5) and no padding')
ax.grid()

ax.legend()

fig.savefig("crf_cnn_acc.png")
plt.show()


# plot the nn result
x = [i for i in range(0, 60, 10)]
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)


y1 = [0.390295, 0.652087, 0.673554, 0.693790, 0.660413, 0.706150]
y2 = [0.000000, 0.125000, 0.156250, 0.187500, 0.062500, 0.125000]
y3 = [0.347072, 0.632411, 0.692964, 0.698545, 0.584034, 0.668737]
y4 = [0.000000, 0.109375, 0.140625, 0.125000, 0.046875, 0.171875]

fig, ax = plt.subplots()
line1, = ax.plot(x, y1, label='tr_letter_wise')
line2, = ax.plot(x, y2, label='tr_word_wise')
line3, = ax.plot(x, y3, label='te_letter_wise')
line4, = ax.plot(x, y4, label='te_word_wise')


ax.set(xlabel='iterator (s)', ylabel='accuracy',
       title='Train and Test Accuracy of CRF + CNN with 64 batch size, kernel (3 * 3) and padding')
ax.grid()

ax.legend()

fig.savefig("crf_cnn_3_3_1_acc.png")
plt.show()


# plot the nn result
x = [i for i in range(0, 60, 10)]
# t = np.arange(0.0, 2.0, 0.01)
# s = 1 + np.sin(2 * np.pi * t)


y1 = [0.319328, 0.330116, 0.523810, 0.226293, 0.515789, 0.547521]
y2 = [0.015625, 0.015625, 0.093750, 0.000000, 0.078125, 0.140625]
y3 = [0.369732, 0.326622, 0.505198, 0.204499, 0.490196, 0.482213]
y4 = [0.000000, 0.015625, 0.046875, 0.000000, 0.093750, 0.046875]

fig, ax = plt.subplots()
line1, = ax.plot(x, y1, label='tr_letter_wise')
line2, = ax.plot(x, y2, label='tr_word_wise')
line3, = ax.plot(x, y3, label='te_letter_wise')
line4, = ax.plot(x, y4, label='te_word_wise')


ax.set(xlabel='iterator (s)', ylabel='accuracy',
       title='Train and Test Accuracy of CRF + CNN(2 layers)')
ax.grid()

ax.legend()

fig.savefig("crf_cnn_3_3_1_and_5_5_0_acc.png")
plt.show()



y1 = [0.373626, 0.506329, 0.543388, 0.523282, 0.593361, 0.614943]
y2 = [0.015625, 0.093750, 0.109375, 0.031250, 0.140625, 0.109375]
y3 = [0.394309, 0.479691, 0.390244, 0.357576, 0.304082, 0.360000]
y4 = [0.015625, 0.046875, 0.000000, 0.000000, 0.000000, 0.000000]

fig, ax = plt.subplots()
line1, = ax.plot(x, y1, label='tr_letter_wise')
line2, = ax.plot(x, y2, label='tr_word_wise')
line3, = ax.plot(x, y3, label='te_letter_wise')
line4, = ax.plot(x, y4, label='te_word_wise')


ax.set(xlabel='iterator (s)', ylabel='accuracy',
       title='Train and Test Accuracy of CRF + CNN')
ax.grid()

ax.legend()

fig.savefig("crf_cnn_5_5_0_acc.png")
plt.show()

