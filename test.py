# coding=utf-8
# author: BebDong
# 10/23/18

import numpy
import matplotlib.pyplot as plt

# 打开并读取文件
data_file = open("mnist_dataset/mnist_train_100.csv")
data_list = data_file.readlines()
data_file.close()

# 拆分绘制28*28图形
all_pixels = data_list[0].split(',')
image_array = numpy.asfarray(all_pixels[1:]).reshape((28, 28))
plt.figure("Image")
plt.imshow(image_array, cmap='gray', interpolation='None')
