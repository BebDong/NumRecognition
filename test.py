# coding=utf-8
# author: BebDong
# 10/23/18

import numpy
import matplotlib.pyplot as plt

# 直接使用plt.imshow无法显示图片，需要导入pylab包
import pylab

# 打开并读取文件
data_file = open("mnist_dataset/mnist_train_100.csv")
data_list = data_file.readlines()
data_file.close()

# 拆分绘制28*28图形
all_pixels = data_list[0].split(',')
image_array = numpy.asfarray(all_pixels[1:]).reshape((28, 28))
plt.figure("Image")
plt.imshow(image_array, cmap='gray', interpolation='None')
pylab.show()

# 缩放输入数据。0.01的偏移量避免0值输入
scaled_inputs = (numpy.asfarray(all_pixels[1:])/255.0 * 0.99) + 0.01

# 构建目标矩阵。sigmoid函数无法取端点值0或者1，使用0.01代替0，0.99代替1
output_nodes = 10
# 产生0值输出矩阵
targets = numpy.zeros(output_nodes) + 0.01
# 将字符串转换为整数，并设置激发节点
targets[int(all_pixels[0])] = 0.99
