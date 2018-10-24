# coding=utf-8
# author: BebDong
# 2018.10.23

import neural_network as nn
import numpy
import time

# 便于计算执行时间
start = time.process_time()

# 指定神经网络的结构。隐藏层节点个数不唯一
input_nodes, hidden_nodes, output_nodes = 784, 100, 10

# 指定权重更新的学习率
learning_rate = 0.3

# 创建神经网络的实例
network = nn.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取训练数据，只读方式
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
# 当数据集很大时，应当分批读入内存。这里仅100条记录，则一次性全部读入内存
training_data_list = training_data_file.readlines()
training_data_file.close()

# 训练神经网络，epochs次
epochs = 2
for e in range(epochs):
    for record in training_data_list:
        # 缩放输入
        all_pixels = record.split(',')
        scaled_inputs = (numpy.asfarray(all_pixels[1:]) / 255.0 * 0.99) + 0.01
        # 创建目标输出
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_pixels[0])] = 0.99
        network.training(scaled_inputs, targets)
        pass
    pass

# 读取测试数据集
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# 测试训练好的神经网络
# 初始化一个数据结构用于记录神经网络的表现
scorecard = []
# 遍历测试数据集
for record in test_data_list:
    # 打印预期输出
    all_pixels = record.split(',')
    correct_label = int(all_pixels[0])
    # 查询神经网络
    inputs = (numpy.asfarray(all_pixels[1:])/255.0 * 0.99) + 0.01
    outputs = network.query(inputs)
    answer = numpy.argmax(outputs)
    # 更新神经网络的表现
    if answer == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    pass

# 打印得分及运行时间
print("time: ", time.process_time()-start)
print("performance: ", sum(scorecard) / len(scorecard))
