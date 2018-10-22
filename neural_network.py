# coding=utf-8
# author: BebDong
# 10/23/18

import numpy
import scipy.special


# neural network definition
class NeuralNetwork:

    # initialise the neural network
    def __init__(self, numInputNodes, numHiddenNodes, numOutputNodes, learningRate):
        # 单隐藏层示例，设置各层的节点个数
        self.numInputNodes = numInputNodes
        self.numHiddenNodes = numHiddenNodes
        self.numOutputNodes = numOutputNodes

        # 权重更新时的学习率
        self.learningRate = learningRate

        # 正态分布初始化权重
        self.weightInputHidden = numpy.random.normal(0.0, pow(self.numHiddenNodes, -0.5),
                                                     (self.numHiddenNodes, self.numInputNodes))
        self.weightHiddenOutput = numpy.random.normal(0.0, pow(self.numOutputNodes, -0.5),
                                                      (self.numOutputNodes, self.numHiddenNodes))

        # 激活函数(lambda创建匿名函数)
        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    # train the network using training data set
    def training(self, inputs_list, targets_list):
        # 第一，同query()函数
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.weightInputHidden, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weightHiddenOutput, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        # 反向传播误差到隐藏层
        hidden_errors = numpy.dot(self.weightHiddenOutput.T, output_errors)

        # 更新隐藏层和输出层之间的权重
        self.weightHiddenOutput += self.learningRate * numpy.dot((output_errors * final_outputs *
                                                                  (1.0 - final_outputs)),
                                                                 numpy.transpose(hidden_outputs))
        # 更新输入层和隐藏层之间的权重
        self.weightInputHidden += self.learningRate * numpy.dot((hidden_errors * hidden_outputs *
                                                                 (1.0 - hidden_outputs)),
                                                                numpy.transpose(inputs))

        pass

    # query the network using test data set
    def query(self, inputs_list):
        # 将输入一维数组转化成二维，并转置
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 计算到达隐藏层的信号，即隐藏层输入
        hidden_inputs = numpy.dot(self.weightInputHidden, inputs)
        # 计算隐藏层输出，即经过sigmoid函数的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算到达输出层的信号，即输出层的输入
        final_inputs = numpy.dot(self.weightHiddenOutput, hidden_outputs)
        # 计算最终的输出
        final_outputs = self.activation_function(final_inputs)

        return final_outputs
