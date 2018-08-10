#!/usr/bin/env python
# encoding: utf-8

from tf_kmeans import TFKMeans
import tensorflow as tf


class TFRBFNet(object):
    '''
    TensorFlow 版的RBF+KMEANS
    其中RBF实现两种训练方式:
        - 使用梯度下降训练隐藏曾到输出的权重,类似BPNN
        - 直接使用Linear Regression, 求beta的线性方程解
    '''

    def __init__(self, k, delta=0.1):
        '''
            _delta: rbf的高斯扩展参数
            _beta: RBF层到输出层的权重
            _input_n: 输入神经元个数
            _hidden_num: 隐层神经元个数
            _output_n: 输出神经元个数
            _max_iter: 迭代次数
            trainWithBeta: 第二种训练方式
        '''
        self._delta = delta
        self._beta = None
        self._input_n = 0
        self._hidden_num = k
        self._output_n = 0
        self.max_iter = 5000
        self.trainWithBeta = True
        self.sess = tf.Session()

    def setup(self, ni, nh, no): # 5维，k=10，no=1维 # 自己构建的：20维，50维，1维
        '''
        网络建立
        '''
        self._input_n = ni # 0
        self._hidden_num = nh #k=10
        self._output_n = no # 0

        self.input_layer = tf.placeholder(tf.float32, [None, self._input_n], name='inputs_layer')
        self.output_layer = tf.placeholder(tf.float32, [None, self._output_n], name='outputs_layer')

        self.getHiddenCenters(self._inputs)
        self.hidden_centers = tf.constant(self.hidden_centers, name="hidden")
        self.hidden_layer = self.rbfunction(self.input_layer, self.hidden_centers)

    def fit(self, inputs, outputs):
        '''
        训练
            inputs: 输入数据
            ouputs: 输出数据
        '''
        self._inputs = inputs
        self._outputs = outputs
        self.setup(inputs.shape[1], self._hidden_num, outputs.shape[1])
        # print('inputs.shape[1] : ',inputs.shape[0],inputs.shape[1])
        # print('outputs.shape[1] : ',outputs.shape[1])

        self.sess.run(tf.global_variables_initializer())
        if self.trainWithBeta: # trainWithBeta: 第二种训练方式
            self.LinearTrain()
        else:
            self.gradientTrain()


    def LinearTrain(self):
        '''
        直接使用公式求解隐层到输出层的beta参数，其中涉及到求逆操作
        '''
        beta = tf.matrix_inverse(tf.matmul(tf.transpose(self.hidden_layer), self.hidden_layer))
        beta_1 = tf.matmul(beta, tf.transpose(self.hidden_layer))
        beta_2 = tf.matmul(beta_1, self.output_layer)

        self._beta = self.sess.run(beta_2, feed_dict={self.input_layer: self._inputs, self.output_layer: self._outputs})
        # 预测输出
        self.predictionWithBeta = tf.matmul(self.hidden_layer, self._beta)


    def gradientTrain(self):
        '''
        梯度下降法训练RBF隐层->输出层的参数
        '''
        self.trainWithBeta = False

        # 最后预测的输出
        self.predictionWithGD = self.addLayer(self.hidden_layer, self._hidden_num, self._output_n)
        # 平方损失误差
        self.loss = tf.reduce_mean(tf.square(self.predictionWithGD - self.output_layer))
        # 梯度下降优化
        self.optimizer = tf.train.GradientDescentOptimizer(0.09).minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())
        for i in range(self.max_iter):
            self.sess.run(self.optimizer, feed_dict={self.input_layer:self._inputs, self.output_layer: self._outputs})
            if i%100 == 0:
                print ('iter: ',i, 'loss:', self.sess.run(self.loss, feed_dict={self.input_layer: self._inputs, self.output_layer: self._outputs}))


    def predict(self, inputs):
        '''
        预测函数,根据不同的训练方式，选择不同的预测函数
        '''
        if self.trainWithBeta:
            # 直接计算beta
            return self.sess.run(self.predictionWithBeta, feed_dict={self.input_layer: inputs})
        else:
            # 梯度下降方式训练权重
            return self.sess.run(self.predictionWithGD, feed_dict={self.input_layer: inputs})

    def addLayer(self, inputs, inputs_size, output_size, activefunc=None):
        '''
        添加隐层->输出层
        只有一层，参数较少，没有必要加正则以及Dropout

        '''

        self.weights = tf.Variable(tf.random_uniform([inputs_size, output_size],-1.0,1.0, tf.float32))
        self.biases = tf.Variable(tf.constant(0.1, tf.float32,shape=[1,output_size]))
        result = tf.matmul(inputs, self.weights)
        if activefunc is None:
            return result
        else:
            return activefunc(result)

    def getHiddenCenters(self, inputs):
        '''
        使用TF版的Kmeans基于欧式距离进行无监督聚类
        获得中心
        '''
        kms = TFKMeans(self._hidden_num, session=self.sess)
        kms.train(tf.constant(inputs))
        self.hidden_centers = kms.centers

    def rbfunction(self, x, c):
        e_c = tf.expand_dims(c, 0)
        e_x = tf.expand_dims(x, 1)
        return tf.exp(-self._delta * tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(e_c, e_x)), 2)))

