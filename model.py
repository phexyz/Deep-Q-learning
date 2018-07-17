from utils import *
import numpy as np
import tensorflow as tf

class Convolution(object):

    def __init__(self, action_number):
        """This is a convolution neural network that output the action value of possible actions given a state"""

        self._action_number = action_number

        self.build_conv_net()

    def build_conv_net(self):

        # input 84 * 84 * 4
        with tf.name_scope(name="layer_input") as scope:
            self._input = tf.placeholder(tf.float32, shape=(None, 84, 84, 4))

        ## layer 1
        # 16 8 * 8 filters with stride of 4
        with tf.name_scope(name="layer_1") as scope:
            self._conv1 = tf.layers.conv2d(self._input, filters=16, kernel_size=[8, 8], strides=[4, 4], padding="valid")
            # RELU
            self._relu1 = tf.nn.relu(self._conv1)

        ## layer 2
        #32 4 * 4 filters with stride of 2
        with tf.name_scope(name="layer_2") as scope:
            self._conv2 = tf.layers.conv2d(self._relu1, filters=32, kernel_size=[4, 4], strides=[2, 2], padding="valid")
            # RELU
            self._relu2 = tf.nn.relu(self._conv2)

        ## layer 3
        # 256 RELU units
        with tf.name_scope(name="layer_3") as scope:
            self._fc3 = tf.layers.dense(tf.contrib.layers.flatten(self._relu2), units=256, activation=tf.nn.relu)
        ## output layer
        # fully connected layer that outputs the probability of each actions
        with tf.name_scope(name="layer_output") as scope:
            self._W4 = tf.get_variable(name="weights", shape=[256, self._action_number])
            self._b4 = tf.get_variable(name="bias", shape=[1, self._action_number])
            self._logits = tf.matmul(self._fc3, self._W4)
            self._output = tf.nn.softmax(self._logits)



class Q_learning_model(object):

    def __init__(self, state_elements, phi):
        """state_elements is a list of elements that would adequately describe a state"""

        self.phi = phi
        self._feature = map(self.phi, state_elements)

