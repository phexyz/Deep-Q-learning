from utils import *
import numpy as np
import tensorflow as tf

class Convolution(object):

    def __init__(self, action_number):
        """This is a convolution neural network that output the action value of possible actions given a state"""

        self._action_number = action_number

        self.build_conv_net()
        self._sess = tf.Session()

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
            self._target = tf.placeholder(tf.float32, tf.shape(self._output))

        self._optimize = tf.train.AdamOptimizer().minimize(loss=tf.losses.mean_squared_error(self._target, self._output))

    def feed_forward(self, input):

        with self._sess as sess:
            return sess.run(self._output, feed_dict={self._input : input})

    def gradient_descent(self, target, input):

        with self._sess as sess:
            sess.run(self._optimize, feed_dict={self._input: input
                                                self._target: target})



class Q_learning_model(object):

    def __init__(self, state_elements, phi):
        """state_elements is a list of elements that would adequately describe a state"""

        self.phi = phi
        self._feature = list(map(self.phi, state_elements))

        self._memory = []
        self._Q = Convolution(action_number=5)

        # number of total episodes to go through
        self._M = 10
        self._epsilon = 0
        self._gamma = 0

    def train(self):

        for episode in range(1, self._M):
            sequence = self.initialize_sequence()
            features = self.phi(sequence)

            total_time_step = 0

            for time_step in range(1, total_time_step + 1):
                action = self.epsilon_greedy()
                reward, next_state = self.step(action)

                next_sequence = [sequence, action, next_state]
                next_features = self.phi(sequence)
                self._memory.append((features, action, reward, next_features))

                mini_batch = np.random.choice(self._memory)

                features_j, action_j, reward_j, features_j_plus_1,  = mini_batch
                y_j = reward_j if TERMINAL else reward_j + self._gamma * np.amax(self._Q.feed_forward(features_j_plus_1))

                self._Q.gradient_descent(y_j)


    def initialize_sequence(self):


    def epsilon_greedy(self):
        pass

    def step(self, action):

        if action == 0:
            reward = 1

        return reward, next_state


class State(object):

    def __init__(self):

        self._dealer_sum = 0
        self._player_sum = 0
        self._player_first_card = 0


class Environment(object):

    def draw_card(self):

        return np.random.randint(1, 10)

    def step(self, state, action):

        if action == 0:

            if state._dealer_sum < 17:

                state._dealer_sum +=



        elif action == 1:



