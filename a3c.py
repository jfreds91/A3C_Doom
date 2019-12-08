import threading
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.signal
from random import choice
from time import sleep
from time import time

from helper import *
from vizdooom import *

class AC_Network():
    def __init__(self, s_size, a_size, scope, trainer):
        with tf.compat.v1.variable_scope(scope):
            # input and visual encoding layers
            self.inputs = tf.compat.v1.placeholder(shape=[None, s_size], dtype=tf.float32)
            self.imageIn = tf.reshape(self.inputs, shape=[-1, 84, 84, 1])
            self.conv1 = tf.compat.v1.keras.layers.conv2d(input = self.imageIn,
                                                        filters = 16,
                                                        kernel_size = [8,8],
                                                        stride = [4,4]
                                                        padding = 'VALID',
                                                        activity_regularizer = 'relu')
            self.conv1 = tf.compat.v1.keras.layers.conv2d(input = self.imageIn,
                                                        filters = 32,
                                                        kernel_size = [4,4],
                                                        stride = [2,2]
                                                        padding = 'VALID',
                                                        activity_regularizer = 'relu')
            