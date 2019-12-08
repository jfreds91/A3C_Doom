import threading
import multiprocessing
import numpy as np
import tensorflow as tf 
import scipy.signal
from random import choice
from time import sleep
from time import time
from PIL import Image

from a3c import *

def update_target_graph(from_scope, to_scope):
    # copies one set of variables to another... thread?
    # used to set worker network parameters to those of the global network... and vice versa?

    from_vars = tf.Graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=from_scope.name)
    to_vars = tf.Graph.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=to_scope.name)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def process_frame(frame):
    # process doom screen image to produce cropped and resized image
    s = frame[10:-10, 30:-30]
    s = np.array(Image.fromarray(s).resize())
    #s = misc.imresize(s, [84,84]) # resize
    s = np.reshape(s, [np.prod(s.shape)]) / 255.0 # flatten and rescale
    return s

def discount(x, gamma):
    # discounting function used to calculate discounted returns
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def normalize_columns_initializer(std=1.0):
    # used to initialize weights for policy and value output layers
    def _initializer(shape, dtype=None, partion_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.square(out).sum(axis=0, keepdims=True)
        return tf.constant(out)
    return _initializer


