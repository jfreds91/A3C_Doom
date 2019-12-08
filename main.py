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
from a3c import *
from vizdooom import * # http://vizdoom.cs.put.edu.pl/tutorial

# re-creating the work done by https://github.com/awjuliani/DeepRL-Agents/blob/master/A3C-Doom.ipynb

