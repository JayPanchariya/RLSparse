from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax
import numpy as np
import tensorflow as tf
import tf_agents
import os,gc

### Environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment


from tf_agents.networks import actor_distribution_network
from tf_agents.networks.categorical_projection_network import CategoricalProjectionNetwork
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

#import agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.utils import value_ops
from tf_agents.trajectories import StepType

#import replay buffer
from tf_agents import replay_buffers as rb

#import driver
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver

# my function
import environmentRL as envRL
import function2D as fun
import utills as utills

import train as model

#To limit TensorFlow to CPU
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# cpus = tf.config.experimental.list_physical_devices('CPU') 
# tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
# tf_agents.system.multiprocessing.enable_interactive_mode()
# gc.collect()




if __name__ == '__main__':
    RL=model.RLOpt() 
    RL.test_policy()