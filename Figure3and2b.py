#NOTE: The objective function in Figure 3 in the paper is denoted as \mathcal{L}(x), but in this code the objective is denoted as f(x).

#import driver
from tf_agents.drivers import py_driver
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as mse

#import environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#import replay buffer
from tf_agents import replay_buffers as rb

#import agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.utils import value_ops
from tf_agents.trajectories import StepType


#other used packages
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from matplotlib.axes import Axes as ax
import numpy as np
import tensorflow as tf
import tf_agents
from tf_agents.networks import actor_distribution_network
from tf_agents.specs import array_spec
from tf_agents.utils import common
from tf_agents.trajectories import time_step as ts
from tf_agents.networks.categorical_projection_network import CategoricalProjectionNetwork
#from custom_normal_projection_network import NormalProjectionNetwork
import os,gc
import pygad
from pyswarms.single.global_best import GlobalBestPSO
import time
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

import function2D as fun
import utills
#To limit TensorFlow to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()


def logInit(log_dir='log1'):
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    # Generate timestamp for log filename
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = os.path.join(log_dir, f'logfile_{timestamp}.log')

    # Set up logger
    logger = logging.getLogger('custom_logger')
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the function is called again
    if not logger.handlers:
        file_handler = logging.FileHandler(log_filename)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Now you can log messagess
# logging.info("This is an info log message.")
# logging.debug("This is a debug log message.")
# logging.error("This is an error log message.")

#Functions needed for training
def extract_episode(traj_batch,epi_length,attr_name = 'observation'):
    """
    This function extract episodes (each episode consists of consecutive time_steps) from a batch of trajectories.
    Inputs.
    -----------
    traj_batch:replay_buffer.gather_all(), a batch of trajectories
    epi_length:int, number of time_steps in each extracted episode
    attr_name:str, specify which data from traj_batch to extract
    
    Outputs.
    -----------
    tf.constant(new_attr,dtype=attr.dtype), shape = [new_batch_size, epi_length, state_dim]
                                        or shape = [new_batch_size, epi_length]
    """
    attr = getattr(traj_batch,attr_name)
    original_batch_dim = attr.shape[0]
    traj_length = attr.shape[1]
    epi_num = int(traj_length/epi_length) #number of episodes out of each trajectory
    batch_dim = int(original_batch_dim*epi_num) #new batch_dim
    
    if len(attr.shape)==3:
        stat_dim = attr.shape[2]
        new_attr = np.zeros([batch_dim, epi_length, state_dim])
    else:
        new_attr = np.zeros([batch_dim, epi_length])
        
    for i in range(original_batch_dim):
        for j in range(epi_num):
            new_attr[i*epi_num+j] = attr[i,j*epi_length:(j+1)*epi_length].numpy()
        
    return tf.constant(new_attr,dtype=attr.dtype)

###############
#Define the objective f to be maximized
N = 2   #This the dimension number
state_dim = N

# ############### -- The objective function. In Figure 3a, f(x) is denoted as \mathcal{L}(x).
# def f(x):
#     return -(2-np.sum(np.cos(10*x)) + 0.05*np.sum(100*x**2))+10
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
# ######################### - Plot the Objective Function in Figure 3
# x1_min = -5.0
# x2_min = -5.0
# x1_max = 5.0
# x2_max = 5.0
# x_num = 500

# X1 = np.linspace(x1_min,x1_max,x_num)
# X2 = np.linspace(x2_min,x2_max,x_num)
# X1, X2 = np.meshgrid(X1, X2)
# Y = np.zeros((x_num,x_num))
# for i in range(x_num):
#     for j in range(x_num):
#         Y[i,j] = -fun.f( x=np.array([X1[i,j],X2[i,j]]) )

# ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
# plt.rcParams['text.usetex'] = True
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.bf'] = 'STIXGeneral:italic:bold'
# plt.rcParams['mathtext.it'] = 'STIXGeneral:italic'
# # Plot the 3D surface
# #ax.title.set_text(r"$f(x)=-ln((x-m_1)^2+0.00001)-ln((x-m_2)^2+0.01)$")
# ax.plot_surface(X1, X2, Y, cmap=cm.coolwarm, edgecolor='royalblue')
# ax.set(xlim=(x1_min, x1_max), ylim=(x2_min, x2_max), zlim=(np.min(Y), np.max(Y)-1.5))
# ax.set_xlabel('$x_1$', fontsize=25)
# ax.set_ylabel('$x_2$', fontsize=25)
# ax.tick_params(labelsize=20)
# #ax.text(-1.2, -1, 16.6, "$\mathcal{L}(x_1,x_2)=8-\cos(10x_1)-\cos(10x_2)-5x_1^2-5x_2^2$",
# #        color='black', size=20)
# plt.savefig("escape-2D-objective2d.png", bbox_inches='tight', transparent=True)
# # plt.show()


################
#Set initial x-value
r = np.random.RandomState(0)
x0_reinforce = np.array([0.5,-1.0])
sub_episode_length = 30 #number of time_steps in a sub-episode. 
episode_length = sub_episode_length*6  #an trajectory starts from the initial timestep and has no other initial timesteps
                                      #each trajectory will be split to multiple episodes
env_num = 10  #Number of parallel environments, each environment is used to generate an episode
print('x0', x0_reinforce)

################
#Set hyper-parameters for REINFORCE-OPT
generation_num = 16000  #number of theta updates for REINFORCE-IP, also serves as the number
                      #of generations for GA, and the number of iterations for particle swarm optimization

disc_factor = 1.0
alpha = 0.2 #regularization coefficient
param_alpha = 0.15 #regularization coefficient for actor_network #tested value: 0.02
sub_episode_num = int(env_num*(episode_length/sub_episode_length)) #number of sub-episodes used for a single update of actor_net params
logging.info(f"sub episode lenght(steps): {sub_episode_length}")
logging.info(f"on trajectory length with 6 batches: {episode_length }")  
logging.info(f"env number: {env_num}")
logging.info(f"number of sub_episodes used for a single param update: {sub_episode_num}")
logging.info(f"generation number: {generation_num}")  

print("number of sub_episodes used for a single param update:", sub_episode_num)

#Learning Schedule = initial_lr * (C/(step+C))
class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = C
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.initial_learning_rate * self.C / (self.C + step)
    
lr = lr_schedule(initial_lr=0.00002, C=50000)   
opt = tf.keras.optimizers.SGD(learning_rate=lr)
#opt = tf.keras.optimizers.Adam( )
logging.info(f"learning rate: {opt}")  

train_step_num = 0

act_min = -1
act_max = 1
step_size = 0.05
def compute_reward(x):
    return fun.f(x)

logging.info(f"action minimum: {act_min}")
logging.info(f"action maximum: {act_max}")
logging.info(f"step size: {step_size}")
# logging.info(f"gradient step size: {step_numGe}")

def test_policy(self):
    self.test_buffer.clear()
    self.test_driver.run(self.eval_env.reset())  # run the current policy
    experience = self.test_buffer.gather_all()
    rl_trajectory = experience.observation.numpy()[0]
    
    ga_trajectory = self.gradientOpt()
    utills.plotTrajectory(rl_trajectory, ga_trajectory, self.step_numGe, self.path)
    
############# - Create the Gradient Ascent Trajectory
def gradientOpt(self):
    ga_trajectory = [self.x0_reinforce]
    current_x = tf.Variable(self.x0_reinforce)
    # current_x = tf.Variable([0.5,-1])
    
    for i in range(self.step_numGe):
        with tf.GradientTape() as tape:
            y = fun.f(current_x) ##-(2-tf.reduce_sum(tf.math.cos(10*current_x)) + 0.05*tf.reduce_sum(100*current_x**2))+10
        gradient = tape.gradient(y, current_x)
        norm_gradient = gradient.numpy()/np.sqrt(np.sum(gradient.numpy()**2))
        current_x.assign(current_x.numpy() + norm_gradient*self.step_size)
        ga_trajectory.append(current_x.numpy())
    return ga_trajectory
#Define the Environment
class Env(py_environment.PyEnvironment):
    def __init__(self):
        '''The function to initialize an Env obj.
        '''
        #Specify the requirement for the value of action, (It is a 2d-array for this case)
        #which is an argument of _step(self, action) that is later defined.
        #tf_agents.specs.BoundedArraySpec is a class.
        #_action_spec.check_array( arr ) returns true if arr conforms to the specification stored in _action_spec
        self._action_spec = array_spec.BoundedArraySpec(
                            shape=(state_dim,), dtype=np.int32, minimum=0, maximum=act_max-act_min, name='action') #a_t is an 2darray
    
        #Specify the format requirement for observation (It is a 2d-array for this case), 
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(state_dim,), dtype=np.float32, name='observation') #default max and min is None
        self._state = np.array(x0_reinforce,dtype=np.float32)
        #self.A = mat
        self._episode_ended = False
        #stop_threshold is a condition for terminating the process for looking for the solution
        #self._stop_threshold = 0.01
        self._step_counter = 0

    def action_spec(self):
        #return the format requirement for action
        return self._action_spec

    def observation_spec(self):
        #return the format requirement for observation
        return self._observation_spec

    def _reset(self):
        self._state = np.array(x0_reinforce,dtype=np.float32)  #initial state
        self._episode_ended = False
        self._step_counter = 0
        
        #Reward
        initial_r = np.float32(0.0)
        
        #return ts.restart(observation=np.array(self._state, dtype=np.float32))
        return ts.TimeStep(step_type=StepType.FIRST, 
                           reward=initial_r, 
                           discount=np.float32(disc_factor), 
                           observation=np.array(self._state, dtype=np.float32)
                           )
    
    def set_state(self,new_state):
        self._state = new_state
    
    def get_state(self):
        return self._state
    
    def _step(self, action):
        '''
        The function for the transtion from (S_t, A_t) to (R_{t+1}, S_{t+1}).
    
        Input.
        --------
        self: contain S_t.
        action: A_t.
    
        Output.
        --------
        an TimeStep obj, TimeStep(step_type_{t+1}, R_{t+1}, discount_{t+1}, observation S_{t+1})
        ''' 
        # Suppose that we are at the beginning of time t 
        
        ################## --- Determine whether we should end the episode.
        if self._episode_ended:  # its time-t value is set at the end of t-1
            return self.reset()
        # Move on to the following if self._episode_ended=False
        
        
        ################# --- Compute S_{t+1} 
        #Note that we set the action space as a set of non-negative vectors
        #action-act_max converts the set to the desired set of negative vectors.
        normalized_act = (action-act_max)/np.sqrt((action-act_max)**2+0.0000001) 
        self._state = self._state + normalized_act*step_size    
        self._step_counter +=1
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = compute_reward(self._state)
        
        #Set conditions for termination
        if self._step_counter>=sub_episode_length-1:
            self._episode_ended = True  #value for t+1

        #Now we are at the end of time t, when self._episode_ended may have changed
        if self._episode_ended:
            #if self._step_counter>100:
            #    reward += np.float32(-100)
            #ts.termination(observation,reward,outer_dims=None): Returns a TimeStep obj with step_type set to StepType.LAST.
            return ts.termination(np.array(self._state, dtype=np.float32), reward=R)
        else:
            #ts.transition(observation,reward,discount,outer_dims=None): Returns 
            #a TimeStep obj with step_type set to StepType.MID.
            return ts.transition(np.array(self._state, dtype=np.float32), reward=R, discount=disc_factor)


if __name__ == "__main__":
    
    logger = logInit()
    
    X1, X2, Y=utills.create2Dfunction(x1Lim=(-1, 1), x2Lim=(-1, 1), N=200)

    #Create a sequence of parallel environments and batch them, for later use by driver to generate parallel trajectories.
    parallel_env = ParallelPyEnvironment(env_constructors=[Env]*env_num, 
                                        start_serially=False,
                                        blocking=False,
                                        flatten=False
                                        )
    #Use the wrapper to create a TFEnvironments obj. (so that parallel computation is enabled)
    train_env = tf_py_environment.TFPyEnvironment(parallel_env, check_dims=True) #instance of parallel environments
    eval_env = tf_py_environment.TFPyEnvironment(Env(), check_dims=False) #instance
    # train_env.batch_size: The batch size expected for the actions and observations.  
    print('train_env.batch_size = parallel environment number = ', env_num)

    logger.info(f"train_env.batch_size = parallel environment number =  {env_num}")

    #actor_distribution_network outputs a distribution
    #it is a neural net which outputs the parameter (mean and sd, named as loc and scale) for a normal distribution
    tf.random.set_seed(0)
    actor_net = actor_distribution_network.ActorDistributionNetwork(   
                                            train_env.observation_spec(),
                                            train_env.action_spec(),
                                            fc_layer_params=(16,16,16), #Hidden layers
                                            seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                                            discrete_projection_net=CategoricalProjectionNetwork,
                                            activation_fn = tf.math.tanh,
                                            #continuous_projection_net=(NormalProjectionNetwork)
                                            )

    #Create the  REINFORCE_agent
    train_step_counter = tf.Variable(0)
    tf.random.set_seed(0)
    REINFORCE_agent = reinforce_agent.ReinforceAgent(
            time_step_spec = train_env.time_step_spec(),
            action_spec = train_env.action_spec(),
            actor_network = actor_net,
            value_network = None,
            value_estimation_loss_coef = 0.2,
            optimizer = opt,
            advantage_fn = None,
            use_advantage_loss = False,
            gamma = 1.0, #discount factor for future returns
            normalize_returns = False, #The instruction says it's better to normalize
            gradient_clipping = None,
            entropy_regularization = None,
            train_step_counter = train_step_counter
            )
        
    REINFORCE_agent.initialize()
    
    # Checkpoint setup
    # checkpoint_dir = 'checkpoints/'
    # checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
    # checkpoint = tf.train.Checkpoint(
    #     step=tf.Variable(0),
    #     optimizer=opt,
    #     agent=REINFORCE_agent,
    #     actor_network=actor_net
    # )
    # checkpoint_manager = tf.train.CheckpointManager(checkpoint,checkpoint_dir, max_to_keep=3)
    # if checkpoint_manager.latest_checkpoint:
    #     checkpoint.restore(checkpoint_manager.latest_checkpoint)
    #     logger.info(f"Restored from {checkpoint_manager.latest_checkpoint}")
    # else:
    #     logger.info(f"Initializing from scratch.")
        


    #################
    #replay_buffer is used to store policy exploration data
    #################
    replay_buffer = rb.TFUniformReplayBuffer(
                    data_spec = REINFORCE_agent.collect_data_spec,  # describe spec for a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                    batch_size = env_num,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                            # One batch corresponds to one parallel environment
                    max_length = episode_length*100    # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                        # if exceeding this number previous trajectories will be dropped
    )

    #test_buffer is used for evaluating a policy
    test_buffer = rb.TFUniformReplayBuffer(
                    data_spec= REINFORCE_agent.collect_data_spec,  # describe a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                    batch_size= 1,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                        # train_env.batch_size: The batch size expected for the actions and observations.  
                                                        # batch_size: Batch dimension of tensors when adding to buffer. 
                    max_length = episode_length         # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                        # if exceeding this number previous trajectories will be dropped
    )


    #A driver uses an agent to perform its policy in the environment.
    #The trajectory is saved in replay_buffer
    collect_driver = DynamicEpisodeDriver(
                                        env = train_env, #train_env contains parallel environments (no.: env_num)
                                        policy = REINFORCE_agent.collect_policy,
                                        observers = [replay_buffer.add_batch],
                                        num_episodes = sub_episode_num   #SUM_i (number of episodes to be performed in the ith parallel environment)
                                        )

    #For policy evaluation
    test_driver = py_driver.PyDriver(
                                        env = eval_env, #PyEnvironment or TFEnvironment class
                                        policy = REINFORCE_agent.policy,
                                        observers = [test_buffer.add_batch],
                                        max_episodes=1, #optional. If provided, the data generation ends whenever
                                                        #either max_steps or max_episodes is reached.
                                        max_steps=sub_episode_length
                                    )



    # Please also see the metrics module for standard implementations of different
    # metrics.

    ######## Train REINFORCE_agent's actor_network multiple times.
    update_num = generation_num 
    eval_intv = 100 #number of updates required before each policy evaluation
    logger.info(f"evaluation interval: {eval_intv}")
    REINFORCE_logs = [] #for logging the best objective value of the best solution among all the solutions used for one update of theta
    final_reward = -1000
    logger.info(f"final reward: {final_reward}")
    plot_intv = 1
    logger.info(f"plot interval: {plot_intv}")
    
    tf.random.set_seed(0)
    for n in range(0,update_num):
        print(n)
        t0 = tf.timestamp()
        #Generate Trajectories
        replay_buffer.clear()
        collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
        t1 = tf.timestamp()
        logger.info(f"time taken after Replay buffer annd self.collect_driver ={t1-t0}")
        
        experience = replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
        rewards = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
        observations = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)
        actions = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
        step_types = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'step_type')
        discounts = extract_episode(traj_batch=experience,epi_length=sub_episode_length,attr_name = 'discount')
        logger.info(f"rewards from experience ={rewards}")
        
        time_steps = ts.TimeStep(step_types,
                                tf.zeros_like(rewards),
                                tf.zeros_like(discounts),
                                observations
                                )
        t2= tf.timestamp()
            # print(" inside actions:", actions)
        logger.info(f"time taken after 6 extract_episods and time_steps={t2-t0}")
        
        rewards_sum = tf.reduce_sum(rewards, axis=1) #shape=(sub_episode_num,)
        logger.info(f"rewards sum ={rewards_sum}")
        
        with tf.GradientTape() as tape:
            #trainable parameters in the actor_network in REINFORCE_agent
            variables_to_train = REINFORCE_agent._actor_network.trainable_weights
        
            ###########Compute J_loss = -J
            actions_distribution = REINFORCE_agent.collect_policy.distribution(
                                time_steps, policy_state=None).action
        
            #log(pi(action|state)), shape = (batch_size, epsode_length)
            action_log_prob = common.log_probability(actions_distribution, 
                                                    actions,
                                                    REINFORCE_agent.action_spec)
        
            J = tf.reduce_sum(tf.reduce_sum(action_log_prob,axis=1)*rewards_sum)/sub_episode_num
            
            ###########Compute regularization loss from actor_net params
            regu_term = tf.reduce_sum(variables_to_train[0]**2)
            num = len(variables_to_train) #number of vectors in variables_to_train
            for i in range(1,num):
                regu_term += tf.reduce_sum(variables_to_train[i]**2)
            
            total = -J + param_alpha*regu_term
            logger.info(f"Total Loss ={total}")
        
        #update parameters in the actor_network in the policy
        grads = tape.gradient(total, variables_to_train)
        grads_and_vars = list(zip(grads, variables_to_train))
        opt.apply_gradients(grads_and_vars=grads_and_vars)
        train_step_num += 1
        
        batch_rewards = rewards.numpy()
        batch_rewards[:,-1] = -np.power(10,8) #The initial reward is set as 0, we set it as this value to not affect the best_obs_index 
        best_step_reward = np.max(batch_rewards)
        best_step_index = [int(batch_rewards.argmax()/sub_episode_length),batch_rewards.argmax()%sub_episode_length+1]
        best_step = observations[best_step_index[0],best_step_index[1],:] #best solution
        #best_step_reward = f(best_solution)
        print("best reward",np.max(batch_rewards) )
        logger.info(f"batch rewards :: {batch_rewards.shape}, Best reward(max): {np.max(batch_rewards)}")
        avg_step_reward = np.mean(batch_rewards[:,0:-1])
        REINFORCE_logs.append(best_step_reward)
        # checkpoint.step.assign(n)
        # checkpoint_manager.save()
        # logger.info(f"Checkpoint saved at step :{int(checkpoint.step)}")
        
        if best_step_reward>final_reward:
            #print("final reward before udpate:",final_reward)
            final_reward = best_step_reward
            final_solution = best_step.numpy()
            #print("final reward after udpate:",final_reward)
            #print('updated final_solution=', final_solution)
        
        #print(compute_reward(best_obs,alpha))
        if n%eval_intv==0:
            
            # print("train_step no.=",train_step_num)
            # print('best_solution of this generation=', best_step.numpy())
            # print('best step reward=',best_step_reward.round(3),f(best_step.numpy()))
            # print('avg step reward=', round(avg_step_reward,3))
            # #print('episode of rewards', rewards.round(3))
            # #print('act_std:', actions_distribution.stddev()[0,0]  )
            # #print('act_mean:', actions_distribution.mean()[0,0] ) #second action mean
            # print('best_step_index:',best_step_index)
            # print(observations[0])
            # print(' ')
            
            replay_buffer.clear()
            logger.info(f"train_step no.={train_step_num}")
            logger.info(f"best_solution of this generation= {best_step.numpy()}")
            logger.info(f"best step reward= {best_step_reward.round(3),fun.f(best_step.numpy())}")
            logger.info(f"avg step reward= {round(avg_step_reward,3)}")
            #logger.info(f"episode of rewards', rewards.round(3))
            #logger.info(f"act_std:', actions_distribution.stddev()[0,0]  )
            #logger.info(f"act_mean:', actions_distribution.mean()[0,0] ) #second action mean
            logger.info(f"best_step_index: {best_step_index}")
            logger.info(f"observation: {observations[0]}")
            # logger.info(f" ')
        
    
    
        if n%plot_intv==0:
            pathInit = os.path.join("Results", "trajInt2")
            os.makedirs(pathInit, exist_ok=True)
            test_buffer.clear()
            test_driver.run(eval_env.reset())  #generate batches of trajectories with agent.collect_policy, and save them in replay_buffer
            experience = test_buffer.gather_all()  #get all the stored items in replay_buffer
            rl_trajectory = experience.observation.numpy()[0]
            
            #Plot
            fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
            CS = ax.contour(X1, X2, Y)
            # cp = plt.contour(X1, X2, Y, levels=np.logspace(-1, 3, 20), cmap='jet')
            # cp = plt.contour(X1, X2, Y,levels= -np.logspace(-1, 3, 20)[::-1], cmap=plt.get_cmap('jet_r'))
            # plt.clabel(cp, inline=True, fontsize=8)
            
            i = 0
            plt.arrow(x=rl_trajectory[i][0],
            y=rl_trajectory[i][1],
            dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
            dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
            color='r',linestyle='--',width=0.01, label='REINFORCE-OPT')

            for i in range(0, len(rl_trajectory)-1):   
                plt.arrow(x=rl_trajectory[i][0],
                y=rl_trajectory[i][1],
                dx=rl_trajectory[i+1][0]-rl_trajectory[i][0],
                dy=rl_trajectory[i+1][1]-rl_trajectory[i][1],
                color='r',linestyle='--',width=0.01)
            
            #plt.plot(x_arr,f_vals,linestyle='--',label='$f(x)=(x^2-1)^2+0.3(x-1)^2$')
            plt.tick_params(size=16)
            plt.xlabel('x',size=20)
            plt.ylabel('f(x)',size=20)
            plt.legend(loc='upper right',fontsize=20)   
            # plt.xlim([-20, 20])
            # plt.ylim([-20, 20])
            plt.plot(0, 0, 'ro')  # global minimum
            plt.grid(True)
            # plt.colorbar(cp, label="f(x, y)")
            plt.savefig(pathInit+"/rl_tr2d_"+str(n)+".pdf", bbox_inches='tight', dpi=300 )
            plt.savefig(pathInit+"/rl_tr12d_"+str(n)+".png", bbox_inches='tight', dpi=300 )
            plt.close()

