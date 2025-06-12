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


from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#agent
from tf_agents.trajectories import StepType


#other used packages
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

seed = 42

import function2D as fu

#To limit TensorFlow to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()


import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler


# Create a 'log' directory if it doesn't exist
# log_dir = 'log'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)

# # Get the current timestamp in a specific format (this will only be generated once at the start)
# timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# # Create a log filename with the timestamp inside the 'log' folder (This file will be created only once per session)
# log_filename = os.path.join(log_dir, f'logfile_{timestamp}.log')

# # Configure the logging system
# logging.basicConfig(
#     filename=log_filename,              # Use the timestamped filename inside the 'log' folder
#     level=logging.DEBUG,                 # Set the logging level
#     format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
#     datefmt='%Y-%m-%d %H:%M:%S'          # Date format for log entries
# )

# # Now you can log messages
# logging.info("This is an info log message.")
# logging.debug("This is a debug log message.")
# logging.error("This is an error log message.")


# # # Create a 'log' directory if it doesn't exist
# # log_dir = 'log'
# # if not os.path.exists(log_dir):
# #     os.makedirs(log_dir)

# # # Get the current timestamp in a specific format
# # timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# # # Create a log filename with the timestamp inside the 'log' folder
# # log_filename = os.path.join(log_dir, f'logfile_{timestamp}.log')

# # # Configure the logging system
# # logging.basicConfig(
# #     filename=log_filename,              # Use the timestamped filename inside the 'log' folder
# #     level=logging.DEBUG,                 # Set the logging level
# #     format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
# #     datefmt='%Y-%m-%d %H:%M:%S'          # Date format for log entries
# # )


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

class lr_schedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, C):
        self.initial_learning_rate = initial_lr
        self.C = C
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        return self.initial_learning_rate * self.C / (self.C + step)
     

#Functions needed for training
def extract_episode(traj_batch,epi_length,state_dim=2 ,attr_name = 'observation'):
    """
    This function extract episodes (each episode consists of consecutive time_steps) from a batch of trajectories.
    Inputs
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




class RLOpt():
    def __init__(self, parallel_env ,env_num=1, start=0, end=16000, dirName='new3d'):
        ### create a folder for  results and take an 
        self.logger = logInit()
        self.path = os.path.join("results", dirName)
        self.start = start
        self.end = end
        
        self.parallel_env =parallel_env
        
        self.checkpoint_dir = 'checkpoints/'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, 'ckpt')
        
        try:
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.path+'/trajInt/', exist_ok=True)
           
            print("Directory '%s' created successfully" % dirName)
        except OSError as error:
            print("Directory '%s' can not be created")
            
        N=2
        self.state_dim =N
        #Actor train program
        self.REINFORCE_logs = [] #for logging the best objective value of the best solution among all the solutions used for one update of theta 
        self.eval_intv = 100 #number of updates required before each policy evaluation
        self.logger.info(f"evaluation interval: {self.eval_intv}")
        self.final_reward = -500
        self.logger.info(f"final reward: {self.final_reward}")
        self.plot_intv = 100 # plot at interval 
        self.logger.info(f"plot interval: {self.plot_intv}")
        
        self.train_step_num = 0
        #Set initial x-value
        # r = np.random.RandomState(0)
        self.x0_reinforce = np.array([0.5,-1])
        self.sub_episode_length = 30 #number of time_steps in a sub-episode. 
        self.logger.info(f"sub episode lenght(steps): {self.sub_episode_length}")
        self.episode_length = self.sub_episode_length*6  #an trajectory starts from the initial timestep and has no other initial timesteps
                                            #each trajectory will be split to multiple episodes
        self.logger.info(f"on trajectory length with 6 batches: {self.episode_length }")                                    
        self.env_num = env_num #Number of parallel environments, each environment is used to generate an episode
        self.logger.info(f"env number: {self.env_num}")
        print('x0', self.x0_reinforce)
        
        self.act_min = -1#-1
        self.logger.info(f"action minimum: {self.act_min}")
        self.act_max = 1 #1cl
        self.logger.info(f"action maximum: {self.act_max}")
        self.step_size = 0.05
        self.logger.info(f"step size: {self.step_size}")
        self.step_numGe = 100 # gredient step size
        self.logger.info(f"gradient step size: {self.step_numGe}")

        #Set hyper-parameters for REINFORCE-OPT
        self.generation_num = self.end #number of theta updates for REINFORCE-IP, also serves as the number
                            #of generations for GA, and the number of iterations for particle swarm optimization
        self.logger.info(f"generation number: {self.generation_num}")                     
        self.disc_factor = 1.0
        self.alpha = 0.2 #regularization coefficient
        self.param_alpha = 0.15 #regularization coefficient for actor_network #tested value: 0.02
        self.sub_episode_num = int(self.env_num*(self.episode_length/self.sub_episode_length)) #number of sub-episodes used for a single update of actor_net params
        #(40*(300/50==240)
        self.logger.info(f"number of sub_episodes used for a single param update: {self.sub_episode_num}")
        
        
        
        
        #parallel_env = ParallelPyEnvironment(env_constructors=[lambda: envRL.Env(self.x0_reinforce,act_min=self.act_min,
        #                                                                               act_max=self.act_max, step_size=self.step_size,
        #                                                                               disc_factor=self.disc_factor,sub_episode_length=self.sub_episode_length,
        #                                                                               N=self.state_dim) for _ in range(self.env_num)],
        #                                           start_serially=False,
        #                                           blocking=False,
        #                                           flatten=False)
        
        #parallel_env = ParallelPyEnvironment(env_constructors=[make_env] * self.env_num)
        #Use the wrapper to create a TFEnvironments obj. (so that parallel computation is enabled)
        self.train_env = tf_py_environment.TFPyEnvironment(self.parallel_env, check_dims=True) #instance of parallel environments
        self.eval_env = tf_py_environment.TFPyEnvironment(envRL.Env(self.x0_reinforce,act_min=self.act_min,
                                                                                      act_max=self.act_max, step_size=self.step_size,
                                                                                      disc_factor=self.disc_factor,sub_episode_length=self.sub_episode_length,
                                                                                      N=self.state_dim), check_dims=False) #instance
        # train_env.batch_size: The batch size expected for the actions and observations.  
        self.logger.info(f"train_env.batch_size = parallel environment number =  {self.env_num}")

        # tf.random.set_seed(0)
        self.actor_net = actor_distribution_network.ActorDistributionNetwork(   
                                                self.train_env.observation_spec(),
                                                self.train_env.action_spec(),
                                                fc_layer_params=(16,16,16), #Hidden layers
                                                seed=0, #seed used for Keras kernal initializers for NormalProjectionNetwork.
                                                discrete_projection_net=CategoricalProjectionNetwork,
                                                activation_fn = tf.math.tanh,
                                                #continuous_projection_net=(NormalProjectionNetwork)
                                                   )
        
        
        lr = lr_schedule(initial_lr=0.00002, C=50000)   
         
        self.opt = tf.keras.optimizers.SGD(learning_rate=lr)
        self.logger.info(f"learning rate: {self.opt}")  
        
        #Create the  REINFORCE_agent
        train_step_counter = tf.Variable(0)
        # tf.random.set_seed(0)
        self.REINFORCE_agent = reinforce_agent.ReinforceAgent(
            time_step_spec = self.train_env.time_step_spec(),
            action_spec = self.train_env.action_spec(),
            actor_network = self.actor_net,
            value_network = None,
            value_estimation_loss_coef = 0.2,
            optimizer = self.opt,
            advantage_fn = None,
            use_advantage_loss = False,
            gamma = 1.0, #discount factor for future returns
            
            normalize_returns = False, #The instruction says it's better to normalize
            gradient_clipping = None,
            entropy_regularization = None,
            train_step_counter = train_step_counter
            )

        self.REINFORCE_agent.initialize()
        
         # Checkpoint setup
        
        self.checkpoint = tf.train.Checkpoint(
            step=tf.Variable(0),
            optimizer=self.opt,
            agent=self.REINFORCE_agent,
            actor_network=self.actor_net
        )
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)
        if self.checkpoint_manager.latest_checkpoint:
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            self.logger.info(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
        else:
            self.logger.info(f"Initializing from scratch.")
        
        #################
        #replay_buffer is used to store policy exploration data
        #################
        self.replay_buffer = rb.TFUniformReplayBuffer(
                data_spec = self.REINFORCE_agent.collect_data_spec,  # describe spec for a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                batch_size = self.env_num,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                         # One batch corresponds to one parallel environment
                max_length = self.episode_length*100    # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                    # if exceeding this number previous trajectories will be dropped
                )
        
        #test_buffer is used for evaluating a policy
        self.test_buffer = rb.TFUniformReplayBuffer(
                data_spec= self.REINFORCE_agent.collect_data_spec,  # describe a single iterm in the buffer. A TensorSpec or a list/tuple/nest of TensorSpecs describing a single item that can be stored in this buffer.
                batch_size= 1,    # number of parallel worlds, where in each world there is an agent generating trajectories
                                                    # train_env.batch_size: The batch size expected for the actions and observations.  
                                                    # batch_size: Batch dimension of tensors when adding to buffer. 
                max_length = self.episode_length         # The maximum number of items that can be stored in a single batch segment of the buffer.     
                                                    # if exceeding this number previous trajectories will be dropped
                )
        
            
        #A driver uses an agent to perform its policy in the environment.
        #The trajectory is saved in replay_buffer
        self.collect_driver = DynamicEpisodeDriver(
                                        env = self.train_env, #train_env contains parallel environments (no.: env_num)
                                        policy = self.REINFORCE_agent.collect_policy,
                                        observers = [self.replay_buffer.add_batch],
                                        num_episodes = self.sub_episode_num   #SUM_i (number of episodes to be performed in the ith parallel environment)
                                        )
        
        #For policy evaluation
        self.test_driver = py_driver.PyDriver(
                                     env = self.eval_env, #PyEnvironment or TFEnvironment class
                                     policy = self.REINFORCE_agent.policy,
                                     observers = [self.test_buffer.add_batch],
                                     max_episodes=1, #optional. If provided, the data generation ends whenever
                                                      #either max_steps or max_episodes is reached.
                                     max_steps=self.sub_episode_length
                                )

            #parallel_env.close()
            
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
        
        
    def run(self):
        update_num=self.generation_num
        # tf.random.set_seed(0)
        ### for plotting contour import create 2D function 
        X1, X2, Y=utills.create2Dfunction(x1Lim=(-2, 2), x2Lim=(-2, 2), N=200)
        self.replay_buffer.clear()
        for n in np.arange(self.start, self.end, 1):
            t0 = tf.timestamp()
            print(f"iteration: {n}")
            #Generate Trajectories
            self.replay_buffer.clear()
            self.collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
            
            #a batch of trajectories will be saved in replay_buffer
            
            t1 = tf.timestamp()
            self.logger.info(f"time taken after self.collect_driver ={t1-t0}")
            experience = self.replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
            rewards = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
            observations = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)
            actions = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
            step_types = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'step_type')
            discounts = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'discount')
            # self.logger.info(f"rewards from experience ={rewards}")
            time_steps = ts.TimeStep(step_types,
                             tf.zeros_like(rewards),
                             tf.zeros_like(discounts),
                             observations)
            
            t2= tf.timestamp()
            # print(" inside actions:", actions)
            self.logger.info(f"time taken after 6 extract_episods and time_steps={t2-t0}")
    
            rewards_sum = tf.reduce_sum(rewards, axis=1) #shape=(sub_episode_num,)
            self.logger.info(f"rewards sum ={rewards_sum}, rewards shape {rewards_sum.shape}, shape = {rewards.numpy().shape}")
    
            with tf.GradientTape() as tape:
                #trainable parameters in the actor_network in REINFORCE_agent
                variables_to_train = self.REINFORCE_agent._actor_network.trainable_weights
            
                ###########Compute J_loss = -J
                actions_distribution = self.REINFORCE_agent.collect_policy.distribution(
                                    time_steps, policy_state=None).action
                # print("actions_distribution: ", actions_distribution, actions_distribution.sample(), 
                #       "self.REINFORCE_agent.action_spec", self.REINFORCE_agent.action_spec)
                #log(pi(action|state)), shape = (batch_size, epsode_length)
                action_log_prob = common.log_probability(actions_distribution, 
                                                        actions,
                                                        self.REINFORCE_agent.action_spec)
                # print("action_log_prob :", action_log_prob )
                J = tf.reduce_sum(tf.reduce_sum(action_log_prob,axis=1)*rewards_sum)/self.sub_episode_num
                
                ###########Compute regularization loss from actor_net params
                regu_term = tf.reduce_sum(variables_to_train[0]**2)
                num = len(variables_to_train) #number of vectors in variables_to_train
                for i in range(1,num):
                    regu_term += tf.reduce_sum(variables_to_train[i]**2)
                
                total = -J + self.param_alpha*regu_term
              
            #update parameters in the actor_network in the policy
            grads = tape.gradient(total, variables_to_train)
            grads_and_vars = list(zip(grads, variables_to_train))
            self.opt.apply_gradients(grads_and_vars=grads_and_vars)
            self.train_step_num += 1
            
            batch_rewards = rewards.numpy()
            
            batch_rewards[:,-1] = -np.power(10,8) #The initial reward is set as 0, we set it as this value to not affect the best_obs_index 
            best_step_reward = np.max(batch_rewards)
            best_step_index = [int(batch_rewards.argmax()/self.sub_episode_length),batch_rewards.argmax()%self.sub_episode_length+1]
            best_step = observations[best_step_index[0],best_step_index[1],:] #best solution
            #best_step_reward = f(best_solution)
            avg_step_reward = np.mean(batch_rewards[:,0:-1])
            self.REINFORCE_logs.append(best_step_reward)
            print("best reward",np.mean(batch_rewards), np.max(batch_rewards), np.min(batch_rewards) )
            self.logger.info(f"iter :: {n} :: avg reward {avg_step_reward} ::batch rewards shape :: {batch_rewards.shape},  mean rewards: {np.mean(batch_rewards)},max rewards: {np.max(batch_rewards)}, min rewards: {np.min(batch_rewards)}")
            self.logger.info(f"iter :: {n}:: Total loss ={total}")  
            if best_step_reward>self.final_reward:
                #print("final reward before udpate:",final_reward)
                self.final_reward = best_step_reward
                final_solution = best_step.numpy()
                #print("final reward after udpate:",final_reward)
                #self.logger.info(f"updated final_solution=', final_solution)
                    
            if n%self.eval_intv==0:
                # self.replay_buffer.clear()
                self.logger.info(f"train_step no.={self.train_step_num}")
                self.logger.info(f"best_solution of this generation= {best_step.numpy()}")
                self.logger.info(f"best step reward= {best_step_reward.round(3),fun.f(best_step.numpy())}")
                self.logger.info(f"avg step reward= {round(avg_step_reward,3)}")
                #self.logger.info(f"episode of rewards', rewards.round(3))
                #self.logger.info(f"act_std:', actions_distribution.stddev()[0,0]  )
                #self.logger.info(f"act_mean:', actions_distribution.mean()[0,0] ) #second action mean
                self.logger.info(f"best_step_index: {best_step_index}")
                self.logger.info(f"observation: {observations[0]}")
                # self.logger.info(f" ')
                # Save checkpoint
                self.checkpoint.step.assign(n)
                self.checkpoint_manager.save()
                self.logger.info(f"Checkpoint saved at step :{int(self.checkpoint.step)}")

            
            if n%self.plot_intv==0:
                self.test_buffer.clear()
                self.test_driver.run(self.eval_env.reset())  #generate batches of trajectories with agent.collect_policy, and save them in replay_buffer
                experience = self.test_buffer.gather_all()  #get all the stored items in replay_buffer
                rl_trajectory = experience.observation.numpy()[0]
                
                #Plot
                fig, ax = plt.subplots(figsize=(8,6), constrained_layout=True)
                # CS = ax.contour(X1, X2, Y)
                cp = plt.contour(X1, X2, Y, levels=np.logspace(-1, 3, 20), cmap='jet')
                # cp = plt.contour(X1, X2, Y,levels= -np.logspace(-1, 3, 20)[::-1], cmap=plt.get_cmap('jet_r'))
                plt.clabel(cp, inline=True, fontsize=8)
                
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
                plt.colorbar(cp, label="f(x, y)")
                plt.savefig(self.path+"/trajInt/rl_tr2d_"+str(n)+".pdf", bbox_inches='tight', dpi=300 )
                plt.savefig(self.path+"/trajInt/rl_tr12d_"+str(n)+".png", bbox_inches='tight', dpi=300 )
                plt.close()
            
            t3= tf.timestamp()
            self.logger.info(f"time taken after each loop={t3-t0}")
            
        self.logger.info(f"final_solution= {final_solution}, final_reward= {self.final_reward}")
        self.REINFORCE_logs = [max(self.REINFORCE_logs[0:i]) for i in range(1, self.generation_num+1)]
        
        ############################################# - Second Part of Figure 3
        ############# - Create a trajectory with REINFORCE_agent.policy (which select action according to the mode of action distribution)
        # self.test_policy()
        # self.test_buffer.clear()
        # self.test_driver.run(self.eval_env.reset())  #generate batches of trajectories with agent.collect_policy, and save them in replay_buffer
        # experience = self.test_buffer.gather_all()  #get all the stored items in replay_buffer
        # rl_trajectory = experience.observation.numpy()[0]
        
        # ga_trajectory=self.gradientOpt()
        
        # utills.plotTrajectory(rl_trajectory, ga_trajectory, self.step_numGe, self.path)

def main(argv):
    env_num=10
    parallel_env = ParallelPyEnvironment(env_constructors=[envRL.Env] * env_num)
    RL=RLOpt(parallel_env, env_num=env_num, start=7400, end=16000) 
    RL.run()
    
if __name__ =='__main__':
    tf_agents.system.multiprocessing.handle_main(main, extra_state_savers=[])

