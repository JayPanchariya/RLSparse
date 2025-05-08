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

#To limit TensorFlow to CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
cpus = tf.config.experimental.list_physical_devices('CPU') 
tf.config.experimental.set_visible_devices(cpus[0], 'CPU')
#enable multiprocessing for parallel computing
tf_agents.system.multiprocessing.enable_interactive_mode()
gc.collect()


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
    def __init__(self):
        N=2
        self.state_dim =N
        #Actor train program
        self.REINFORCE_logs = [] #for logging the best objective value of the best solution among all the solutions used for one update of theta 
        self.eval_intv = 2 #number of updates required before each policy evaluation
        self.final_reward = -1000
        self.plot_intv = 500 # plot at interval 
        self.train_step_num = 0
        #Set initial x-value
        r = np.random.RandomState(0)
        self.x0_reinforce = np.array([0.5,-1.0])
        self.sub_episode_length = 30 #number of time_steps in a sub-episode. 
        self.episode_length = self.sub_episode_length*6  #an trajectory starts from the initial timestep and has no other initial timesteps
                                            #each trajectory will be split to multiple episodes
        self.env_num = 7 #Number of parallel environments, each environment is used to generate an episode
        print('x0', self.x0_reinforce)
        
        self.act_min = -1
        self.act_max = 1
        self.step_size = 0.05
        self.step_numGe = 30 # gredient step size

        #Set hyper-parameters for REINFORCE-OPT
        self.generation_num = 10 #number of theta updates for REINFORCE-IP, also serves as the number
                            #of generations for GA, and the number of iterations for particle swarm optimization

        self.disc_factor = 1.0
        self.alpha = 0.2 #regularization coefficient
        self.param_alpha = 0.15 #regularization coefficient for actor_network #tested value: 0.02
        self.sub_episode_num = int(self.env_num*(self.episode_length/self.sub_episode_length)) #number of sub-episodes used for a single update of actor_net params
        print("number of sub_episodes used for a single param update:", self.sub_episode_num)
        
        self.parallel_env = ParallelPyEnvironment(env_constructors=[lambda: envRL.Env(self.x0_reinforce,act_min=self.act_min,
                                                                                      act_max=self.act_max, step_size=self.step_size,
                                                                                      disc_factor=self.disc_factor,sub_episode_length=self.sub_episode_length,
                                                                                      N=self.state_dim) for _ in range(self.env_num)],
                                                  start_serially=False,
                                                  blocking=False,
                                                  flatten=False)
        
        #Use the wrapper to create a TFEnvironments obj. (so that parallel computation is enabled)
        self.train_env = tf_py_environment.TFPyEnvironment(self.parallel_env, check_dims=True) #instance of parallel environments
        self.eval_env = tf_py_environment.TFPyEnvironment(envRL.Env(self.x0_reinforce,act_min=self.act_min,
                                                                                      act_max=self.act_max, step_size=self.step_size,
                                                                                      disc_factor=self.disc_factor,sub_episode_length=self.sub_episode_length,
                                                                                      N=self.state_dim), check_dims=False) #instance
        # train_env.batch_size: The batch size expected for the actions and observations.  
        print('train_env.batch_size = parallel environment number = ', self.env_num)

        tf.random.set_seed(0)
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
        
        #Create the  REINFORCE_agent
        train_step_counter = tf.Variable(0)
        tf.random.set_seed(0)
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
        test_driver = py_driver.PyDriver(
                                     env = self.eval_env, #PyEnvironment or TFEnvironment class
                                     policy = self.REINFORCE_agent.policy,
                                     observers = [self.test_buffer.add_batch],
                                     max_episodes=1, #optional. If provided, the data generation ends whenever
                                                      #either max_steps or max_episodes is reached.
                                     max_steps=self.sub_episode_length
                                )

            # self.parallel_env.close()
            
    
    ############# - Create the Gradient Ascent Trajectory
    def gradientOpt(self):
        ga_trajectory = [self.x0_reinforce]
        current_x = tf.Variable(self.x0_reinforce)
        # current_x = tf.Variable([0.5,-1])
        
        for i in range(self.step_numGe):
            with tf.GradientTape() as tape:
                y = -(2-tf.reduce_sum(tf.math.cos(10*current_x)) + 0.05*tf.reduce_sum(100*current_x**2))+10
            gradient = tape.gradient(y, current_x)
            norm_gradient = gradient.numpy()/np.sqrt(np.sum(gradient.numpy()**2))
            current_x.assign(current_x.numpy() + norm_gradient*self.step_size)
            ga_trajectory.append(current_x.numpy())
        
        
    def run(self):
        update_num=self.generation_num
        tf.random.set_seed(0)
        for n in range(0,update_num):
            print(n)
            #Generate Trajectories
            self.replay_buffer.clear()
            self.collect_driver.run()  #a batch of trajectories will be saved in replay_buffer
            
            experience = self.replay_buffer.gather_all() #get the batch of trajectories, shape=(batch_size, episode_length)
            rewards = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'reward') #shape=(sub_episode_num, sub_episode_length)
            observations = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'observation') #shape=(sub_episode_num, sub_episode_length, state_dim)
            actions = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'action') #shape=(sub_episode_num, sub_episode_length, state_dim)
            step_types = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'step_type')
            discounts = extract_episode(traj_batch=experience,epi_length=self.sub_episode_length,attr_name = 'discount')
            
            time_steps = ts.TimeStep(step_types,
                             tf.zeros_like(rewards),
                             tf.zeros_like(discounts),
                             observations)
    
            rewards_sum = tf.reduce_sum(rewards, axis=1) #shape=(sub_episode_num,)
    
            with tf.GradientTape() as tape:
                #trainable parameters in the actor_network in REINFORCE_agent
                variables_to_train = self.REINFORCE_agent._actor_network.trainable_weights
            
                ###########Compute J_loss = -J
                actions_distribution = self.REINFORCE_agent.collect_policy.distribution(
                                    time_steps, policy_state=None).action
            
                #log(pi(action|state)), shape = (batch_size, epsode_length)
                action_log_prob = common.log_probability(actions_distribution, 
                                                        actions,
                                                        self.REINFORCE_agent.action_spec)
            
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
            
            if n%self.eval_intv==0:
                print("train_step no.=",self.train_step_num)
                print('best_solution of this generation=', best_step.numpy())
                print('best step reward=',best_step_reward.round(3),fun.f(best_step.numpy()))
                print('avg step reward=', round(avg_step_reward,3))
                #print('episode of rewards', rewards.round(3))
                #print('act_std:', actions_distribution.stddev()[0,0]  )
                #print('act_mean:', actions_distribution.mean()[0,0] ) #second action mean
                print('best_step_index:',best_step_index)
                print(observations[0])
                print(' ')
            
            

if __name__ =='__main__':
    RL=RLOpt()  
    RL.run()

    