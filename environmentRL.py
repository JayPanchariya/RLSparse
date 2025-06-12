import numpy as np
import tensorflow as tf
#import environment
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment #allows parallel computing for generating experiences
from tf_agents.environments.parallel_py_environment import ParallelPyEnvironment

#agent
from tf_agents.trajectories import StepType


#other used packages
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

seed = 42

import function2D as fun

# r = np.random.RandomState(0)
# tf.random.set_seed(42) 


#Define the Environment(from original code Line 135-231) 
class Env(py_environment.PyEnvironment):
    def __init__(self, x0_reinforce=np.array([0.5,-1.0]), act_min = -1, act_max = 1, step_size=0.05, disc_factor = 1.0, sub_episode_length = 30, N=2):
        '''The function to initialize an Env obj.
        '''
        print("dddd")
        self.state_dim=N ### number of dim
        self.x0_reinforce = x0_reinforce ## it for intialization 
        self.step_size = step_size
        self.disc_factor = disc_factor 
        self.sub_episode_length = sub_episode_length
        #Specify the requirement for the value of action, (It is a 2d-array for this case)
        #which is an argument of _step(self, action) that is later defined.
        #tf_agents.specs.BoundedArraySpec is a class.
        #_action_spec.check_array( arr ) returns true if arr conforms to the specification stored in _action_spec
        self.act_min =  act_min ## action confined
        self.act_max = act_max
        
        
        
        self._action_spec = array_spec.BoundedArraySpec(
                            shape=(self.state_dim,), dtype=np.int32, minimum=0, maximum=self.act_max-self.act_min, name='action') #a_t is an 2darray

        # print(self._action_spec)
        #Specify the format requirement for observation (It is a 2d-array for this case), 
        #i.e. the observable part of S_t, and it is stored in self._state
        self._observation_spec = array_spec.BoundedArraySpec(
                                 shape=(self.state_dim,), dtype=np.float32, name='observation') #default max and min is None
        self._state = np.array(self.x0_reinforce,dtype=np.float32)
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
        self._state = np.array(self.x0_reinforce,dtype=np.float32)  #initial state
        self._episode_ended = False
        self._step_counter = 0
        
        #Reward
        initial_r = np.float32(0.0)
        
        #return ts.restart(observation=np.array(self._state, dtype=np.float32))
        return ts.TimeStep(step_type=StepType.FIRST, 
                           reward=initial_r, 
                           discount=np.float32(self.disc_factor), 
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
        # print("action:",  action)
        normalized_act = (action-self.act_max)/np.sqrt((action-self.act_max)**2+0.0000001) 
        # print(" normalized_act:",  normalized_act)
        self._state = self._state + normalized_act*self.step_size    
        self._step_counter +=1
        
        ################# --- Compute R_{t+1}=R(S_t,A_t)
        R = fun.compute_reward(self._state)
        
        #Set conditions for termination
        if self._step_counter>=self.sub_episode_length-1:
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
            return ts.transition(np.array(self._state, dtype=np.float32), reward=R, discount= self.disc_factor)
        

if __name__ == '__main__':
    x0_reinforce = np.array([0.5,-1.0])
    EnvTest=Env(x0_reinforce)