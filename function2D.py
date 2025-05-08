import numpy as np


############### -- The objective function. In Figure 3a, f(x) is denoted as \mathcal{L}(x).
def f(x):
    return -(2-np.sum(np.cos(10*x)) + 0.05*np.sum(100*x**2))+10

def compute_reward(x):
    return f(x)

