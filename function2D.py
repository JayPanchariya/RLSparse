import numpy as np
import tensorflow as tf

############### -- The objective function. In Figure 3a, f(x) is denoted as \mathcal{L}(x).
# def norm (y):
#     mue = np.mean(y)
#     sigma = np.std(y)
#     return (y-mue)/sigma**2

# def f(x):
#     return -(2-np.sum(np.cos(10*x)) + 0.05*np.sum(100*x**2))+10

def f(x):
    x1 = x[0]
    x2 = x[1]
    return -((1-x1)**2 + 100 * (x2 - x1**2)**2)

def compute_reward(x):
    return f(x)

if __name__ == "__main__":
    a=tf.Variable(np.array([-1,1]))
    print(f(a))

