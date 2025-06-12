import numpy as np
import tensorflow as tf

############### -- The objective function. In Figure 3a, f(x) is denoted as \mathcal{L}(x).
# def norm (y):
#     mue = np.mean(y)
#     sigma = np.std(y)
#     return (y-mue)/sigma**2

def f(x):
    return -(2-np.sum(np.cos(10*x)) + 0.05*np.sum(100*x**2))+10

# def f(x):
#     x1 = x[0]
#     x2 = x[1]
#     return -((1-x1)**2 + 100 * (x2 - x1**2)**2)

#f = sinc(X.^2).*sinc((Y-1));
# def f(x):
#     x1 = x[0]
#     x2 = x[1]
#     return -(np.sinc((x1)**2)+ np.sinc(x2-1))- 0.1* (x1**2 + (x2 - 1)**2)

# def f(x):
#     x1 = x[0]
#     x2 = x[1]
#     return -(abs(x1-1) + 2*abs(x2-2))
# def f(x):
#     """
#     Lagrangian function:
#     L(x1, x2, λ) = x1^2 + x2^2 - x1*x2 - λ*(x1^2 + x2^2 - 1)
    
#     Parameters:
#         x1 (float): variable x1
#         x2 (float): variable x2
#         lam (float): Lagrange multiplier λ
        
#     Returns:
#         float: value of the Lagrangian
#     """
#     lam=1
#     x1 = x[0]
#     x2 = x[1]
#     return -(x1**2 + x2**2 - x1 * x2 - lam * (x1**2 + x2**2 - 1))

def compute_reward(x):
    return f(x)

if __name__ == "__main__":
    a=tf.Variable(np.array([-1,1]))
    print(f(a))

