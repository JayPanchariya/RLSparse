import numpy as np
import matplotlib.pyplot as plt

# Define lambda_coef
lambda_coef = 0.5

# Define the function
def f(x):
    return (x - 2)**2 + lambda_coef * (x**2)

# Create x values
x = np.linspace(-5, 5, 400)
y = f(x)

# Plotting
plt.figure(figsize=(8,5))
plt.plot(x, y, label=r'$f(x) = (x-2)^2 + \lambda x^2$', color='blue')
plt.title('Objective Function Plot')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.savefig("objFn.png")
plt.show()
