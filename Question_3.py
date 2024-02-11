import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**3 + x**2 + (21/5)*x + 5

# Bisection Method
def my_bisection(f, a, b, maxiter):
    root_approximations = []
    
    for iteration in range(1, maxiter+1):
        c = (a + b) / 2
        root_approximations.append(c)
        
        if f(c) * f(b) < 0:
            a = c
        else:
            b = c
    
    return root_approximations

# Secant Method
def my_secant(f, a, b, maxiter):
    root_approximations = [a, b]
    
    for iteration in range(2, maxiter):
        
        if abs(f(b) - f(a)) < 1e-10:
           print("ZeroDivisionError: float division by zero")
           break
       
        c = b - f(b) * (b - a) / (f(b) - f(a))
        root_approximations.append(c)
        
        a, b = b, c
    #Control to plot
    while len(root_approximations) < maxiter:
        root_approximations.append(None)
    
    return root_approximations

a = -3
b = 0
maxiter = 20

bisection_results = my_bisection(f, a, b, maxiter)
secant_results = my_secant(f, a, b, maxiter)

# Plotting
iterations = np.arange(1, maxiter+1)

plt.plot(iterations, bisection_results[:maxiter], label='Bisection Method')
plt.plot(iterations, secant_results[:maxiter], label='Secant Method')
plt.xlabel('Number of Iterations')
plt.ylabel('Root Approximations')
plt.title('Root Approximations using Bisection and Secant Methods')
plt.legend()
plt.show()
