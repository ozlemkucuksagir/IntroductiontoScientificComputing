import timeit
import time
import numpy as np
import sympy
from scipy import linalg as la 


def solve_linear_system(A, b, method):
  
    if method == "symbolic_lu":
        tic=time.process_time()
        L, U, _= A.LUdecomposition()
        x = A.LUsolve(b)
        toc=time.process_time()
        print("Symbolic LU Elapsed Time:"+str(1000*(toc-tic))+" ms")
 
    elif method == "symbolic_inverse":
        tic=time.process_time()
        x = A.inv() @ b
        toc=time.process_time()
        print("Symbolic Inverse Elapsed Time::"+str(1000*(toc-tic))+" ms")
        
    elif method == "numeric_lu":
        #Measure very short time
        tic=timeit.default_timer()
        P, L, U = la.lu(A)
        x=la.solve(A,b)
        toc=timeit.default_timer()
        print("Numeric LU Elapsed Time:"+str(1000*(toc-tic))+" ms")
                
    elif method == "numeric_inverse":
        #Measure very short time
        tic=timeit.default_timer()
        x =la.inv(A) @ b
        toc=timeit.default_timer()
        print("Numeric Inverse Elapsed Time::"+str(1000*(toc-tic))+" ms")
        
    else:
        print("Invalid method")

    return x


A = np.random.rand(50, 50)
b = np.random.rand(50, 1)

A_sympy = sympy.Matrix(A)
b_sympy = sympy.Matrix(b)

# Symbolic LU
solve_linear_system(A_sympy, b_sympy, "symbolic_lu")

# Symbolic inverse
result = solve_linear_system(A_sympy, b_sympy, "symbolic_inverse")

# Numeric LU
result = solve_linear_system(A, b, "numeric_lu")

# Numeric inverse
result = solve_linear_system(A, b, "numeric_inverse")
