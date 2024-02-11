import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import linalg as la

absent_hours = np.array([1, 6, 8, 12])
grades = np.array([90, 80, 75 - (21/5), 40]) 

#Interpolation
linear_interp = interp1d(absent_hours, grades, kind='linear')
quadratic_interp = interp1d(absent_hours, grades, kind='quadratic')
cubic_interp = interp1d(absent_hours, grades, kind='cubic')


hours_10 = 10
grade_linear = linear_interp(hours_10)
grade_quadratic = quadratic_interp(hours_10)
grade_cubic = cubic_interp(hours_10)

# LU factorization 
A = np.vstack([np.ones_like(absent_hours), absent_hours, absent_hours**2, absent_hours**3]).T
b = grades
P, L, U = la.lu(A)
print("L:",L)
print("U:",U)

# Ly = P^T b
y = la.solve(L, P @ b)

# Ux = y
x = la.solve(U, y)

# 10 saatteki tahmini performans
grade_lu = x[0] + x[1]*hours_10 + x[2]*(hours_10**2) + x[3]*(hours_10**3)

# 100 noktalı grafik için absent hours değerleri
plot_hours = np.linspace(1, 12, 100)

# Performance predictions using interpolation functions
plot_grade_linear = linear_interp(plot_hours)
plot_grade_quadratic = quadratic_interp(plot_hours)
plot_grade_cubic = cubic_interp(plot_hours)

# Plot
plt.plot(plot_hours, plot_grade_linear, label='Linear Interpolation')
plt.plot(plot_hours, plot_grade_quadratic, label='Quadratic Interpolation')
plt.plot(plot_hours, plot_grade_cubic, label='Cubic Interpolation')

# Pointers showing predicted performance
plt.scatter([10], [grade_linear], color='red', marker='o', label='Linear at 10h')
plt.scatter([10], [grade_quadratic], color='blue', marker='o', label='Quadratic at 10h')
plt.scatter([10], [grade_cubic], color='green', marker='o', label='Cubic at 10h')
plt.scatter([10], [grade_lu], color='purple', marker='o', label='LU at 10h')


plt.xlabel('Absent Hours')
plt.ylabel('Grade')
plt.title('Performance vs. Absent Hours')
plt.legend()
plt.show()
