import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#******************************** part a ********************************
# Variables
x = cp.Variable(shape=(3, 1))
y = cp.Variable(shape=(3, 1))
A = cp.Variable((3, 3), PSD = True)
x3 = cp.Variable()
x2 = cp.Variable()
x1 = cp.Variable()
c3 = cp.Variable()

# Parameters
Q = cp.Parameter(shape=(3, 3), PSD=True)
b = cp.Parameter(shape=(3, ))
c = cp.Parameter()

# Values
Q.value = np.array([[4/9, 0, 0], [0, 4, 0], [0, 0, 1/4]])
b.value = np.transpose(np.array([16/9, 16, 1]))
c.value = 169/9

v1 = [1, 1, 0]
v2 = [1, 0, 1]
v3 = [0, 1, 1]

x1 = (cp.quad_form(v1, A) - 2) * 0.5
x2 = (cp.quad_form(v2, A) - 2) * 0.5
c3 = cp.quad_form(v3, A) - 2

# Constraints
constraints = [cp.quad_form(y, Q) + y.T * b + c <= 1]  # S2
constraints += [x3 >= 0, x3 <= 1, c3 == 1, cp.diag(A) == [1, 1, 1]] # S1

# Construct X
x = cp.vstack([x1, x2, x3])

obj = cp.Minimize(cp.norm(x-y))
prob = cp.Problem(obj, constraints)
res = prob.solve()

print("status:", prob.status)
print("optimal value", prob.value)
print("optimal var x", x.value)
print("optimal var y", y.value)
print("optimum distance", np.sqrt(np.sum(np.square(x.value - y.value))))
dist = x.value - y.value

#*********************** part b ****************************

# Plot S2
fig = plt.figure(figsize=plt.figaspect(1))  # Square figure
ax = fig.add_subplot(111, projection='3d')
coefs = (4/9, 4, 1/4)
rx, ry, rz = 1/np.sqrt(coefs)
# Set of all spherical angles:
u = np.linspace(0, 2 * np.pi, 101)
v = np.linspace(0, np.pi, 101)
# Cartesian coordinates that correspond to the spherical angles:
# (this is the equation of an ellipsoid):
x_axis = rx * (np.outer(np.cos(u), np.sin(v))) - 2.0
y_axis = ry * (np.outer(np.sin(u), np.sin(v)))- 2.0
z_axis = rz * (np.outer(np.ones_like(u), np.cos(v)))- 2.0
ax.plot_surface(x_axis, y_axis, z_axis,  rstride=4, cstride=4, color='r')

# Plot Line Distance
ax1 = plt.gca(projection='3d')
x_1, y_1, z_1 = [x1.value, y[0].value], [x2.value, y[1].value], [x3.value, y[2].value]
ax1.scatter(x_1, y_1, z_1, c='r', s=100)
ax1.plot(x_1, y_1, z_1, color='r', label=r'elliptical')


# Plot S1
a = 1/np.sqrt(1/2)
b = 1/np.sqrt(1/2)
us = np.linspace(0, np.pi * 2, 101)

zs = np.linspace(0, 1, 101)

xs = a * np.cos(us)
xs, zs = np.meshgrid(xs, zs)
ys = b * np.sin(us)

ax.plot_surface(xs, ys, zs,  rstride=4, cstride=4, color='r', label=r'cylinder elliptical')
# Adjustment of the axes, so that they all have the same span:
max_radius = max(rx, ry, rz, a, b)
for axis in 'xyz':
    getattr(ax, 'set_{}lim'.format(axis))((-max_radius-2, max_radius))

plt.show()

