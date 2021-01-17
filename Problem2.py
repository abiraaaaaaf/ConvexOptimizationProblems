import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

Num = 100

# data generation
np.random.seed(0)
(m, n) = (300, 100)
A = np.random.rand(m, n);
A = np.asmatrix(A);  #nonnegative
b = A.dot(np.ones((n, 1)))/2;
b = np.asmatrix(b);
c = -np.random.rand(n, 1);
c = np.asmatrix(c);

# relaxed LP :D
x = cp.Variable(shape=(n, 1))
constraints = [x >= 0, x <= 1, A*x <= b]
objt = cp.Minimize(c.T * x)
prob = cp.Problem(objt, constraints)
res = prob.solve()
L = objt.value
print("The lower bound found from the relaxed LP:",  L)
xrlx = x.value

t = np.linspace(0, 1, num=Num).reshape(Num, 1)
max_violation = np.zeros((Num, 1))
obj = np.zeros((Num, 1))
U = float('inf')
t_best = float('nan')

for i in range(Num):
    x = np.matrix(xrlx >= t[i], dtype=float)
    obj[i] = c.T * x
    max_violation[i] = max(A * x-b)
    if max_violation[i] <= 0 and obj[i] < U:
        U = float(obj[i])
        x_best = x
        t_best = t[i]

print("t_best", t_best)
print("U is:", U)

diff = np.abs(U-L)
print("Gap is", diff)

#Plot max_violation
plt.figure(1)
plt.subplot(211)
plt.plot(t[max_violation <= 0], max_violation[max_violation <= 0], 'c')
plt.plot(t[max_violation > 0], max_violation[max_violation > 0], 'm')
plt.ylabel('max_violation')
plt.xlabel('threshold')


#Plot objective_function
plt.subplot(212)
plt.plot(t[max_violation <= 0], obj[max_violation <= 0], 'c')
plt.plot(t[max_violation > 0], obj[max_violation > 0], 'm')
plt.plot(t, objt.value * np.ones((Num, 1)), 'r')
plt.ylabel('objective_function')
plt.xlabel('threshold')
plt.savefig('q9_lotfi.png')
plt.show()