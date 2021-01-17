import numpy as np
import cvxpy
from scipy.io import loadmat
import matplotlib.pyplot as plt

ALPHA = 0.1
BETA = 0.8
MAXITERS = 100
EPSILON = 1e-3
case = 0  # 0 for euclinian  1 for quadratic
x_0 = np.zeros((10, 1))
# x_hist = np.zeros((100, 1))

def ditance(a, b):
    return np.sqrt(np.sum(np.square(a, b)))

def plot_total(a, b, c, k):
    x = np.zeros((k, 1))
    cnt = 1
    for i in range(k):
        x[i] = cnt
        cnt += 1
    y = [a, b, c]
    plt.scatter(x, y)
    plt.show()

def plot(y):
    x = np.zeros((len(y), 1))
    cnt = 1
    for i in range(len(y)):
        x[i] = cnt
        cnt += 1
    plt.scatter(x, y)
    plt.show()

def sd_quad(A, b, c, p_star, case=0):
    x_hist = []
    if case == 0:
        print('Steepest Descent Euclinian:')
    elif case == 1:
        print('Steepest Descent Quadratic:')
    iteration = 0
    A = np.matrix(A)
    b = b.reshape(len(b), 1)
    c = c.reshape(len(c), 1)
    x = x_0

    for iterNum in range(MAXITERS):
        iteration += 1
        val = 1 / 2 * x.T * A * x + b.T * x + c
        grad = (A * x) + b

        #    1. Compute steepest descent direction as delta_x

        delta_x_e = - grad  # euclinian
        delta_x_q = - np.linalg.inv(A) * grad  # quadratic
        if case == 0:
            delta_x = delta_x_e
        elif case == 1:
            delta_x = delta_x_q
        else:
            print('Error!')

        #   2. Backtracking Choose t

        t = 1
        while ((1 / 2 * (x + t * delta_x).T * A * (x + t * delta_x) + b.T * (x + t * delta_x) + c) > (
                    ((1 / 2 * x.T * A * x) + np.dot(b.T, x) + c) + ALPHA * t * np.dot(grad.T, delta_x))):
            t *= BETA

        #   3. Update x

        x = x + t * delta_x
        dist = np.abs(p_star - ((1 / 2 * x.T * A * x) + np.dot(b.T, x) + c))
        x_hist.append(dist)
        condition = (dist <= EPSILON)
        if condition == True:
            break
    if iteration == 100:
        print("with max iterations did not converge!")
    else:
        print("with %d iterations converged!" %iteration)
    return x, x_hist

mat = loadmat('data.mat')
A = mat['A']
b = mat['b']
c = mat['c']

x_star_1 = - (np.matmul(np.linalg.inv(A), b))
p_star_1 = (1 / 2 * np.matmul(np.matmul(x_star_1.T, A), x_star_1)) + np.dot(b.T, x_star_1) + c

x_1, x_hist_1 = sd_quad(A, b, c, p_star_1, 0)
p_1 = (1 / 2 * np.matmul(np.matmul(x_1.T, A), x_1)) + np.dot(b.T, x_1) + c

x_3, x_hist_3 = sd_quad(A, b, c, p_star_1, 1)
p_3 = (1 / 2 * np.matmul(np.matmul(x_3.T, A), x_3)) + np.dot(b.T, x_3) + c

A = 5 * np.diag(np.diag(np.ones((10, 10))))
x_star_2 = - (np.matmul(np.linalg.inv(A), b))
p_star_2 = (1 / 2 * np.matmul(np.matmul(x_star_2.T, A), x_star_2)) + np.dot(b.T, x_star_2) + c

x_2, x_hist_2 = sd_quad(A, b, c, p_star_2, 1)
p_2 = (1 / 2 * np.matmul(np.matmul(x_2.T , A), x_2)) + np.dot(b.T, x_2) + c

x_4, x_hist_4 = sd_quad(A, b, c, p_star_2, 0)
p_4 = (1 / 2 * np.matmul(np.matmul(x_4.T, A), x_4)) + np.dot(b.T, x_4) + c

print(np.sqrt(np.sum(np.square(x_1, x_star_1))))
print(np.sqrt(np.sum(np.square(x_2, x_star_2))))

print(p_1)
print(p_3)
print('P* with mat is : %f'%p_star_1)
print(p_2)
print(p_4)
print('P* with 5I is : %f'%p_star_2)

# plot_total(p_1, p_3, p_star_1, 3)
# plot_total(p_2, p_4, p_star_2, 3)

plot(x_hist_1)
plot(x_hist_3)
plot(x_hist_2)
plot(x_hist_4)

