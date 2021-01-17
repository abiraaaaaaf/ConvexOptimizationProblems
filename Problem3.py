

import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import pylab as pl
# from sklearn import SVC


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def rbf_kernel_10(x, y, lamda = 10):
    return np.exp(-lamda *(linalg.norm(x-y)**2))

def rbf_kernel_50(x, y, lamda = 50):
    return np.exp(-lamda *(linalg.norm(x-y)**2))

def rbf_kernel_100(x, y, lamda = 100):
    return np.exp(-lamda *(linalg.norm(x-y)**2))

def rbf_kernel_500(x, y, lamda = 500):
    return np.exp(-lamda *(linalg.norm(x-y)**2))

class SVM(object):

    def __init__(self, kernel = linear_kernel, C = None):
        self.kernel = kernel
        self.C = C
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        print(n_samples)
        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            tmp1 = np.diag(np.ones(n_samples) * -1)
            tmp2 = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(n_samples)
            tmp2 = np.ones(n_samples) * self.C
            h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # print(solution)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # print(a)



        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
        self.b /= len(self.a)

        # Weight vector
        if self.kernel == linear_kernel:
            self.w = np.zeros(n_features)
            for n in range(len(self.a)):
                self.w += self.a[n] * self.sv_y[n] * self.sv[n]
        else:
            self.w = None

        print(self.b)
        print(self.w)

    def fit_primal(self, phi, y):
        ## primal quad prog.
        ## let n be the number of data points, and let m be the number of features
        n, m = phi.shape
        print(phi.shape)
        P = np.zeros((m + n + 1, m + n + 1))
        for i in range(m):
            P[i, i] = 1
        q = np.vstack([np.zeros((m + 1, 1)), self.C * np.ones((n, 1))])
        G = np.zeros((2 * n, m + 1 + n))
        y = np.reshape(y, (160, 1))
        assert y.shape == (160, 1)
        G[:n, 0:m] = y * phi
        G[:n, m] = y.T
        G[:n, m + 1:] = np.eye(n)
        G[n:, m + 1:] = np.eye(n)
        G = -G
        h = np.zeros((2 * n, 1))
        h[:n] = -1
        ## convert to array
        ## have to convert everything to cxvopt matrices
        P = cvxopt.matrix(P, P.shape, 'd')
        q = cvxopt.matrix(q, q.shape, 'd')
        G = cvxopt.matrix(G, G.shape, 'd')
        h = cvxopt.matrix(h, h.shape, 'd')
        ## set up cvxopt
        ## z (the vector being minimized for) in this case is [w, b, eps].T
        sol = cvxopt.solvers.qp(P, q, G, h)
        print(sol)


        self.w = np.array(sol['x'][:m])
        self.b = sol['x'][m]

        # Support vectors have non zero lagrange multipliers

        # sv = a > 1e-5
        # ind = np.arange(len(a))[sv]
        # self.a = a[sv]
        # self.sv = phi[sv]
        # self.sv_y = y[sv]
        # print("%d support vectors out of %d points" % (len(self.a), n_samples))

        print(self.b)
        print(self.w)


    def project(self, X):
        if self.w is not None:
            return np.dot(X, self.w) + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                    s += a * sv_y * self.kernel(X[i], sv)
                y_predict[i] = s
            return y_predict + self.b

    def predict(self, X):
        return np.sign(self.project(X))

if __name__ == "__main__":

    def gen_lin_separable_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[0.8, 0.6], [0.6, 0.8]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2

    def gen_non_lin_separable_data():
        mean1 = [-1, 2]
        mean2 = [1, -1]
        mean3 = [4, -4]
        mean4 = [-4, 4]
        cov = [[1.0,0.8], [0.8, 1.0]]
        X1 = np.random.multivariate_normal(mean1, cov, 50)
        X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 50)
        X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2


    def split_train(X1, y1, X2, y2):
        X1_train = X1[:80]
        y1_train = y1[:80]
        X2_train = X2[:80]
        y2_train = y2[:80]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[80:]
        y1_test = y1[80:]
        X2_test = X2[80:]
        y2_test = y2[80:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def accuracy(a, b):
        return (a/b)*100

    def plot_margin(X1_train, X2_train, clf):
        def f(x, w, b, c=0):
            # given x, return y such that [x,y] in on the line
            # w.x + b = c
            return (-w[0] * x - b + c) / w[1]

        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        # w.x + b = 0
        a0 = -4;
        a1 = f(a0, clf.w, clf.b)
        b0 = 4;
        b1 = f(b0, clf.w, clf.b)
        pl.plot([a0,b0], [a1,b1], "k")

        # w.x + b = 1
        a0 = -4;
        a1 = f(a0, clf.w, clf.b, 1)
        b0 = 4;
        b1 = f(b0, clf.w, clf.b, 1)
        pl.plot([a0,b0], [a1,b1], "k--")

        # w.x + b = -1
        a0 = -4;
        a1 = f(a0, clf.w, clf.b, -1)
        b0 = 4;
        b1 = f(b0, clf.w, clf.b, -1)
        pl.plot([a0,b0], [a1,b1], "k--")

        pl.axis("tight")
        pl.show()

    def plot_contour(X1_train, X2_train, clf):
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-6,6,50), np.linspace(-6,6,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z + 1, [0.0], colors='grey', linewidths=1, origin='lower')
        pl.contour(X1, X2, Z - 1, [0.0], colors='grey', linewidths=1, origin='lower')

        pl.axis("tight")
        pl.show()
        # pl.savefig(name + '_.png')

    def test_non_linear(c, lamda, flag):
        X1, y1, X2, y2 = gen_non_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)
        if flag:
            clf = SVM(gaussian_kernel)
        else:
            if lamda == 10:
                clf = SVM(rbf_kernel_10, C=c)
            if lamda == 50:
                clf = SVM(rbf_kernel_50, C=c)
            if lamda == 100:
                clf = SVM(rbf_kernel_100, C=c)
            if lamda == 500:
                clf = SVM(rbf_kernel_500, C=c)

        clf.fit(X_train, y_train)

        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))

        if flag:
            acc = accuracy(correct,len(y_predict))
            print('accuracy of dual svm guassian kernel is ', acc, 'percent')
        else:
            acc = accuracy(correct,len(y_predict))
            print('accuracy of dual svm rbf kernel lamda %d and C %f is ' %(lamda,c), acc)

        plot_contour(X_train[y_train == 1], X_train[y_train == -1], clf)

    def test_linear_primal():
        X1, y1, X2, y2 = gen_lin_separable_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)
        clf = SVM(C=0.1)
        clf.fit_primal(X_train, y_train)
        y_predict = clf.predict(X_test)
        y_predict = np.reshape(y_predict, (40,))
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
        acc = accuracy(correct,len(y_predict))
        print('accuracy of primal svm is ', acc)


    # alef + be :)
    test_linear_primal()

    # khe :)
    test_non_linear(0.01, 10, 1)

    test_non_linear(0.01, 10, 0)
    test_non_linear(0.01, 50, 0)
    test_non_linear(0.01, 100, 0)
    test_non_linear(0.01, 500,0)


    test_non_linear(0.1, 10, 0)
    test_non_linear(0.1, 50, 0)
    test_non_linear(0.1, 100, 0)
    test_non_linear(0.1, 500, 0)


    test_non_linear(0.5, 10, 0)
    test_non_linear(0.5, 50, 0)
    test_non_linear(0.5, 100, 0)
    test_non_linear(0.5, 500, 0)

    test_non_linear(1, 10, 0)
    test_non_linear(1, 50, 0)
    test_non_linear(1, 100, 0)
    test_non_linear(1, 500, 0)
