import GPy as gp
import numpy as np
import autograd.numpy as np
from autograd import grad

def compute_cov(l, d=1, sdim=None, X=None):
    ker1 = gp.kern.Matern52(1, lengthscale=l)
    if not X:
        if d == 1:
            xx = np.mgrid[0:sdim]
            X = np.array([xx]).T
        elif d == 2:
            xx, yy = np.mgrid[0:sdim, 0:sdim]
            X = np.vstack((xx.flatten(), yy.flatten())).T
        else:
            raise ValueError("Dimension > 2 not supported")
    C = ker1.K(X)
    return C


def logistic(x, k=1, L=1, x0=0):
    return L / (1 + np.exp(-k * (x - x0)))


def log_p_i(yi, fi):
    return -np.log(np.ones_like(fi) + np.exp(-yi*fi))


# Compute gradients
log_p = lambda y, f: np.product(np.vectorize(log_p_i)(y, f))
d_log_pis = [log_p_i]
d_log_ps = [log_p]
for i in range(3):
    d_log_p_i = grad(d_log_ps[i])
    d_log_p = lambda y, f: np.product(np.vectorize(d_log_p_i))
    d_log_pis.append(d_log_p_i)
    d_log_ps.append(d_log_p)
_, d_log_p_df, d2_log_p_df2, d3_log_p_df3 = d_log_ps


def laplace_approximation(K, y):
    f = np.zeros_like(y)
    converged = False
    while not converged:
        W = -d2_log_p_df2(y, f)
        W2 = sp.linalg.sqrtm(W)
        B = np.eye(K.shape[0]) + np.linalg.multi_dot([W2, K, W2])
        L = np.linalg.cholesky(B)
        b = np.dot(W, f) + d_log_p_df(y, f)
        a3 = np.linalg.solve(L, np.linalg.multi_dot([W2, K, b]))
        a2 = np.linalg.solve(np.dot(W2, L.T), a3)
        a = b - a2
        f = np.dot(K, a)
        
        obj = -0.5 * np.dot(a.T, f) + log_p(y, f)
        converged = (obj < 0.001)

    log_q = -0.5 * np.dot(a.T, f) + log_p(y, f) + np.trace(np.log(L))
    return (f, log_q)
