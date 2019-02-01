import GPy as gp
import numpy as np
import autograd.numpy as np
from autograd import grad

def compute_cov(l, d=1, sdim=None, X=None, ker=gp.kern.Matern52):
    ker1 = ker(d, lengthscale=l)
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


def negloglik(l, l_A, ys, d=1, sdim=None, X=None):
    C = compute_cov(l, d, sdim, X)
    A = compute_cov(l_A, d, sdim, X)
    ACA = np.linalg.multi_dot([A, C, A])

    return -np.sum([sp.stats.multivariate_normal.logpdf(y.flatten(), mean=None, cov=ACA, 
                                                        allow_singular=True) for y in ys])


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

    log_q = -0.5 * np.dot(a.T, f) + log_p(y, f) + np.sum(np.log(np.diag(L)))
    return (f, log_q)


def RBF(l, v=1):
    def kern(x, x_):
        return v**2 * np.exp((-(x-x_).T(x-x)) / (2 * l**2))
    return kern


def dRBF_dl(l, v=1):
    def kern(x, x_):
        return v**2 * np.exp((-(x-x_).T(x-x)) / (2 * l**2)) * (-(x-x_).T(x-x)) / l**3
    return kern


def loglik_la(X, y, l):
    K = compute_cov(l, X, ker=RBF)
    f, a = laplace_approximation(K, y)
    W = -d2_log_p_df2(y, f)
    W2 = sp.linalg.sqrtm(W)
    B = np.eye(K.shape[0]) + np.linalg.multi_dot([W2, K, W2])
    L = np.linalg.cholesky(B)
    z = -0.5 * np.dot(a.T, f) + log_p(y, f) + np.sum(np.log(np.diag(L)))
    R1 = np.linalg.solve(L, W2)
    R = np.linalg.solve(np.dot(W2, L), R1)
    C = np.linalg.solve(L, np.dot(W2, K))
    s2 = -0.5 * np.diag(np.diag(K) - np.diag(np.dot(C.T, C))) * d3_log_p_df3(y, f)

    C = compute_cov(l, X, ker=dRBF_dl)
    s1 = 0.5 * np.linalg.multi_dot([a.T, C, a]) - 0.5 * np.trace(np.dot(R, C))
    b = np.dot(C, d_log_p_df)
    s3 = b - np.linalg.multi_dot([K, R, b])
    dz_dl = s1 + np.dot(s2.T, s3)

    return (z, dz_dl)
