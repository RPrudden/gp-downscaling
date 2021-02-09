import iris
import numpy as np
import scipy as sp
import GPy as gp
from scipy import signal
from scipy.fftpack import fft2, fftshift, ifft2
from functools import partial


# ## Covariances

def compute_cov_block(l, sdim):
    ar = np.zeros([sdim,sdim,sdim,sdim])
    
    for i in range(0, sdim, l):
        for j in range(0, sdim, l):
            ar[i:i+l, j:j+l, i:i+l, j:j+l] = 1
    
    ar = ar.reshape(sdim//l, l, sdim//l, l, sdim, sdim).mean(axis=(1,3))
    return ar.reshape((sdim//l)**2, (sdim//1)**2) / l**2


def compute_cov(l, sdim, v=1):
    ker1 = gp.kern.Matern52(2, lengthscale=l, variance=v)
    xx, yy = np.mgrid[0:sdim, 0:sdim]
    X = np.vstack((xx.flatten(), yy.flatten())).T
    C = ker1.K(X)
    return C


def make_pd(A):
    c = 0.005
    return A + np.eye(A.shape[0]) * c


# ## Conditioning

def negloglik(l, sdim, A, y, v=1):
    C = compute_cov(l, sdim, v=v)
    C = make_pd(C)
    ACA = np.linalg.multi_dot([A, C, A.T])
    
    nll = -sp.stats.multivariate_normal.logpdf(y.flatten(), mean=None, cov=ACA, allow_singular=True)# / 20000
    return nll


def fit_v(f, A, l, sdim):
    ml = sp.optimize.minimize_scalar(lambda v: negloglik(l, sdim, A, f, v),
                                         method='Bounded', bounds=(0.1, 10000), tol=0.5).x
    return ml


def condition(f, l, a2, sdim):
    C1 = compute_cov(l, sdim, v=1)
    v = fit_v(f, a2, l, sdim)
    
    C = compute_cov(l, sdim, v=v)
    C_hl = np.dot(a2,C)
    C_l = np.linalg.multi_dot([a2,C,a2.T])
    C_l = make_pd(C_l)

    C_p = C - np.linalg.multi_dot([C_hl.T, np.linalg.pinv(C_l), C_hl])
    C_p = make_pd(C_p)
#     cp2 = sp.linalg.sqrtm(C_p)
    cp2 = sp.linalg.cholesky(C_p).T
    mu_p = np.linalg.multi_dot([C_hl.T, np.linalg.pinv(C_l), f.flatten()])
    
    return (mu_p, C_p, cp2)


# ## Benchmark

def bicub(f, l, sdim):
    xx, yy = np.mgrid[0:sdim, 0:sdim]
    x = xx[l//2:sdim:l,0]
    y = yy[0,l//2:sdim:l]

    interp = sp.interpolate.interp2d(x, y, f, kind='cubic')
    return interp(xx[:,0], yy[0,:])


def expand(f, d):
    return np.repeat(np.repeat(f, d, axis=1), d, axis=0)


# ## Verification

def get_neighbourhoods(X, n, sdim):
    X = X.reshape(sdim, sdim)
    
    s = np.array(list(np.ndindex(n, n))).T
    f = np.zeros((n, n, s.shape[1]))
    f[s[0], s[1], range(s.shape[1])] = 1
    
    return signal.convolve(X[:,:,np.newaxis], f)


def process_psd(t, sdim):
    test = fftshift(fft2(t.reshape(sdim, sdim)).real)

    x,y = np.meshgrid(np.arange(sdim),np.arange(sdim))
    R = np.sqrt(x**2+y**2)

    f = lambda r : np.abs(test[(R >= r-.5) & (R < r+.5)]).mean()
    r = np.linspace(1,30,num=30)
    mean = np.abs(np.vectorize(f)(r))
    return mean


def process_samples(s, map_s, comp_s, ens_reducer):
    res = {}
    orig = map_s(s['orig_hr'])
    
    for i in ['mean', 'orig_lr', 'bicub']:
        res[i] = comp_s(orig, map_s(s[i]))
        
    samples = [comp_s(orig, map_s(sample)) for sample in s['samples']]
    res['GRF'] = ens_reducer(samples)
    
    return res


get_psd_mse = lambda s, d: process_samples(s, 
                                           map_s = partial(process_psd, sdim=d), 
                                           comp_s = lambda x, y: np.mean(np.square(x - y)),
                                           ens_reducer = np.mean)


get_psd_wass = lambda s, d: process_samples(s, 
                                            map_s = partial(process_psd, sdim=d), 
                                            comp_s = lambda x, y: sp.stats.wasserstein_distance(x, y),
                                            ens_reducer = np.mean)


get_mse = lambda s: process_samples(s,
                                    map_s = lambda x: x,
                                    comp_s = lambda x, y: np.mean(np.square(x - y)),
                                    ens_reducer = np.mean)


mk_spatial_wass_fn = lambda k: lambda s, d: process_samples(s,
                                                            map_s = lambda x: partial(get_neighbourhoods, sdim=d)(x, k),
                                                            comp_s = lambda x, y: sp.stats.wasserstein_distance(x.flatten(), y.flatten()),
                                                            ens_reducer = np.mean)


get_spatial_wass_4 = mk_spatial_wass_fn(4)
get_spatial_wass_8 = mk_spatial_wass_fn(8)
