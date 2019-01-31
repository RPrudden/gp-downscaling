import GPy as gp
import numpy as np

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



