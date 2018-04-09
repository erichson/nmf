import numpy as np
import scipy as sci
import timeit

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.available
mpl.style.use('seaborn-paper') 

#mpl.rcParams.update(mpl.rcParamsDefault)
   

import seaborn as sns
cmap = sns.cubehelix_palette(light=1, dark=0, start=3,  as_cmap=True)
import cmocean

import ristretto as ro

from scipy.io import loadmat

#******************************************************************************
# Read in Bird data
#******************************************************************************
dat = loadmat('/home/benli/Dropbox/Shared Projects/Sparse Tensor/Python/MackeviciusData.mat')
X = dat['SONG']

plt.figure()
plt.imshow(X,cmap=cmocean.cm.amp)
plt.show()

W, H, S = rconvnmf_hals(X, k=15, l=3, gamma=0.8, tol=1e-15, max_iter=100, random_state=None, verbose=True)

plt.figure()
plt.imshow(S,cmap=cmocean.cm.amp)
plt.show()

XRe = cnmf_reconstruct(W, H)
plt.figure()
plt.imshow(XRe, cmap=cmocean.cm.amp)
plt.colorbar()

sci.linalg.norm(X-XRe) / sci.linalg.norm(X) 


plt.figure()
plt.imshow(W[:,0,:], cmap=cmocean.cm.amp)
plt.show()

plt.figure()
plt.imshow(W[:,1,:], cmap=cmocean.cm.amp)
plt.show()




#******************************************************************************
# Read in Neuro data
#******************************************************************************
dat = loadmat('/home/benli/Dropbox/Shared Projects/Sparse Tensor/Python/MackeviciusData.mat')
X = dat['NEURAL']

plt.figure()
plt.imshow(X,cmap=cmocean.cm.amp)
plt.show()

W, H, S = rconvnmf_hals(X, k=2, l=2, gamma=10.5, tol=1e-15, max_iter=100, random_state=None, verbose=True)

plt.figure()
plt.imshow(S,cmap=cmocean.cm.amp)
plt.show()

XRe = cnmf_reconstruct(W, H)
plt.figure()
plt.imshow(XRe, cmap=cmocean.cm.amp)
plt.colorbar()

plt.figure()
plt.imshow(W[:,0,:], cmap=cmocean.cm.amp)
plt.show()

plt.figure()
plt.imshow(W[:,1,:], cmap=cmocean.cm.amp)
plt.show()





#******************************************************************************
# Toydata
#******************************************************************************

Wtrue1 = np.array([[0,0], [0,0], [8.,0], [0,0], [0,0], [0,9.], [0,0], [0,0], [0,0]])
Wtrue2 = np.array([[0,0], [0,0], [0,0], [8.,0], [0,0], [0,0], [0,9.], [0,0], [0,0]])

Htrue = np.array([[0,0], [15.,0], [0,0], [0,0], [0,15.], [0,0], [0,0]]).T

Xtest = Wtrue1.dot(shift(Htrue,0))
Xtest += Wtrue2.dot(shift(Htrue,1))
      
test = np.asarray( [Wtrue1.T,Wtrue2.T] ).T      
cnmf_reconstruct(test, Htrue)        


plt.figure()
plt.imshow(Xtest,cmap=cmocean.cm.amp)


W, H, S = rconvnmf_hals(Xtest, k=2, l=2, gamma=150, tol=1e-5, max_iter=150, random_state=None, verbose=True)

XRe = cnmf_reconstruct(W, H)
plt.figure()
plt.imshow(XRe, cmap=cmocean.cm.amp)
plt.colorbar()

plt.figure()
plt.imshow(W[:,0,:], cmap=cmocean.cm.amp)
plt.show()

plt.figure()
plt.imshow(W[:,1,:], cmap=cmocean.cm.amp)
plt.show()

