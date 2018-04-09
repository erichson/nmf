"""
CP decomposition by classic alternating least squares (ALS).

Author: N. Benjamin Erichson <erichson@uw.edu>
"""

import numpy as np
import scipy as sci

import pyximport; pyximport.install()
from _rconnmf_update import _rconnmf_update

from functools import reduce

def shift(X, j):
    if j == 0:
        return(X)
        
    elif j > 0:
        X_shifted = np.zeros(X.shape)
        X_shifted[:, j::] = X[:, :-j:]
        return(X_shifted)

    elif j < 0:
        X_shifted = np.zeros(X.shape)
        X_shifted[:, :j:] = X[:, -j::]
        return(X_shifted)
        



def cnmf_reconstruct(W, H):

    m, k, l = W.shape  

    X = sum([W[:,:,j].dot(shift(H, j)) for j in range(l)])
    
    return(X)
 
#test = np.array([[1,2,3,4], [5,6,7,8]])  
#shift(test, 1)
#shift(test, -2)    
    

def rconvnmf_hals(X, k, l, gamma=0, tol=1e-5, max_iter=200, random_state=None, init='normal', verbose=True):
    """
    % DESCRIPTION:
    %
    %   Factorizes the m x n data matrix X into k components. 
    %   Factor exemplars are returned in the m x k x l tensor W
    %   Factor timecourses are returned in the kxn matrix H
    %
    %                                    ----------    
    %                               l   /         /|
    %                                  /         / |
    %        ----------------         /---------/  |          ----------------
    %        |              |         |         |  |          |              |
    %      m |      X       |   =   m |    W    |  /   (*)  k |      H       |           
    %        |              |         |         | /           |              |
    %        ----------------         /----------/            ----------------
    %               n                     k                          n
    %
    % ------------------------------------------------------------------------
    """

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error catching
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~     
    m, n = X.shape  
    
    if (X < 0).any():
        raise ValueError("Input matrix with nonnegative elements is required.")    
    
    if  X.dtype == sci.float32: 
        data_type = sci.float32
        
    elif X.dtype == sci.float64: 
        data_type = sci.float64  

    else:
        raise ValueError("X.dtype is not supported.")    
    
    
    #--------------------------------------------------------------------
    #   Set Tuning Parameters
    #--------------------------------------------------------------------  
    #nu   = 1.0 / (1 + beta)
    #kappa = nu * alpha
        

    

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                            
    # Initialization methods for factor matrices W and H
    # 'normal': nonnegative standard normal random init
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    
    if init == 'normal':
        m, n = X.shape
        W =  sci.maximum(0.0, sci.random.standard_normal((m, k, l)))
        Ht = sci.maximum(0.0, sci.random.standard_normal((n, k)))

        #W =  sci.maximum(0.0, sci.random.uniform(size=(m, k, l)))
        #Ht = sci.maximum(0.0, sci.random.uniform(size=(n, k)))
                
    else:
        raise ValueError('Initialization method is not supported.')
    #End if
    
    
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Iterate the HALS algorithm until convergence or maxiter is reached
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
    
    fitchange = []
    obj = []
    itr = 0
    
    S = sci.zeros(X.shape)    

    
    while max_iter > itr:
        violation = 0.0
        
        
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update input matrix 
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        XS = X - S

        if (XS < 0).any():
            raise ValueError("Input matrix with nonnegative elements is required.")       



        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update H
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
        Ht_old = Ht.copy()
        Ht = np.zeros(Ht.shape)
        
        for i in range(l):   
            # Update factor matrix H
            WtW = W[:,:,i].T.dot(W[:,:,i])
            
            XtW = XS.T.dot(W[:,:,i])
  
          
            sumHtWtj = np.zeros((n,m))
            for j in range(l):
                if j != i:
                    sumHtWtj += shift(Ht.T, j).T.dot(W[:,:,j].T)
            
            temp = XtW - sumHtWtj.dot(W[:,:,i])
            
            Ht_update = np.array(shift(Ht_old.T, i).T, order='C')

            
            violation += _rconnmf_update(Ht_update, WtW, temp)

            Ht += shift(Ht_update.T, -i).T
        
        
        # Average over updated HT
        # Is this necessary ??
        Ht /= l

        # Renormalize so rows of H have constant energy
        norms = np.sum(Ht**2, axis=0)**0.5
        Ht /= norms
    
        for i in range(l):
            W[:, :, i] *= norms




        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Update W
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #W_old = W.copy()

        for i in range(l):
            # Update factor matrix W
            Ht_shifted = np.array(shift(Ht.T, i).T, order='C')
            XHt = XS.dot(Ht_shifted) 
    

            sumWHj = np.zeros((m,n))
            for j in range(l):
                if j != i:
                    sumWHj += W[:,:,j].dot(shift(Ht.T, j))
                        

            temp = XHt - sumWHj.dot(Ht_shifted)
            
    
            HHt = Ht_shifted.T.dot(Ht_shifted)
            
            W_update = np.array(W[:,:,i], order='C')

            violation += _rconnmf_update(W_update, HHt, temp)
            
            W[:,:,i] = W_update







        # compute residual
        R = X - cnmf_reconstruct(W, Ht.T)  

        
        # l1 soft-threshold for robustification
        idxH = R > gamma
        idxL = R <= -gamma
        S = np.zeros( S.shape )
        S[idxH] = R[idxH] - gamma    
        S[idxL] = R[idxL] + gamma    
         
        
        # compute objective
        #alphaB = np.sum([alpha * sci.sum(np.abs(U[i])) for i in range(N)])
        #betaB = np.sum([beta * sci.sum(U[i]**2) for i in range(N)] )       
        
        #obj = ( 0.5 * sci.sum((R-S)**2) + alphaB + 0.5 * betaB + gamma * sci.sum(abs(S)))  
        obj.append( 0.5 * sci.sum((R-S)**2) + gamma * sci.sum(abs(S)))  


        # Compute stopping condition.
        if itr == 0:
            violation_init = violation

        if violation_init == 0:
            break       

        fitchange = violation / violation_init
        
        if verbose == True:
            print('Iteration: %s obj: %s, fitchange: %s' %(itr, obj[-1], violation))        

        if fitchange <= tol:
            break
        
        if itr > 1 and obj[-2] - obj[-1] <= tol:
            break

        itr += 1
    #End for

        
    
    
    return( W, Ht.T, S )






 





