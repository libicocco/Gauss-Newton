import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp

spnorm = lambda x,t: LA.norm(x.data,t) if len(x.data) else 0

def spones(x):
    y = x.copy()
    y.data = (x.data > 0).astype(x.data.dtype)

def dirsq(c,x,g,h,ups):
    '''This routine computes the direction d whose ith component equals
    that of (-q_H(x;1),...,-q_H(x;n) if -q_H(x;j) is greater than or equal to
    upsilon*max_i{-q_H(x;i)}, otherwise equals zero.
    Here H is a diagonal positive definite matrix (or the identity matrix).
    It also outputs maxhR, which is norm(Hd_{H}(x),inf) and will be used
    for checking the termination.

    Args:
        c: a non-negative constant
        x: the current point
        g: the gradient at x
        h: the diagonal of Hessian approximation
        ups: a threshold for choosing J

    Returns:
        maxhR: norm(Hd_H(x),inf)
        d: a descent direction
        nonx: indices of nonzero components of d'''
    # R = d_{H}(x)

    #R = -median([ x' ; (g'+c)./h' ; (g'-c)./h' ]);
    #R = R';

    inv_h = h.copy()
    inv_h.data = 1/inv_h.data
    tmp = x - g*inv_h
    tmp2 = abs(tmp) - c*inv_h
    tmp2.data[tmp2.data < 0] = 0
    R = np.sign(tmp)*tmp2 - x
    hR = h*R
    Q = -g*R - 0.5*R*hR - c*abs(x+R) + c*abs(x)
    maxhR = max(abs(hR));

    # max_i{-q_H(x;i)}
    # set d(i)=R(i) if Q(i) > ups*maxQ

    maxQ = max(Q);
    indx = find(Q > ups*maxQ);
    d = sp.csc_matrix(R.shape[0]);
    d.data[Q > ups*maxQ] = R[Q > ups*maxQ];


def cgd_cont(y,A,tau):
    # init
    x = sp.csc_matrix((A.shape[1],y.shape[1]))
    maxiter = 1000
    targettau = tau # final value of tau
    nu = 0.5 # constant to control the reduced amount of tau
    Aty = A.T*y
    temptau = 0.01* spnorm(Aty, np.inf)
    tau = temptau if targettau > temptau else min(temptau,2*targettau)

    nzx = spones(x)
    resid = y - A*x
    f = 0.5*spnorm(resid,2)**2 + tau*spnorm(x,1)
    objective = [f]
    dir = 'sd'
    ups = 0.5 if dir == 'sq' else 0.9
    step = 1
    maxdtol = 1e-3

    h = np.clip([spnorm(A.getcol(i),2)**2 for i in range(A.shape[1])],1e-1,1e10)
    alpha = 1

    # main loop
    for it in range(maxiter):
        g = -A.T*resid
        maxd, d, nonx = dirsq(tau,x,g,h,ups)




if __name__ == '__main__':
    k, n = (100,20)
    y = sp.rand(k,1)
    A = sp.rand(k,n)
    tau = 0.1
    x = cgd_cont(y,A,tau)