import numpy as np
from numpy import linalg as LA
import scipy.sparse as sp
from itertools import ifilter

spnorm = lambda x,t: LA.norm(x.data,t) if len(x.data) else 0

def spones(x):
    y = x.copy()
    y.data = (x.data > 0).astype(x.data.dtype)
    return y

def findstep(Ad, resid, x, d, tau):
    '''This routine compute the stepsize by finding the minimum of
    piecewise quadratic function

    min {0.5*norm(resid-alp*Ad)^2 + tau*norm(x+alp*d,1) : alp > 0}

    Args:
        x   : the current point
        d   :  the current descent direction
        tau : a non-negative real parameter of the objective function

    Returns:
        alpha: the stepsize'''

    # Nonzero component of vector d
    nd = d.data
    nx = np.asarray(x[d.nonzero()]).flatten()

    # Reformulate the piecewise quadratic function as
    # 0.5*a*alp^2 - b*alp + tau*sum_{i \in nonz} |x_i+alp*d_i| + c
    a = Ad.T.dot(Ad)[0,0]
    b = Ad.T.dot(resid)[0,0]

    tmp = -nx/nd
    idx = np.argsort(tmp)
    breakpts = np.asarray(tmp).flatten()[idx]
    nx = nx[idx]
    nd = nd[idx]
    normnd = LA.norm(nd,1)

    objfunc = lambda alp: (0.5*a)*alp*alp - b*alp + tau*LA.norm(nx+alp*nd,1)
    alphafunc = lambda norm: (b - tau*norm)/a

    if (breakpts <= 0).all():
        if a==0 :
            print('Something is wrong')
        else:
            alpha = alphafunc(normnd)
    else:
        # pick the first occurrence of tied break points
        # and find the first positive break-point with
        # increased objective value
        idxpos2 = [k for k in range(0,len(breakpts)) if
                   ((breakpts[k] - (breakpts[k-1] if k>0 else -1)) > 1e-13 and
                    breakpts[k] > 0)]

        alps =  list(breakpts[idxpos2])

        i_alp = next(ifilter(lambda k: objfunc(alps[k]) >
                             (objfunc(alps[k-1]) if k>0 else objfunc(np.zeros_like(breakpts[0]))),
                             range(len(alps))), None)

        def choose_alp(alp,objold,alp1,alp2):
            alp1,f1 = (alp,objold) if (alp1 > alp) else (alp1,objfunc(alp1))
            alp2,f2 = (alp,objold) if (alp2 < alp) else (alp2,objfunc(alp2))
            return [alp1,alp2,alp][np.argmin([f1,f2,objold])]

        alp_norm = lambda i: normnd - 2*LA.norm(nd[idxpos2[i]:],1)

        # find no positive break-point with
        # increased objective value
        if i_alp == None:
            i_alp = -1
            alp = alps[i_alp]
            alpha = alp if a==0 else choose_alp(alp,objfunc(alp),
                                                alphafunc(alp_norm(i_alp)),
                                                alphafunc(normnd))
        else:
            if (i_alp==0):
                if (a==0):
                    print('Something is wrong')
                else:
                    alpha = alphafunc(alp_norm(i_alp))
            else:
                alptest = alps[i_alp-1]
                alp = alps[i_alp]
                # find the exact minimum in either the interval
                # [breakpts(idxpos2(k-2)), alptest] or
                # [alptest, breakpts(idxpos2(k))]
                alpha = alptest if a==0 else choose_alp(alptest,objfunc(alp),
                                                        alphafunc(alp_norm(i_alp-1)),
                                                        alphafunc(alp_norm(i_alp)))

    return (alpha,breakpts)


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

    tmp = x - (g.T/h).T
    tmp2 = abs(tmp).T - c/h
    tmp2[tmp2 < 0] = 0
    R = np.multiply(np.sign(tmp),tmp2.T) - x
    hR = np.multiply(h,R.T)
    Q = -np.multiply(g,R) - 0.5*np.multiply(R,hR.T) - c*abs(x+R) + c*abs(x)
    maxhR = np.max(abs(hR))

    # max_i{-q_H(x;i)}
    # set d(i)=R(i) if Q(i) > ups*maxQ

    maxQ = np.max(Q)
    indx = np.where(np.asarray(Q > ups*maxQ).squeeze())[0]
    d = sp.csc_matrix((np.asarray(R).squeeze()[indx],(indx,np.zeros_like(indx))),shape=R.shape)

    return (maxhR, d, indx)


def cgd_cont(y,A,tau):
    # init
    x = sp.csc_matrix((A.shape[1],y.shape[1]))
    maxiter = 1000
    targettau = tau # final value of tau
    nu = 0.5 # constant to control the reduced amount of tau
    Aty = A.T.dot(y)
    temptau = 0.01* LA.norm(Aty, np.inf)
    tau = targettau if targettau > temptau else min(temptau,2*targettau)

    nzx = spones(x)
    resid = y - A.dot(x)
    f = 0.5*LA.norm(resid,2)**2 + tau*spnorm(x,1)
    objective = [f]
    ups = 0.5 # if dir = 'sq' else 0.9
    step = 1
    maxdtol = 1e-3

    h = np.clip([spnorm(A.getcol(i),2)**2 for i in range(A.shape[1])],1e-1,1e10)
    alpha = 1

    # main loop
    for it in range(maxiter):
        g = -A.T.dot(resid)
        maxd, d, nonx = dirsq(tau,x,g,h,ups)
        relmaxd = maxd/max(1,np.max(abs(x.todense())))
        if (relmaxd < maxdtol):
            if(tau == targettau):
                break;
        Ad = A.dot(d)
        step,dummy = findstep(Ad,resid,x,d,tau);

        fprev = f
        xprev = x
        nzx_prev = nzx
        x = x + step*d
        resid = resid - step*Ad
        f = 0.5*LA.norm(resid,'fro')**2 + tau*spnorm(x,1)
        nzx = spones(x)
        num_nzx = nzx.nnz

        # Update the threshold for choosing J, based on the current stepsize.
        if (step > 10):
            ups = max(1e-2,0.80*ups) # old: ups = max(1e-2,0.50*ups);
            alpha = max(alpha/step,1)
            h = alpha*np.ones(x.shape[0]*x.shape[1])
        elif (step > 1.0):
            ups = max(1e-2,0.90*ups) # old: ups = max(1e-2,0.5*ups);
        elif (step > 0.5):           # old: not present
            ups = max(1e-2,0.98*ups)
        elif (step < 0.1):
            ups = min(0.2,2*ups)     #  ups = min(0.2,10*ups);
            alpha = min(alpha/step,1)
            h = alpha*np.ones(x.shape[0]*x.shape[1])

        objective.append(f)
        print('%d %f %f %d %f %f %f %f' % (it,f,step,nzx.nnz,ups,relmaxd,alpha,tau))
    return (x,objective)

if __name__ == '__main__':
    A_nsp = np.array([[0.7952,0.1626,0.6991,0.2435,0.5497],
                      [0.1869,0.1190,0.8909,0.9293,0.9172],
                      [0.4898,0.4984,0.9593,0.3500,0.2858],
                      [0.4456,0.9597,0.5472,0.1966,0.7572],
                      [0.6463,0.3404,0.1386,0.2511,0.7537],
                      [0.7094,0.5853,0.1493,0.6160,0.3804],
                      [0.7547,0.2238,0.2575,0.4733,0.5678],
                      [0.2760,0.7513,0.8407,0.3517,0.0759],
                      [0.6797,0.2551,0.2543,0.8308,0.0540],
                      [0.6551,0.5060,0.8143,0.5853,0.5308]])
    A = sp.csc_matrix(A_nsp)
    y = np.array([[0.4359],[0.4468],[0.3063],[0.5085],[0.5108],[0.8176],[0.7948],[0.6443],[0.3786],[0.8116]])
    tau = 0.9
    x,obj = cgd_cont(y,A,tau)
    print(x)
    print(spnorm(x,1))
    print(LA.norm(A.dot(x)-y))
    print(A.dot(x)-y)

    # Ad = np.array([[0.8147],[0.9058],[0.1270],[0.9134],[0.6324],[0.0975],[0.2785],[0.5469],[0.9575],[0.9649]])
    # resid = np.array([[0.1576],[0.9706],[0.9572],[0.4854],[0.8003],[0.1419],[0.4218],[0.9157],[0.7922],[0.9595]])
    # d = sp.csc_matrix([[0],[0],[0.4387],[0.3816],[0.7655],[0],[0],[0],[0],[0]])
    # x = np.array([[-0.7060],[-0.0318],[-0.2769],[-0.0462],[-0.0971],[0.8235],[0.6948],[0.3171],[0.9502],[0.0344]])
    # print(findstep(Ad,resid,x,d,0.5))
