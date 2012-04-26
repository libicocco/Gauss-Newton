#!/usr/bin/python
from sympy import *
import random

def main():
    a,b,c,realx= [random.random() for i in range(4)]

    x, e, sq_e, jtj = symbols('x e sq_e jtj')

    f = x**30 + a*(x**2) + b*x + c
    #f = x**2/(a**2 + x**2)

    y = f.subs(x,realx).evalf()

    e = y - f

    sq_e = e**2

    def optimize(currx, step_exp, acc = 1e-15):
        curre = lambda currx: sq_e.subs(x,currx).evalf()
        for it in range(100):
            currx += step_exp.subs(x,currx).evalf()
            if curre(currx) < acc: break
        return currx,it

    x0 = 1 + random.random()

    step_n = -diff(sq_e,x) / diff( diff(sq_e,x) , x)
    finalx_n, it_n = optimize(x0, step_n)
    print('Newton\t\t: Real x %s and final x %s in %s it' % (realx, finalx_n, it_n))

    #step_n_vb = -e*diff(e,x) / (diff(e,x)**2 +e*diff(diff(e,x),x))
    step_n_vb = e*diff(f,x) / (diff(f,x)**2 -e*diff(diff(f,x),x))
    finalx_n_vb, it_n_vb = optimize(x0, step_n_vb)
    print('Newton verbose\t: Real x %s and final x %s in %s it' % (realx, finalx_n_vb, it_n_vb))

    step_gn = e*diff(f,x) / (diff(f,x)**2)
    finalx_gn, it_gn = optimize(x0, step_gn)
    print('Gauss-Newton\t: Real x %s and final x %s in %s it' % (realx, finalx_gn, it_gn))

    step_weird = -e*diff(e,x) / (diff(e,x)**2 -e*diff(diff(e,x),x))
    finalx_w, it_w = optimize(x0, step_weird)
    print('Weird\t\t: Real x %s and final x %s in %s it' % (realx, finalx_w, it_w))

    step_weird_sq = -sq_e*diff(sq_e,x) / (diff(sq_e,x)**2 -sq_e*diff(diff(sq_e,x),x))
    finalx_w_sq, it_w_sq = optimize(x0, step_weird_sq)
    print('Squared weird\t: Real x %s and final x %s in %s it' % (realx, finalx_w_sq, it_w_sq))

if __name__ == '__main__':
    for i in range(10):
        main()
        print('=======')
