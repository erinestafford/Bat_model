import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import random

def resources_i(t,Ri,vi,ki,beta,bi):
    return Ri*(vi*(1-Ri/ki) - beta*bi)

#def mu(t):
#    return 0.02

def energy(t, E,phi, beta, bi, Ri):
    return phi*beta*sum(bi*Ri) - mu(t)*E

def location(t,Bi,p):
    temp = 0
    for i in range(len(Bi)):
        for j in range(len(Bi)):
            temp = temp + p[j,i]*Bi[j] - p[i,j]*Bi[i]
    return temp

def all(t,v ,vi,ki,beta, phi, mu, num_p,p):
    out=np.zeros(len(v))
    rec = v[0:num_p]
    e = v[num_p]
    bis = v[num_p+1:]
    out[0:num_p] = rec*(vi*(1-rec/ki) - beta*bis)
    out[num_p] = phi*beta*sum(bis*rec) - mu*e
    temp = np.zeros(len(v[num_p+1:]))
    for i in range(len(bis)):
        for j in range(len(bis)):
            temp[i] = temp[i] + p[j, i] * bis[j] - p[i, j] * bis[i]
    out[num_p+1:] = temp
    return out

p = np.empty(5)
p.fill(1.0)
p = np.random.dirichlet(p, 5)
vi = 0.5*np.ones(5)
ki = 1000*np.ones(5)
beta = 0.05
phi = 0.9
mu = 0.02
num_p = 5
F = lambda t,v: all(t,v ,vi,ki,beta, phi, mu, num_p,p)
t_eval = np.arange(0, 100, 1)
sol = solve_ivp(F, [0, 100], [100,100,100,100,100,5000,10,10,10,10,10], t_eval=t_eval)
plt.plot(t_eval,sol.y[num_p,:])
plt.show()

