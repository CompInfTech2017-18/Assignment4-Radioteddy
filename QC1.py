# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 08:55:07 2018

@author: Элвис
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate

Ox = 20
Nx = 500
T = 1000
dt = 0.1
xc = 3.
h = 3.

def norma(psi, xarr):
    return integrate.simps(psi*np.conj(psi), xarr)

def Psi0(x, xc, sigma):
    return np.exp(-0.5*(x+xc)**2/sigma**2)/(sigma*np.sqrt(2*np.pi))

Psi = np.zeros((Nx, T), dtype=np.complex128) #matrix of wave-function
W = np.zeros((Nx, T), dtype=np.complex128) #matrix of probability
Um = np.zeros((Nx,), dtype=np.complex128) #matrix of potential
Wt = np.zeros((Nx,), dtype=np.complex128) #matrix of probability in time-space
xarr = np.linspace(-Ox, Ox, Nx)
dx = np.fabs(xarr[0]-xarr[1])
eta = dt/dx**2
print(eta)


Psi[:,0] = Psi0(xarr, xc, 0.5*h)
Psi[:,0] = Psi[:,0]/np.sqrt(norma(Psi[:,0], xarr))
W[:,0] = Psi[:,0]*np.conj(Psi[:,0])
U0 = -2.*np.amax(Psi[:,0]**2).astype(float)
Um[(xarr >= -xc - 0.6*h)*(xarr <= -xc + 0.6*h)] = U0
Um[(xarr >= xc - 0.6*h)*(xarr <= xc + 0.6*h)] = U0




#first step - explicit scheme
Psi[1:-1,1] = Psi[1:-1,0] + dt/1j*Um[1:-1]*Psi[1:-1,0] + 1j*eta*(Psi[2:,0] - 2*Psi[1:-1,0] + Psi[:-2,0])
Psi[:,1] = Psi[:,1]/np.sqrt(norma(Psi[:,1], xarr))
W[:,1] = Psi[:,1]*np.conj(Psi[:,1])


#next steps - Dufort-Frankel scheme
for i in range(2, T):
    Psi[1:-1,i] = (2j*eta*(Psi[2:,i-1] + Psi[:-2,i-1]\
        - Psi[1:-1,i-2]) - 2j*dt*Um[1:-1]*Psi[1:-1,i-1] + Psi[1:-1,i-2])/(1. + 2j*eta)
    Psi[:,i] = Psi[:,i]/np.sqrt(norma(Psi[:,i], xarr))
    W[:,i] = Psi[:,i]*np.conj(Psi[:,i])


fig = plt.figure()
ax = plt.subplot()


def updatefig(i):
   Wt = W[:,i]
   ax.clear()
   ax.set_ylim(-np.fabs(U0) - .2, np.fabs(U0) + .2)
   ax.plot(xarr, Um, color='black', linewidth=1.0)
   line = ax.plot(xarr, Wt, color='blue', linewidth=1.0)
   return line,

ani = animation.FuncAnimation(fig, updatefig, np.arange(0, T), interval=0)

plt.show()