# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 08:55:07 2018

@author: Элвис
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=np.nan)

#initial data
Ox = 20
Nx = 500
T = 1000
dt = 0.1
h = 5. #width of potential
x0 = 0 #center of gap
d = 2. #0.5 width of gap
xc = x0-d-h/2 #center left of potential

Psi = np.zeros((Nx, T), dtype=np.complex128) #matrix of wave-function
W = np.zeros((Nx, T), dtype=np.complex128) #matrix of probability
Um = np.zeros((Nx,), dtype=np.complex128) #matrix of potential
Wt = np.zeros((Nx,), dtype=np.complex128) #matrix of probability in time-space
xarr = np.linspace(-Ox, Ox, Nx)
dx = np.fabs(xarr[0]-xarr[1])
eta = dt/dx**2
print(eta)

#function of potential
def U(x, x0, h, d, U0):
    if x>=x0-d-h and x<=x0-d:
        return U0
    elif x>=x0+d and x<=x0+d+h:
        return U0
    else:
        return 0

#function of normalization
def norma(psi, xarr):
    return integrate.simps(psi*np.conj(psi), xarr)

#initial WF of particle
def Psi0(x, xc, sigma):
    return np.exp(-0.5*(x-xc)**2/sigma**2)/(sigma*np.sqrt(2*np.pi))

#initial distribution of WF
Psi[:,0] = Psi0(xarr, xc, 0.5*h)
Psi[:,0] = Psi[:,0]/np.sqrt(norma(Psi[:,0], xarr))
W[:,0] = Psi[:,0]*np.conj(Psi[:,0])
U0 = -2.*np.amax(Psi[:,0]*np.conj(Psi[:,0])).astype(float)
for i in range(Nx):
    x = xarr[i]
    Um[i] = U(x, x0, h, d, U0)


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

ani = animation.FuncAnimation(fig, updatefig, np.arange(0, T), interval=1)
ani.save('QC.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

plt.show()