# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 08:55:07 2018

@author: Элвис
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.integrate as integrate

U0 = -5. #deep of potential
h = 2. #width of potential
d = 0.2 #gap between potentials
xc = 2.5 #center of symmetry
dx = 0.01 #coordinate step
dt = 0.1 #time step
eta = dt/dx**2
Ox = 2*h + 1. #x-size of system
Nx = int(Ox/dx) 
T = 1000
x0 = 0 #first point
Psi = np.zeros((Nx, T), dtype=complex) #matrix of wave-function
W = np.zeros((Nx, T), dtype=complex) #matrix of probability
Um = np.zeros((Nx,), dtype=complex) #matrix of potential
xarr = np.zeros((Nx,), dtype=complex) #matrix of x-steps
Wt = np.zeros((Nx,), dtype=complex) #matrix of probability in time-space
#A = np.zeros((Nx, Nx), dtype=complex)
#B = np.zeros((Nx,), dtype=complex)
#Psif = np.zeros((Nx,), dtype=complex)
#print(j)

def U(x): #potential
    if x >= xc-d-h and x<=xc-d:
        return U0
    elif x >= xc+d and x<=xc+d+h:
        return U0
    else:
        return 0

def Psi0(x): #initial conditions
    if x >= xc-d/2-h and x<=xc-d/2: 
        return np.exp(-(x-(xc-d-h/2))**2/(0.01*h**2))
       # return np.sqrt(2/h)*np.sin(np.pi*x/h) + np.sqrt(2/h)*np.sin(2*np.pi*x/h)
    else:
        return 0
    
def norma(psi, xarr):
    return integrate.simps(psi*np.conj(psi), xarr)
#print(norma)

for i in range(Nx): #initial parameters
    x = x0 + i*dx
    xarr[i] = x
    Psi[i,0] = Psi0(x)
    Um[i] = U(x)


Psi[:,0] = Psi[:,0]/np.sqrt(norma(Psi[:,0], xarr))
W[:,0] = Psi[:,0]*np.conj(Psi[:,0])

#first step - explicit scheme
Psi[1:Nx-1,1] = Psi[1:Nx-1,0] + dt/1j*Um[1:Nx-1]*Psi[1:Nx-1,0] + eta/1j*(Psi[2:,0] - 2*Psi[1:Nx-1,0] + Psi[:Nx-2,0])
Psi[:,1] = Psi[:,1]/np.sqrt(norma(Psi[:,1], xarr))
W[:,1] = Psi[:,1]*np.conj(Psi[:,1])
#print(Psi[:,1])

#next steps - Dufort-Frankel scheme
for i in range(1, T-1):
    Psi[1:Nx-1,i+1] = (Psi[1:Nx-1,i-1]*(Um[1:Nx-1]*dt/1j + eta/1j + 1)\
       - eta/1j*(Psi[2:,i] + Psi[:Nx-2,i]))/(1 - eta/1j - Um[1:Nx-1]*dt/1j)
    Psi[:,i+1] = Psi[:,i+1]/np.sqrt(norma(Psi[:,i+1], xarr))
    # print(norma)
#    print(Psi[:,i+1])
    W[:,i+1] = Psi[:,i+1]*np.conj(Psi[:,i+1])

# for t in range(T):
#     fig = plt.figure()
#     ax = plt.subplot()
#     ax.plot(xarr, Um, color='black', linewidth=1.0)
#     ax.plot(xarr, Psi[:,t])
#     plt.show()

fig = plt.figure()
ax = plt.subplot()
ax.plot(xarr, Um, color='black', linewidth=1.0)

def updatefig(i):
   Wt = W[:,i]
   ax.clear()
   ax.plot(xarr, Um, color='black', linewidth=1.0)
   line = ax.plot(xarr, Wt, color='blue', linewidth=1.0)
   return line,

ani = animation.FuncAnimation(fig, updatefig, np.arange(0, T), interval=10)

plt.show()