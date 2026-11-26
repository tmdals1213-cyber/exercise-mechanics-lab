"""The original quintic target polynomial, without anatomical model imports."""
import numpy as np

class Polynomial:
    def __init__(self,p0,v0,a0,p1,v1,a1,T):
        p0=np.atleast_1d(p0);v0=np.broadcast_to(v0,p0.shape);a0=np.broadcast_to(a0,p0.shape);p1=np.broadcast_to(p1,p0.shape);v1=np.broadcast_to(v1,p0.shape);a1=np.broadcast_to(a1,p0.shape)
        self.c=np.zeros((6,len(p0)));self.c[:3]=[p0,v0,a0/2]
        A=np.array([[T**3,T**4,T**5],[3*T**2,4*T**3,5*T**4],[6*T,12*T**2,20*T**3]])
        self.c[3:]=np.linalg.solve(A,np.array([p1-p0-v0*T-a0*T*T/2,v1-v0-a0*T,a1-a0]))
    def value(self,t,derivative=0):
        c=self.c.copy()
        for _ in range(derivative):c=c[1:]*np.arange(1,len(c))[:,None]
        return np.array([t**k for k in range(len(c))])@c
