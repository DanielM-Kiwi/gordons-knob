import numpy as np, math
S='work'
C=18032.0; f=C/(2*math.pi); K=368.0
_G=dict(zip('e0 n0 t0 c a b w1 w2 w3 w4'.split(),np.load(S+'/fit3.npy')))
def brg_of_x(x,G=_G):
    u=2*np.pi*x/C; return G['t0']+360*x/C+G['w1']*np.sin(u)+G['w2']*np.cos(u)+G['w3']*np.sin(2*u)+G['w4']*np.cos(2*u)
def hor_of_x(x,G=_G):
    u=2*np.pi*x/C; return G['c']+K*x/C+G['a']*np.sin(u)+G['b']*np.cos(u)
def pano_to_dir(x,y):   # strip px -> bearing deg, elev rad
    return brg_of_x(x)%360, np.arctan((hor_of_x(x)-y)/f)
def x_of_brg(B):        # inverse (fixed point), returns x in [0,C)
    x=((B-_G['t0'])%360)/360*C
    for _ in range(6): x=x-((brg_of_x(x)-B+180)%360-180)/360*C
    return x%C
def dir_vec(B,el):  # E,N,Up
    B=np.radians(B); return np.stack([np.sin(B)*np.cos(el),np.cos(B)*np.cos(el),np.sin(el)],-1)
