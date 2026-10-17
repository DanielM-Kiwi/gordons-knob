import numpy as np, math
from scipy.optimize import least_squares
exec(open('tools/stitch/demsky.py').read().split("z0=elev")[0])
S='work'
C=18032.0; f=C/(2*math.pi)
ys=np.load(S+'/skyline_px.npy'); ys=ys.reshape(-1,2).mean(1); n=len(ys); x=(np.arange(n)+0.5)*C/n
B=np.arange(0,360,0.05); br=np.radians(B)
d=np.concatenate([np.arange(400,2000,10),np.geomspace(2000,155000,1200)])
def skyfrom(e0,n0,eye=2):
    zc=elev(np.array([e0]),np.array([n0]))[0]+eye
    E=e0+np.sin(br)[:,None]*d; N=n0+np.cos(br)[:,None]*d
    return np.arctan2(elev(E,N)-zc-d*d/(2*6371000)*0.87,d).max(1)
u=2*np.pi*x/C
def model(q,sky):
    t0,c,k,a,b,w1,w2,w3,w4=q[:9]
    brg=t0+360*x/C+w1*np.sin(u)+w2*np.cos(u)+w3*np.sin(2*u)+w4*np.cos(2*u)
    el=np.interp(brg%360,B,sky,period=360)
    yp=c+k*x/C+a*np.sin(u)+b*np.cos(u)-f*np.tan(el)
    return yp
def fitpos(e0,n0,verbose=False):
    sky=skyfrom(e0,n0)
    best=None
    for t0 in np.arange(0,360,0.5):
        q0=np.array([t0,700,370,0,86,0,0,0,0.])
        r=lambda q: model(q,sky)-ys
        # quick linear solve for vertical params
        el=np.interp((t0+360*x/C)%360,B,sky,period=360)
        A=np.stack([np.ones(n),x/C,np.sin(u),np.cos(u)],1); p,*_=np.linalg.lstsq(A,ys+f*np.tan(el),rcond=None)
        e=np.sqrt(np.mean((A@p-f*np.tan(el)-ys)**2))
        if best is None or e<best[0]: best=(e,t0,p)
    q0=np.array([best[1],*best[2],0,0,0,0])
    sol=least_squares(lambda q: model(q,sky)-ys,q0,loss='soft_l1',f_scale=8,diff_step=1e-3)
    rr=model(sol.x,sky)-ys; rms=np.sqrt(np.mean(np.clip(rr,-60,60)**2)); med=np.median(np.abs(rr))
    return med,rms,sol.x
out=[]
for e0 in np.arange(200,801,100):
  for n0 in np.arange(-800,1,100):
    med,rms,q=fitpos(e0,n0); out.append((med,rms,e0,n0,q)); 
out.sort(key=lambda t:t[0])
for o in out[:6]: print('med %.1f rms %.1f E %d N %d'%o[:4], np.round(o[4],2))
np.save(S+'/fit2.npy',np.array([out[0][2],out[0][3],*out[0][4]]))
