import numpy as np, math
from scipy.optimize import least_squares
exec(open('tools/stitch/fit2.py').read().split("def fitpos")[0].replace("from scipy.optimize import least_squares",""))
K=368.0
def model3(q,sky):
    t0,c,a,b,w1,w2,w3,w4=q
    brg=t0+360*x/C+w1*np.sin(u)+w2*np.cos(u)+w3*np.sin(2*u)+w4*np.cos(2*u)
    el=np.interp(brg%360,B,sky,period=360)
    return c+K*x/C+a*np.sin(u)+b*np.cos(u)-f*np.tan(el)
def fitpos(e0,n0):
    sky=skyfrom(e0,n0); best=None
    for t0 in np.arange(335,340.01,0.1):
        el=np.interp((t0+360*x/C)%360,B,sky,period=360)
        A=np.stack([np.ones(n),np.sin(u),np.cos(u)],1); p,*_=np.linalg.lstsq(A,ys-K*x/C+f*np.tan(el),rcond=None)
        e=np.median(np.abs(A@p+K*x/C-f*np.tan(el)-ys))
        if best is None or e<best[0]: best=(e,t0,p)
    q0=np.array([best[1],*best[2],0,0,0,0])
    lo=[max(q0[0]-1.5,334.5),-1e4,-1e4,-1e4,-3,-3,-2,-2]; hi=[min(q0[0]+1.5,340.5),1e4,1e4,1e4,3,3,2,2]
    sol=least_squares(lambda q: model3(q,sky)-ys,q0,bounds=(lo,hi),loss='soft_l1',f_scale=6,diff_step=1e-3)
    rr=model3(sol.x,sky)-ys
    return np.median(np.abs(rr)),np.percentile(np.abs(rr),90),sol.x
out=[]
for e0 in np.arange(-800,1401,200):
  for n0 in np.arange(-1400,801,200):
    m,p90,q=fitpos(e0,n0); out.append((m,p90,e0,n0,q))
out.sort(key=lambda t:t[0]+0.3*t[1])
for o in out[:6]: print('med %.1f p90 %.1f E %d N %d'%o[:4], np.round(o[4],2))
np.save(S+'/fit4.npy',np.array([out[0][2],out[0][3],*out[0][4]]))
