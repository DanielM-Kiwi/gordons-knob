import numpy as np, cv2, json
S='work'
a=cv2.imread(S+'/band_final.png').astype(np.float32); H,W=a.shape[:2]; ppd=W/360; TOP=20
r=cv2.GaussianBlur(a[...,2],(3,3),0); cm=np.maximum.accumulate(r,axis=0); land=(r<cm-18); land[:int(10*ppd)]=False
row=np.argmax(land,axis=0).astype(np.float32); row=cv2.medianBlur(row.reshape(1,-1),5).ravel()
band_el=TOP-(row+0.5)/ppd                      # skyline elevation in band coordinates, per band column
bb=(np.arange(W)+0.5)/ppd
Bt,skyT,_=np.load(S+'/sky_trig.npy')
def true_sky(t): return np.interp(t%360,Bt,skyT,period=360)
c1=np.array([-9.776,-2.146,0.609])
def prior(b): r=np.radians(b); return c1[0]+c1[1]*np.sin(r)+c1[2]*np.cos(r)
out=[]
for b0 in np.arange(0,360,3.0):
    m=np.abs(((bb-b0+180)%360)-180)<4.0
    xb=bb[m]; yb=band_el[m]
    if np.ptp(yb)<0.15: continue                # too flat to say anything
    best=None
    for s in np.arange(-6,6.01,0.05):
        yt=true_sky(xb+prior(xb)+s)
        A=np.stack([np.ones_like(yt),yt],1); cf,*_=np.linalg.lstsq(A,yb,rcond=None)
        if not (0.85<cf[1]<1.15): continue
        e=np.median(np.abs(A@cf-yb))
        if best is None or e<best[0]: best=(e,s,cf[0])
    if best is None: continue
    # sharpness: how much worse 1 deg away
    e0,s,off=best
    out.append((b0,s,off,e0,np.ptp(yb)))
out=np.array(out)
for o in out: print('band %5.1f  shift %+.2f  elev off %+.2f  err %.3f  relief %.2f'%tuple(o))
np.save(S+'/skymatch.npy',out)
