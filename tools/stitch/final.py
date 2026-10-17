import cv2, numpy as np, math, gc
exec(open('tools/stitch/geo.py').read())
R=''
W=18032; TOP=20.0; BOT=-27.0
ppd=W/360.0; Hh=int(round((TOP-BOT)*ppd))
OV=18704-C
p2=cv2.imread(R+'pano2.jpg'); PH,PW=p2.shape[:2]
beta=(np.arange(W)+0.5)/ppd
eps=np.radians(TOP-(np.arange(Hh)+0.5)/ppd); T=np.tan(eps).astype(np.float32)
rowel=np.degrees(eps).astype(np.float32)[:,None]
xs=x_of_brg(beta); hor=hor_of_x(xs)
el_top=np.degrees(np.arctan((hor-2)/f)); el_bot=np.degrees(np.arctan((hor-(PH-3))/f))
el_top=np.full_like(el_top,el_top.min()); el_bot=np.full_like(el_bot,el_bot.max())
print('photo covers el top %.1f..%.1f  bottom %.1f..%.1f'%(el_top.min(),el_top.max(),el_bot.min(),el_bot.max()),flush=True)
img=cv2.imread('work/real.png'); ALPHA=np.load('work/alpha.npy').astype(np.float32)
CH=1024
def wrapblur(a,sig):
    pad=int(sig*4); ap=np.concatenate([a[-pad:],a,a[:pad]]); return cv2.GaussianBlur(ap[None].astype(np.float32),(0,0),sig)[0][pad:-pad]
# ---- sky ----
bd=2.0
def bandmean(lo,hi):
    out=np.zeros((W,3),np.float32)
    for xcol in range(W):
        r0=int((TOP-hi[xcol])*ppd); r1=int((TOP-lo[xcol])*ppd); out[xcol]=img[r0:r1,xcol].reshape(-1,3).mean(0)
    return out
col_top=wrapblur(bandmean(el_top-bd-2.5,el_top-bd),ppd*4)
col_lo=wrapblur(bandmean(el_top-bd-6.5,el_top-bd-4),ppd*4)
grad=(col_top-col_lo)/4.0
Z=np.array([178.,92.,38.],np.float32)
for c0 in range(0,W,CH):
    sl=slice(c0,min(W,c0+CH)); et=el_top[sl][None,:]
    e0_=et-bd-1.25
    c0c=col_top[sl][None]; gr=grad[sl][None]
    k=np.clip(gr/np.minimum(Z-c0c,-1e-3*np.ones_like(c0c))*-1,0.005,0.2)  # per-channel decay so slope matches at e0
    k=np.where((Z-c0c)*gr>0, gr/(Z-c0c), 0.02)
    k=np.clip(k,0.008,0.15)
    de=np.maximum(rowel-e0_,0)[...,None]
    sky=Z+(c0c-Z)*np.exp(-de*k)
    ws=np.clip((rowel-(et-bd))/bd,0,1); ws=ws*ws*(3-2*ws)
    noise=np.random.default_rng(c0).normal(0,0.6,sky.shape).astype(np.float32)
    blk=img[:,sl].astype(np.float32)
    img[:,sl]=np.clip(blk*(1-ws[...,None])+(sky+noise)*ws[...,None],0,255).astype(np.uint8)
print('sky done',flush=True)
# ---- ground quilting ----
gb=2.5
rng=np.random.default_rng(7)
P=int(ppd*3.2); step=P//3
yy,xx=np.mgrid[:P,:P]; g=np.exp(-(((yy-P/2)/(P/3))**2+((xx-P/2)/(P/3))**2)).astype(np.float32)
rstart=int((TOP-(el_bot.max()+gb+3))*ppd)-P
G0=max(0,rstart); Hg=Hh-G0
acc=np.zeros((Hg,W,3),np.float32); wacc=np.zeros((Hg,W),np.float32)
src=img
b0=int((TOP-(el_bot[0]+2.0))*ppd); b1=int((TOP-el_bot[0])*ppd)
GREF=np.median(img[b0:b1].reshape(-1,3).astype(np.float32),0); print('ground ref',GREF)
for ry in range(G0,Hh,step):
    el_here=TOP-(ry+P/2)/ppd
    for rx in range(-P//2,W,step):
        cx=(rx+P//2)%W
        if el_here>el_bot[cx]+gb+3: continue
        best=None
        for _t in range(10):
            sx=int((cx+rng.integers(-int(ppd*(6+_t*2)),int(ppd*(6+_t*2))))%W)
            sb=el_bot[sx]+0.4; st=sb+4.5
            sy0=int((TOP-st)*ppd); sy1=int((TOP-sb)*ppd)-P
            if sy1<=sy0: continue
            sy=rng.integers(sy0,sy1)
            cols=(np.arange(sx-P//2,sx-P//2+P))%W
            patch=src[sy:sy+P][:,cols].astype(np.float32)
            d=np.abs(patch.reshape(-1,3).mean(0)-GREF).sum()
            if best is None or d<best[0]: best=(d,patch)
            if d<25: break
        if best is None: continue
        patch=best[1]
        if rng.random()<0.5: patch=patch[:,::-1]
        dcols=(np.arange(rx,rx+P))%W
        y1=min(Hh,ry+P); n_=y1-ry
        acc[ry-G0:y1-G0][:,dcols]+=patch[:n_]*g[:n_,:,None]; wacc[ry-G0:y1-G0][:,dcols]+=g[:n_]
acc/=np.maximum(wacc,1e-6)[...,None]; del wacc; gc.collect()
print('quilted',flush=True)
ground=acc
mean=cv2.GaussianBlur(ground,(0,0),ppd*1.5); ground+= (ground-mean)*0.6; del mean
re=rowel[G0:]
deep=np.clip((el_bot[None,:]-6-re)/18,0,1)
sel=(rowel<el_bot[None,:]+5)&(rowel>el_bot[None,:]+1)
gmean=img[sel].astype(np.float32).mean(0)
bl=cv2.GaussianBlur(ground,(0,0),ppd*0.8)
ground=ground*(1-deep[...,None])+(bl*0.6+gmean*0.4)*deep[...,None]; del bl
ground*=(1-0.25*deep)[...,None]
wg=(1-ALPHA[G0:])*(re<0)
reg=img[G0:].astype(np.float32)
img[G0:]=np.clip(reg*(1-wg[...,None])+ground*wg[...,None],0,255).astype(np.uint8)
cv2.imwrite('work/band_base.png',img)
np.save('work/band_meta.npy',np.array([W,Hh,TOP,BOT]))
print('band',W,Hh)
