import cv2, numpy as np, math, gc
S='work'
exec(open(S+'/geo.py').read())
R=''  # run from the repo root
W=12288; TOP=36.0; BOT=-54.0; ppd=W/360.0; Hh=int(round((TOP-BOT)*ppd)); OV=18704-C
p2=cv2.imread(R+'pano2.jpg'); p1=cv2.imread(R+'pano1.jpg'); pm=cv2.imread(R+'looking at mot.jpg')
P2H=p2.shape[0]; P1H,P1W=p1.shape[:2]; MH,MW=pm.shape[:2]
m1=np.load(S+'/p1model.npz'); qm=np.load(S+'/rot_mot.npy'); fm,cxm,cym=qm[:3]; Rmot=qm[3:].reshape(3,3)
beta=(np.arange(W)+0.5)/ppd
epsd=(TOP-(np.arange(Hh)+0.5)/ppd).astype(np.float32); eps=np.radians(epsd); T=np.tan(eps).astype(np.float32)
rowel=epsd[:,None]
xs=x_of_brg(beta); hor=hor_of_x(xs)
EL_TOP=np.degrees(np.arctan((hor-2)/f)).min(); EL_BOT2=np.degrees(np.arctan((hor-(P2H-3))/f)).max()
print('p2 safe band %.2f .. %.2f'%(EL_TOP,EL_BOT2),flush=True)
def ramp(d,w): t=np.clip(d/w,0,1); return t*t*(3-2*t)
FE=1.2
# ---------- layer samplers (per column chunk) ----------
def L2(sl):
    x=xs[sl]
    def one(xc,hc):
        mx=np.broadcast_to(xc[None].astype(np.float32),(Hh,len(xc))).copy(); my=(hc[None].astype(np.float32)-f*T[:,None])
        o=cv2.remap(p2,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        a=ramp(np.minimum(my,P2H-1-my),FE*ppd*f/2870)*(mx<=p2.shape[1]-1)
        return o,a.astype(np.float32)
    A,aa=one(x,hor[sl])
    if (x<OV).any():
        B,ab=one(x+C,hor_of_x(x+C)); w=ramp(x,OV)
        wa=np.where(x<OV,w,1)[None]*aa; wb=np.where(x<OV,1-w,0)[None]*ab
        A=(A*wa[...,None]+B*wb[...,None])/np.maximum(wa+wb,1e-6)[...,None]; aa=np.maximum(aa*np.where(x<OV,1,1)[None],ab*(x<OV)[None])
    return A,aa
def p1_xy(b,e):
    bx,bb,ss,xh,hh,f1=m1['bx'],m1['bb'],m1['ss'],m1['x'],m1['h'],float(m1['f1'])
    bu=np.where(b<bb[0],b+360,b)
    x=np.interp(bu,bb,bx)
    for _ in range(5):
        y=np.interp(x,xh,hh)-f1*np.tan(e)
        bp=np.interp(x,bx,bb)+np.interp(x,bx,ss)*(y-1400)
        x=x-(bp-bu)*(P1W/207.2)
    y=np.interp(x,xh,hh)-f1*np.tan(e)
    inr=ramp(bu-bb[0],8.0)*ramp(bb[-1]-bu,8.0)
    return x,y,inr
def L1(sl):
    Bg=np.broadcast_to(beta[sl][None],(Hh,sl.stop-sl.start)); Eg=np.broadcast_to(eps[:,None],Bg.shape)
    x,y,inr=p1_xy(Bg,Eg)
    o=cv2.remap(p1,x.astype(np.float32),y.astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    a=ramp(np.minimum(y,P1H-1-y),2.5*48.5)*ramp(np.minimum(x,P1W-1-x),10*48.5)*inr*(Eg<0)
    return o,a.astype(np.float32)
def LM(sl):
    B=np.radians(beta[sl]); Bg,Eg=np.meshgrid(B,eps)
    d=np.stack([np.sin(Bg)*np.cos(Eg),np.cos(Bg)*np.cos(Eg),np.sin(Eg)],-1)
    cam=d@Rmot; z=-cam[...,2]; zz=np.where(z>1e-3,z,1e-3)
    mx=(cxm+fm*cam[...,0]/zz).astype(np.float32); my=(cym-fm*cam[...,1]/zz).astype(np.float32)
    o=cv2.remap(pm,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    a=ramp(np.minimum(np.minimum(mx,MW-1-mx)/MW,np.minimum(my,MH-1-my)/MH),0.12)*(z>0)*(Eg<0)
    return o,a.astype(np.float32)
CH=768
chunks=[slice(c,min(W,c+CH)) for c in range(0,W,CH)]
# ---------- pass 1: per-column means in overlap band for gain matching ----------
band=(rowel<EL_BOT2+5)&(rowel>EL_BOT2+1.5)   # rows where p2 is solid and below horizon
sum2=np.zeros((W,3));sum1=np.zeros((W,3));n1=np.zeros(W);summ=np.zeros((W,3));nm=np.zeros(W);sum2m=np.zeros((W,3));sum2_1=np.zeros((W,3))
for sl in chunks:
    A,aa=L2(sl); O1,a1=L1(sl); OM,am=LM(sl)
    bm=np.broadcast_to(band,aa.shape)
    w1=(bm&(a1>0.99)).astype(np.float32); wm=(bm&(am>0.99)).astype(np.float32)
    sum2_1[sl]=(A*w1[...,None]).sum(0); sum1[sl]=(O1*w1[...,None]).sum(0); n1[sl]=w1.sum(0)
    sum2m[sl]=(A*wm[...,None]).sum(0); summ[sl]=(OM*wm[...,None]).sum(0); nm[sl]=wm.sum(0)
def gainfrom(sa,sb,n,sig=ppd*3):
    pad=int(sig*4)
    def wb(a): ap=np.concatenate([a[-pad:],a,a[:pad]]); return cv2.GaussianBlur(ap.astype(np.float32).reshape(1,-1,a.shape[1] if a.ndim>1 else 1),(0,0),sig).reshape(ap.shape)[pad:-pad]
    A_=wb(sa); B_=wb(sb); N_=wb(n[:,None].repeat(3,1))
    g=(A_/np.maximum(N_,1e-3)+1)/(B_/np.maximum(N_,1e-3)+1)
    g=np.where(N_>0.5,g,1.0)
    return np.clip(g,0.7,1.4).astype(np.float32), N_[:,0]
g1,N1=gainfrom(sum2_1,sum1,n1); gm,Nm=gainfrom(sum2m,summ,nm)
print('gain p1 median',np.median(g1[N1>0.5],0),'gain mot median',np.median(gm[Nm>0.5],0),flush=True)
# ---------- pass 2: composite ----------
img=np.zeros((Hh,W,3),np.uint8); alpha=np.zeros((Hh,W),np.float32)
for sl in chunks:
    A,aa=L2(sl); O1,a1=L1(sl); OM,am=LM(sl)
    O1*=g1[sl][None]; OM*=gm[sl][None]
    lower=O1*a1[...,None]+OM*((1-a1)*am)[...,None]; al=a1+(1-a1)*am
    col=A*aa[...,None]+lower*(1-aa)[...,None]; at=aa+(1-aa)*al
    img[:,sl]=np.clip(col/np.maximum(at,1e-4)[...,None],0,255).astype(np.uint8); alpha[:,sl]=at
del A,O1,OM; gc.collect()
print('composited; real-photo coverage lowest elevation per 30deg:',[round(float(epsd[np.where(alpha[:,c]>0.5)[0].max()]),1) for c in range(0,W,W//12)],flush=True)
np.save(S+'/alpha.npy',alpha.astype(np.float16)); cv2.imwrite(S+'/real.png',img)
