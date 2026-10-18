import cv2, numpy as np
S='work'
exec(open(S+'/geo2.py').read())
p2=cv2.imread('pano2.jpg'); PH,PW=p2.shape[:2]
band=cv2.imread(S+'/band_final_preseam.png').astype(np.float32); H,W=band.shape[:2]; ppd=W/360; TOP=20
sm=np.load(S+'/seam_map.npy')
def feats(x,y): x=x/1000; y=y/1000; return np.stack([np.ones_like(x),x,y,x*y,x*x,y*y,y**3],-1)
# band columns covered by the overlap (start copy x2 in [0, OV))
OVp=18704-C
allc=np.arange(W); xx,_=band_to_p2((allc+0.5)/ppd,np.zeros(W))
sel=(xx>=0)&(xx<OVp-4); cols=allc[sel][np.argsort(xx[sel])]; print('overlap band cols',cols.min()/ppd,cols.max()/ppd,len(cols))
B=(cols+0.5)/ppd; Eps=np.radians(TOP-(np.arange(H)+0.5)/ppd)
Bg,Eg=np.meshgrid(B,Eps); x2,y2=band_to_p2(Bg,Eg)
F0=feats(np.zeros_like(y2),y2); xe=x2+F0@sm[0]; ye=F0@sm[1]       # same loop-closure mapping the band uses at its edge
Sc=cv2.remap(p2,x2.astype(np.float32),y2.astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
Ec=cv2.remap(p2,xe.astype(np.float32),ye.astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
valid=(y2>=2)&(y2<PH-3)&(ye>=2)&(ye<PH-3)&(xe<PW-2)
# dense refinement: flow from start copy to end copy
gS=cv2.cvtColor(Sc.astype(np.uint8),cv2.COLOR_BGR2GRAY); gE=cv2.cvtColor(Ec.astype(np.uint8),cv2.COLOR_BGR2GRAY)
dis=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
flow=dis.calc(gS,gE,None); fm=np.hypot(flow[...,0],flow[...,1]); flow[fm>160]=0; flow=np.dstack([cv2.medianBlur(flow[...,k].astype(np.float32),5) for k in (0,1)])
flow=cv2.GaussianBlur(flow,(0,0),6)
print('flow magnitude median %.1f px, p95 %.1f px'%(np.median(np.hypot(*flow.transpose(2,0,1))),np.percentile(np.hypot(*flow.transpose(2,0,1)),95)))
gx,gy=np.meshgrid(np.arange(len(cols),dtype=np.float32),np.arange(H,dtype=np.float32))
# warp both halfway toward each other so the blend has no ghost and no net shift at the ends
t=np.linspace(0,1,len(cols))[None,:]; w=(t*t*(3-2*t)).astype(np.float32)                        # 0 at the left edge (end copy) .. 1 at the right (start copy)
Ew=cv2.remap(Ec,(gx+flow[...,0]*w).astype(np.float32),(gy+flow[...,1]*w).astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
Sw=cv2.remap(Sc,(gx-flow[...,0]*(1-w)).astype(np.float32),(gy-flow[...,1]*(1-w)).astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
# exposure: match the end copy to the start copy with a smooth per-row gain
gain=(cv2.GaussianBlur(Sw,(0,0),30)+2)/(cv2.GaussianBlur(Ew,(0,0),30)+2); gain=np.clip(gain,0.85,1.15)
Ew=Ew*(1+(gain-1)*w[...,None])
blend=Ew*(1-w[...,None])+Sw*w[...,None]
# only rows where pano2 is the source; keep what's there below (pano1 / fills), fade over 1 degree
cur=band[:,cols]
a=valid.astype(np.float32); a=cv2.erode(a,np.ones((int(ppd),1),np.uint8)); a=cv2.GaussianBlur(a,(0,0),ppd*0.4)
band[:,cols]=cur*(1-a[...,None])+blend*a[...,None]
# the band's old cut sits at the right edge of the overlap; make sure the columns either side match
cv2.imwrite(S+'/band_final.png',np.clip(band,0,255).astype(np.uint8))
c=cols[len(cols)//2]; crop=np.clip(band[:, np.r_[np.arange(cols[0]-150,cols[0]),cols,np.arange(cols[-1]+1,cols[-1]+150)]%W],0,255).astype(np.uint8)
m=cv2.GaussianBlur(crop.astype(np.float32),(0,0),25); cv2.imwrite(S+'/st/seam_hp2.jpg',cv2.resize(np.clip((crop-m)*4+128,0,255).astype(np.uint8),None,fx=0.5,fy=0.5))
cv2.imwrite(S+'/st/seam_rgb2.jpg',cv2.resize(crop,None,fx=0.5,fy=0.5))
cv2.imwrite(S+'/st/sc.jpg',np.vstack([cv2.resize(Sc.astype(np.uint8),None,fx=0.4,fy=0.4),cv2.resize(Ec.astype(np.uint8),None,fx=0.4,fy=0.4)]))
print('valid frac',valid.mean(), 'xe range',xe.min(),xe.max(),'x2 range',x2.min(),x2.max())
r0=int((TOP+3)*ppd); r1=int((TOP+8)*ppd)
fl=dis.calc(gS,gE,None)
print('near-ridge rows raw flow: median dx %.1f dy %.1f; abs>60 frac %.2f'%(np.median(fl[r0:r1,...,0]),np.median(fl[r0:r1,...,1]),(np.hypot(fl[r0:r1,...,0],fl[r0:r1,...,1])>60).mean()))
d0=np.abs(gS[r0:r1].astype(float)-gE[r0:r1].astype(float)).mean(); 
print('mean abs diff copies near ridge %.1f'%d0)
