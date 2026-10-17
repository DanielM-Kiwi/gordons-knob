import cv2, numpy as np
S='work'
a=cv2.imread(S+'/band_final.png').astype(np.float32); H,W=a.shape[:2]; ppd=W/360; TOP=20
SEAM=int(round(348.90*ppd)); RAMP=int(20*ppd)
# 1) per-row colour gain so the left (end of sweep) matches the right (start) at the join
Lm=a[:,SEAM-40:SEAM-4].mean(1); Rm=a[:,SEAM+4:SEAM+40].mean(1)
from scipy.ndimage import gaussian_filter1d
g=(gaussian_filter1d(Rm,40,axis=0)+2)/(gaussian_filter1d(Lm,40,axis=0)+2)
g=np.clip(g,0.85,1.15)
cols=np.arange(SEAM-RAMP,SEAM); t=(cols-(SEAM-RAMP))/RAMP; w=t*t*(3-2*t)
a[:,cols]*=1+(g[:,None,:]-1)*w[None,:,None]
# 2) far skyline rows: vertical nudge so the ridgeline meets at the join
r=cv2.GaussianBlur(a[...,2],(3,3),0); cm=np.maximum.accumulate(r,axis=0); land=(r<cm-18); land[:int(10*ppd)]=False
sky=np.argmax(land,axis=0).astype(np.float32)
dy=np.median(sky[SEAM+3:SEAM+40])-np.median(sky[SEAM-40:SEAM-3])
print('gain range %.3f..%.3f  skyline step %.1f px'%(g.min(),g.max(),dy))
y0=np.median(sky[SEAM-60:SEAM+60])
rows=np.arange(H,dtype=np.float32)
fade=np.clip(1-(rows-y0)/(3*ppd),0,1)            # full above/at the skyline, gone 3 deg below
mx=np.tile(np.arange(W,dtype=np.float32),(H,1)); my=np.tile(rows[:,None],(1,W))
sub=slice(SEAM-RAMP,SEAM)
my[:,sub]-=dy*fade[:,None]*w[None,:]
a=cv2.remap(a,mx,my,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
cv2.imwrite(S+'/band_final.png',np.clip(a,0,255).astype(np.uint8))
c=np.clip(a[int((TOP-14)*ppd):int((TOP+4)*ppd), SEAM-500:SEAM+500],0,255).astype(np.uint8); cv2.imwrite(S+'/st/wrap2.jpg',c)
# 3) the join column itself: interpolate across a few pixels
a=cv2.imread(S+'/band_final.png').astype(np.float32)
L=a[:,SEAM-4]; R=a[:,SEAM+4]
for k,cx in enumerate(range(SEAM-3,SEAM+4)): t=(k+1)/8; a[:,cx]=L*(1-t)+R*t
cv2.imwrite(S+'/band_final.png',np.clip(a,0,255).astype(np.uint8))
g2=a.mean(2); r=int((TOP-12)*ppd); st=np.abs(np.diff(g2[r-50:r+50, SEAM-30:SEAM+30],axis=1)).mean(0)
print('max column step near join now %.2f (sky noise ~%.2f)'%(st.max(),np.median(st)))
