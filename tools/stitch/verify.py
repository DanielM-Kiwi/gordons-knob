import cv2, numpy as np, math
from scipy.ndimage import median_filter
S='work'; ppd=12288/360; TOP=36
def colratio(g,r0,r1,wrap=True):
    x=g[r0:r1]; d=np.abs(np.diff(np.concatenate([x,x[:,:1]],1) if wrap else x,axis=1)).mean(0)
    return d/np.maximum(median_filter(d,61,mode='wrap' if wrap else 'nearest'),1e-3)
# 1) baseline: same metric on raw pano2 (resampled to the band's px/deg so scales match)
p2=cv2.cvtColor(cv2.imread('pano2.jpg'),cv2.COLOR_BGR2GRAY).astype(np.float32)
p2s=cv2.resize(p2,None,fx=ppd/(18032/360),fy=ppd/(18032/360),interpolation=cv2.INTER_AREA)
raw=colratio(p2s,int(p2s.shape[0]*0.35),p2s.shape[0]-5,wrap=False)
fin=cv2.cvtColor(cv2.imread(S+'/band_final.png'),cv2.COLOR_BGR2GRAY).astype(np.float32)
f_=colratio(fin,int((TOP-5)*ppd),int((TOP+30)*ppd))
print('1) column-step ratio  raw unstitched pano2: max %.2f p99.9 %.2f | final stitched: max %.2f p99.9 %.2f'%(raw[30:-30].max(),np.percentile(raw[30:-30],99.9),f_.max(),np.percentile(f_,99.9)))
# 2) steepness of every blend mask in the final composite (alpha of real photo layers)
al=np.load(S+'/alpha.npy').astype(np.float32)
gy=np.abs(np.diff(al,axis=0)).max()*ppd; gx=np.abs(np.diff(np.concatenate([al,al[:,:1]],1),axis=1)).max()*ppd
print('2) photo-layer mask: steepest change %.2f per degree vertically, %.2f per degree horizontally -> narrowest transition %.2f deg'%(gy,gx,1/max(gy,gx)))
# 3) crew photo alignment: features in the inserted region vs the pano underneath
a=cv2.imread(S+'/band_base.png',0); b=cv2.imread(S+'/band_final.png',0)
c0,c1=int(171.5*ppd),int(197.7*ppd); r0,r1=int((TOP-12)*ppd),int((TOP+7)*ppd)
A=a[r0:r1,c0:c1]; B=b[r0:r1,c0:c1]
sift=cv2.SIFT_create(20000); ka,da=sift.detectAndCompute(A,None); kb,db=sift.detectAndCompute(B,None)
m=cv2.BFMatcher().knnMatch(db,da,k=2); g=[x for x,y in m if x.distance<0.7*y.distance]
d=np.float32([np.subtract(kb[x.queryIdx].pt,ka[x.trainIdx].pt) for x in g]); e=np.hypot(d[:,0],d[:,1])
e=e[e<20]
print('3) summit-crew photo vs pano underneath: %d features, displacement median %.2f px (%.3f deg), p90 %.2f px'%(len(e),np.median(e),np.median(e)/ppd,np.percentile(e,90)))
# 4) how much of what the viewer can show is synthetic ground, per lowest-view limit
rowel=TOP-(np.arange(al.shape[0])+0.5)/ppd
for L in (-50,-40,-35,-32,-30):
    rows=(rowel>=L)&(rowel<0)
    syn=(1-al[rows])
    view=(rowel>=L)&(rowel<=35)
    print('4) lowest view %d deg: synthetic share of the viewable sphere %.2f%%, of the viewable ground %.1f%%; worst bearing %.1f%%'%(L,100*syn.sum()/(view.sum()*al.shape[1]),100*syn.mean(),100*syn.mean(0).max()))
