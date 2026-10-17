import numpy as np, math, cv2
exec(open('tools/stitch/demsky.py').read().split("z0=elev")[0])
S='work'
e0,n0=500.,-400.
zc=elev(np.array([e0]),np.array([n0]))[0]+2
ppd=12288/360; TOP=36
B=np.arange(0,70,1/ppd)            # bearings of interest (coast NNE)
E=np.arange(-4,0.5,1/ppd)          # elevations
d=np.concatenate([np.arange(400,2000,10),np.geomspace(2000,90000,2500)])
sea=np.zeros((len(E),len(B)),np.uint8)
for j,b in enumerate(B):
    br=math.radians(b); z=elev(e0+np.sin(br)*d,n0+np.cos(br)*d)
    ang=np.degrees(np.arctan2(z-zc-d*d/(2*6371000)*0.87,d))
    # first hit for each elevation ray: first distance where terrain angle >= ray elevation
    for i,e in enumerate(E):
        hit=np.where(ang>=e)[0]
        if len(hit): sea[i,j]=1 if z[hit[0]]<=2 else 0
        else: sea[i,j]=2   # sky
np.save(S+'/demsea.npy',sea)
# photo: sea = smooth & bluish & low texture in the band
a=cv2.imread(S+'/band_final.png').astype(np.float32)
r0=int((TOP-0.5)*ppd); r1=r0+len(E); c1=len(B)
P=a[r0:r1,:c1]
g=cv2.cvtColor(P.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
tex=cv2.GaussianBlur(np.abs(cv2.Laplacian(g,cv2.CV_32F)),(0,0),3)
b_,g_,r_=P[...,0],P[...,1],P[...,2]
ph=((tex<2.2)&(b_>r_+25)).astype(np.uint8)
cv2.imwrite(S+'/st/coast_cmp.png',np.vstack([ (sea==1)*255, ph*255, cv2.cvtColor(P.astype(np.uint8),cv2.COLOR_BGR2GRAY)]).astype(np.uint8))
# horizontal shift search (and small vertical)
best=None
dm=(sea==1).astype(np.float32); valid=(sea!=2)
for dy in range(-12,13,2):
  for dx in range(-int(12*ppd),int(12*ppd)+1,2):
    s=np.roll(np.roll(dm,dx,1),dy,0); v=np.roll(np.roll(valid,dx,1),dy,0)
    m=v[:, int(12*ppd):-int(12*ppd)]
    A=s[:, int(12*ppd):-int(12*ppd)][m]; Pm=ph[:, int(12*ppd):-int(12*ppd)][m]
    sc=((A>0.5)&(Pm>0)).sum()/max(1,((A>0.5)|(Pm>0)).sum())
    if best is None or sc>best[0]: best=(sc,dx,dy)
print('best IoU %.3f  shift %.2f deg (DEM->photo, + = photo features are to the right)  dy %.2f deg'%(best[0],best[1]/ppd,best[2]/ppd))
