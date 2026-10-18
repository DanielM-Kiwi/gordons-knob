import numpy as np, cv2, json, math
S='work'
d14=np.load(S+'/localdem.npy'); HALF=7000.0; CELL=14.0
TE,TN=59.0,-213.0                         # trig, fitted-origin frame
cb=json.load(open(S+'/calib_b.json')); CB=np.array(cb['bearing']); CEl=np.array(cb['elev'])
def four(c,b):
    r=np.radians(b); v=c[0]+0*r
    for k in range(1,(len(c)-1)//2+1): v=v+c[2*k-1]*np.sin(k*r)+c[2*k]*np.cos(k*r)
    return v
def true2band(az,el):
    b=az-CB[0]
    for _ in range(6): b=az-four(CB,b)
    b%=360; return b, el+four(CEl,b)
band=cv2.imread(S+'/band_final.png',0).astype(np.float32); BH,BW=band.shape; ppd=BW/360; TOP=20
RES=4.0                                    # grid px per degree
AZ=np.arange(0,360,1/RES); EL=np.arange(2,-26,-1/RES)
A,EG=np.meshgrid(AZ,EL)
bb,be=true2band(A,EG)
mx=(bb*ppd).astype(np.float32); my=((TOP-be)*ppd).astype(np.float32)
photo=cv2.remap(cv2.GaussianBlur(band,(0,0),2.5),mx,my,cv2.INTER_AREA)
ed=cv2.Canny(cv2.GaussianBlur(photo,(3,3),0).astype(np.uint8),18,45)
dt=cv2.distanceTransform((ed==0).astype(np.uint8),cv2.DIST_L2,3)
dt=np.minimum(dt,8)
dist=np.concatenate([np.arange(20,600,4),np.arange(600,3000,10)])
def render(ce,cn,dz):
    zc=cv2.remap(d14,np.float32([[(ce+HALF)/CELL-0.5]]),np.float32([[(HALF-cn)/CELL-0.5]]),cv2.INTER_LINEAR)[0,0]+dz
    br=np.radians(AZ)
    E=ce+np.sin(br)[:,None]*dist; N=cn+np.cos(br)[:,None]*dist
    z=cv2.remap(d14,((E+HALF)/CELL-0.5).astype(np.float32),((HALF-N)/CELL-0.5).astype(np.float32),cv2.INTER_LINEAR)
    ang=np.degrees(np.arctan2(z-zc,dist)); run=np.maximum.accumulate(ang,axis=1)
    dep=np.zeros((len(EL),len(AZ)),np.float32)
    for j in range(len(AZ)):
        idx=np.searchsorted(run[j],EL[::-1])[::-1]; dep[:,j]=np.log(np.where(idx<len(dist),dist[np.minimum(idx,len(dist)-1)],5000))
    sil=(np.abs(np.diff(dep,axis=0,prepend=dep[:1]))>0.35)&(np.exp(dep)<2500)
    return sil
def score(ce,cn,dz):
    sil=render(ce,cn,dz); n=sil.sum()
    return (dt[sil].mean() if n>200 else 99), n
best=[]
for ce in np.arange(TE-90,TE+91,15):
  for cn in np.arange(TN-90,TN+91,15):
    for dz in (0.5,2,4,7):
      sc,n=score(ce,cn,dz); best.append((sc,ce,cn,dz,n))
best.sort(); print('coarse:',[tuple(np.round(b,1)) for b in best[:5]])
_,e0,n0,z0,_=best[0]; fine=[]
for ce in np.arange(e0-12,e0+13,3):
  for cn in np.arange(n0-12,n0+13,3):
    for dz in np.arange(max(0,z0-2),z0+2.1,0.5):
      sc,n=score(ce,cn,dz); fine.append((sc,ce,cn,dz,n))
fine.sort(); print('fine:',[tuple(np.round(b,2)) for b in fine[:5]])
sc,ce,cn,dz,_=fine[0]
print('camera vs trig: E %+.0f m N %+.0f m, eye %.1f m above model ground; previous estimate E +15.5 N -7.4'%(ce-TE,cn-TN,dz))
json.dump({'cam_en_fitted':[ce,cn],'dz':dz,'vs_trig':[ce-TE,cn-TN]},open(S+'/campos.json','w'))
sil=render(ce,cn,dz); vis=cv2.cvtColor(photo.astype(np.uint8),cv2.COLOR_GRAY2BGR); vis[ed>0]=(0,255,255); vis[sil]=(0,0,255)
cv2.imwrite(S+'/st/sil.png',cv2.resize(vis,None,fx=1.5,fy=1.5,interpolation=cv2.INTER_NEAREST))
