import math, os, urllib.request, numpy as np
from PIL import Image
S='work'
Z=10; LAT0,LON0=-41.6039,172.9369; HALF=160000.0
def tx(lon): return (lon+180)/360*2**Z
def ty(lat):
    r=np.radians(lat); return (1-np.log(np.tan(r)+1/np.cos(r))/np.pi)/2*2**Z
mlat=111320.0; mlon=111320.0*math.cos(math.radians(LAT0))
x0,x1=int(tx(LON0-HALF/mlon)),int(tx(LON0+HALF/mlon)); y0,y1=int(ty(LAT0+HALF/mlat)),int(ty(LAT0-HALF/mlat))
W=(x1-x0+1)*256; H=(y1-y0+1)*256; mos=np.zeros((H,W),np.float32)
for x in range(x0,x1+1):
  for y in range(y0,y1+1):
    p=f"{S}/dem9/{x}_{y}.png"
    if not os.path.exists(p): urllib.request.urlretrieve(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png",p)
    a=np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
    mos[(y-y0)*256:(y-y0+1)*256,(x-x0)*256:(x-x0+1)*256]=a[...,0]*256+a[...,1]+a[...,2]/256-32768
mos=np.clip(mos,0,3000)
import cv2; mos=np.where(mos-cv2.medianBlur(mos,5)>300,cv2.medianBlur(mos,5),mos)
def elev(e,n):
    lat=LAT0+n/mlat; lon=LON0+e/mlon
    px=(tx(lon)-x0)*256-0.5; py=(ty(lat)-y0)*256-0.5
    ix=np.clip(px.astype(int),0,W-2); iy=np.clip(py.astype(int),0,H-2); fx=px-ix; fy=py-iy
    return (mos[iy,ix]*(1-fx)+mos[iy,ix+1]*fx)*(1-fy)+(mos[iy+1,ix]*(1-fx)+mos[iy+1,ix+1]*fx)*fy
z0=elev(np.array([0.]),np.array([0.]))[0]
print('z0 dem',z0)
z0=max(z0,1592)+6
brg=np.arange(0,360,0.05)*math.pi/180
d=np.concatenate([np.arange(30,2000,10),np.geomspace(2000,155000,1500)])
E=np.sin(brg)[:,None]*d[None]; N=np.cos(brg)[:,None]*d[None]
z=elev(E,N)
ang=np.arctan2(z-z0-d*d/(2*6371000)*0.87,d)
sky=ang.max(1); arg=d[ang.argmax(1)]
np.save(S+'/demsky.npy',np.stack([sky,arg]))
print('sky deg range',np.degrees(sky).min(),np.degrees(sky).max())
