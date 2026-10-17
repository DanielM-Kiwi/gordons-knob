import math, os, urllib.request, numpy as np, cv2
from PIL import Image
S='work'; Z=13
LAT0,LON0=-41.60794,172.94411     # fitted camera position
HALF=7000.0; CELL=14.0; N=int(2*HALF/CELL)
def tx(lon): return (lon+180)/360*2**Z
def ty(lat): r=np.radians(lat); return (1-np.log(np.tan(r)+1/np.cos(r))/np.pi)/2*2**Z
mlat=111320.0; mlon=111320.0*math.cos(math.radians(LAT0))
x0,x1=int(tx(LON0-HALF/mlon)),int(tx(LON0+HALF/mlon)); y0,y1=int(ty(LAT0+HALF/mlat)),int(ty(LAT0-HALF/mlat))
mos=np.zeros(((y1-y0+1)*256,(x1-x0+1)*256),np.float32)
for x in range(x0,x1+1):
  for y in range(y0,y1+1):
    p=f"{S}/dem13/{x}_{y}.png"
    if not os.path.exists(p): urllib.request.urlretrieve(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png",p)
    a=np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
    mos[(y-y0)*256:(y-y0+1)*256,(x-x0)*256:(x-x0+1)*256]=a[...,0]*256+a[...,1]+a[...,2]/256-32768
e=-HALF+(np.arange(N)+0.5)*CELL; n=HALF-(np.arange(N)+0.5)*CELL
E,Nn=np.meshgrid(e,n)
px=((tx(LON0+E/mlon)-x0)*256-0.5).astype(np.float32); py=((ty(LAT0+Nn/mlat)-y0)*256-0.5).astype(np.float32)
dem=cv2.remap(mos,px,py,cv2.INTER_LINEAR)
np.save(S+'/localdem.npy',dem); print('dem',dem.shape,dem.min(),dem.max(),'centre',dem[N//2,N//2])
