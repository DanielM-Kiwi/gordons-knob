import math, io, os, sys, urllib.request, numpy as np, base64, json
from PIL import Image
Z=11
LAT0,LON0=-41.6039,172.9369
HALF=60000.0; N=600   # 120 km square, 200 m cells
D=sys.argv[1]; os.makedirs(D, exist_ok=True)
def tx(lon): return (lon+180)/360*2**Z
def ty(lat):
    r=math.radians(lat); return (1-math.log(math.tan(r)+1/math.cos(r))/math.pi)/2*2**Z
mlat=111320.0; mlon=111320.0*math.cos(math.radians(LAT0))
lats=[LAT0+HALF/mlat, LAT0-HALF/mlat]; lons=[LON0-HALF/mlon, LON0+HALF/mlon]
x0,x1=int(tx(lons[0])),int(tx(lons[1])); y0,y1=int(ty(lats[0])),int(ty(lats[1]))
print(x0,x1,y0,y1,(x1-x0+1)*(y1-y0+1))
W=(x1-x0+1)*256; H=(y1-y0+1)*256
mos=np.zeros((H,W),np.float32)
for x in range(x0,x1+1):
  for y in range(y0,y1+1):
    p=f"{D}/{x}_{y}.png"
    if not os.path.exists(p):
      urllib.request.urlretrieve(f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{Z}/{x}/{y}.png",p)
    a=np.asarray(Image.open(p).convert('RGB')).astype(np.float32)
    e=a[...,0]*256+a[...,1]+a[...,2]/256-32768
    mos[(y-y0)*256:(y-y0+1)*256,(x-x0)*256:(x-x0+1)*256]=e
# sample grid: row 0 = north
out=np.zeros((N,N),np.float32)
cs=2*HALF/N
for j in range(N):
  n=HALF-(j+0.5)*cs; lat=LAT0+n/mlat
  py=(ty(lat)-y0)*256-0.5
  e_=-HALF+(np.arange(N)+0.5)*cs; lon=LON0+e_/mlon
  px=(tx(lon)-x0)*256-0.5
  ix=np.floor(px).astype(int); fx=px-ix; iy=int(math.floor(py)); fy=py-iy
  r0=mos[iy]; r1=mos[iy+1]
  v0=r0[ix]*(1-fx)+r0[ix+1]*fx; v1=r1[ix]*(1-fx)+r1[ix+1]*fx
  out[j]=v0*(1-fy)+v1*fy
out=np.clip(out,-50,3000)
print('center',out[N//2-2:N//2+2,N//2-2:N//2+2].max(),'max',out.max())
q=np.clip(np.round(out)+100,0,65535).astype('<u2')  # offset +100 so sea floor ok
open(sys.argv[2],'w').write(base64.b64encode(q.tobytes()).decode())
print(len(open(sys.argv[2]).read()))
