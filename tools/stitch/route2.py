import numpy as np, json, math, cv2
from skimage.graph import route_through_array
from scipy.ndimage import gaussian_filter, distance_transform_edt, gaussian_filter1d
S='work'
d14=np.load(S+'/localdem.npy'); HALF=7000.0; UP=2; CELL=14.0/UP
D=cv2.resize(d14,None,fx=UP,fy=UP,interpolation=cv2.INTER_CUBIC); N=D.shape[0]
def en2rc(e,n): return (HALF-n)/CELL-0.5, (e+HALF)/CELL-0.5
def rc2en(r,c): return (c+0.5)*CELL-HALF, HALF-(r+0.5)*CELL
# user's line in the fitted frame (from the topo map georeference), plus its map-pixel vertices
g=json.load(open(S+'/georef.json')); U=np.array(g['route_en'])
userpx=[(370,5),(445,45),(460,90),(470,130),(480,170),(520,200),(560,230),(595,270),(620,310),(645,350),(660,380),(690,410),(720,428),(750,432),(770,440),(785,470),(790,520),(795,560),(800,575),(800,610),(775,625),(750,640),(735,660),(728,690),(715,720),(695,760),(670,800),(645,840),(625,875),(600,900),(570,915),(540,930),(510,945),(470,950),(440,960),(425,975),(422,1000),(435,1020),(455,1048)]
TE,TN=59.0,-213.0; CE,CN=TE+15.5,TN-7.4
U[-1]=[TE,TN]
# ridge-ness: topographic position (height above the local mean)
tpi=D-gaussian_filter(D,60/CELL)
ridge=1/(1+np.exp(tpi/4.0))                     # ~0 on crests, ~1 in gullies
# corridor around the drawn line
mask=np.zeros(D.shape,np.uint8)
pts=np.array([en2rc(e,n) for e,n in U])[:,::-1].round().astype(np.int32)   # (x=c,y=r)
cv2.polylines(mask,[pts.reshape(-1,1,2)],False,1,1)
dist=distance_transform_edt(mask==0)*CELL
# sidle section (through the bush on the way round): drawn-line vertices 14..22
sid=np.zeros(D.shape,np.uint8); cv2.polylines(sid,[pts[13:23].reshape(-1,1,2)],False,1,int(90/CELL))
wr=np.where(sid>0,0.15,1.0)
cost=1+6*wr*ridge+(dist/45.0)**2
cost[dist>140]=1e4
# waypoints: drawn-line vertices at the main turns; snap each to the local crest unless in the sidle
W_IDX=[0,4,9,13,19,23,28,32,len(U)-1]
path=[]
for a,b in zip(W_IDX[:-1],W_IDX[1:]):
    ra,ca=[int(round(v)) for v in en2rc(*U[a])]; rb,cb=[int(round(v)) for v in en2rc(*U[b])]
    seg,_=route_through_array(cost,(ra,ca),(rb,cb),fully_connected=True,geometric=True)
    path+=seg if not path else seg[1:]
P=np.array([rc2en(r,c) for r,c in path])
# smooth and resample every 10 m
P=np.stack([gaussian_filter1d(P[:,0],2.5,mode='nearest'),gaussian_filter1d(P[:,1],2.5,mode='nearest')],1)
s=np.concatenate([[0],np.cumsum(np.hypot(*np.diff(P,axis=0).T))]); t=np.arange(0,s[-1],10.0)
P=np.stack([np.interp(t,s,P[:,0]),np.interp(t,s,P[:,1])],1); P[-1]=[TE,TN]
def z_at(e,n):
    r,c=en2rc(e,n); return cv2.remap(D,np.float32(c).reshape(1,-1),np.float32(r).reshape(1,-1),cv2.INTER_LINEAR).ravel()
z=z_at(P[:,0],P[:,1]); zc=z_at(np.array([CE]),np.array([CN]))[0]
E=P[:,0]-CE; Nn=P[:,1]-CN; zrel=z-zc                 # heights relative to the ground at the camera
eye=1.6; vis=np.zeros(len(P),bool)
for i in range(len(P)):
    dd=math.hypot(E[i],Nn[i])
    if dd<30: vis[i]=True; continue
    ts=np.linspace(12/dd,1-15/dd,max(8,int(dd/7)))
    zz=z_at(CE+E[i]*ts,CN+Nn[i]*ts)-zc
    vis[i]=not (np.arctan2(zz-eye,ts*dd)>np.arctan2(zrel[i]+0.8-eye,dd)+0.003).any()
dev=[np.hypot(*(U-p).T).min() for p in P]
json.dump({'pts':np.round(np.stack([E,Nn,zrel],1),1).tolist(),'vis':vis.astype(int).tolist(),'length_m':float(s[-1]),'zRelative':True},open(S+'/route.json','w'))
print('pts',len(P),'length %.2f km'%(s[-1]/1000),'visible %.0f%%'%(100*vis.mean()),'deviation from drawn line: median %.0f m, max %.0f m'%(np.median(dev),np.max(dev)),'crest TPI median %.1f m'%np.median(tpi[tuple(np.array([en2rc(e,n) for e,n in P]).round().astype(int).T)]))
img=cv2.imdecode(np.frombuffer(open(S+'/topo_local.webp','rb').read(),np.uint8),1); sc=1400/14000
px=lambda e,n:(int((e+7000)*sc),int((7000-n)*sc))
for a,b in zip(U[:-1],U[1:]): cv2.line(img,px(*a),px(*b),(0,0,255),1)
for a,b in zip(P[:-1],P[1:]): cv2.line(img,px(*a),px(*b),(255,0,255),2)
x0,y0=px(-1500,4300); x1,y1=px(2600,-700); cv2.imwrite(S+'/st/route_map2.png',cv2.resize(img[y0:y1,x0:x1],None,fx=2,fy=2))
