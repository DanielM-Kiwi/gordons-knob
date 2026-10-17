import numpy as np, json, math, cv2
from scipy.ndimage import gaussian_filter1d
S='work'
g=json.load(open(S+'/georef.json')); R=np.array(g['route_en'])     # relative to (-41.60794,172.94411)
loc=np.load(S+'/localdem.npy'); CELL=14.0; HALF=7000.0
TE,TN=59.0,-213.0                                                  # trig in the same frame
def z_at(e,n):
    fx=((e+HALF)/CELL-0.5).astype(np.float32); fy=((HALF-n)/CELL-0.5).astype(np.float32)
    return cv2.remap(loc,fx.reshape(1,-1),fy.reshape(1,-1),cv2.INTER_LINEAR).ravel()
# densify to 10 m
seg=np.hypot(*np.diff(R,axis=0).T); s=np.concatenate([[0],np.cumsum(seg)])
t=np.arange(0,s[-1],10.0); P=np.stack([np.interp(t,s,R[:,0]),np.interp(t,s,R[:,1])],1)
P=np.stack([gaussian_filter1d(P[:,0],3,mode='nearest'),gaussian_filter1d(P[:,1],3,mode='nearest')],1)
# snap toward the ridge crest: search +-35 m across the route direction for the highest ground
dirv=np.gradient(P,axis=0); dirv/=np.linalg.norm(dirv,axis=1,keepdims=True); nrm=np.stack([-dirv[:,1],dirv[:,0]],1)
offs=np.arange(-35,36,5.0); best=np.zeros(len(P))
Z=np.stack([z_at(P[:,0]+o*nrm[:,0],P[:,1]+o*nrm[:,1]) for o in offs],1)
pen=Z-0.08*np.abs(offs)[None]                     # prefer staying near the drawn line
best=offs[pen.argmax(1)]; best=gaussian_filter1d(best,4)
P=P+best[:,None]*nrm
P[-1]=[TE,TN]                                     # ends at the trig
z=z_at(P[:,0],P[:,1])
# heights relative to the trig: the DEM reads the summit ~14 m low; blend that correction in near the top
zt=z_at(np.array([TE]),np.array([TN]))[0]
DATUM=7.0                                         # elevation model reads ~7 m below the Topo50 spot heights
CE,CN=TE+15.5,TN-7.4                              # camera: calibrated offset from the trig
E=P[:,0]-CE; N=P[:,1]-CN
# visibility from the eye at the trig (1.6 m), terrain from the hi-res DEM
eye=z_at(np.array([TE+15.5]),np.array([TN-7.4]))[0]+1.6   # eye in the model's own datum
print('camera ground (model) %.1f m, eye %.1f m'%(eye-1.6,eye))
vis=np.zeros(len(P),bool)
for i in range(len(P)):
    d=math.hypot(E[i],N[i])
    if d<20: vis[i]=True; continue
    ts=np.linspace(15/d,1-20/d,max(8,int(d/10)))
    zz=z_at(CE+E[i]*ts,CN+N[i]*ts)
    a_t=np.arctan2(z[i]+1.0-eye,d); a_s=np.arctan2(zz-eye,ts*d)
    vis[i]=not (a_s>a_t+0.002).any()
ls=np.round(np.stack([E,N,z+DATUM],1),1).tolist()
json.dump({'pts':ls,'vis':vis.astype(int).tolist(),'length_m':float(s[-1])},open(S+'/route.json','w'))
print('pts',len(P),'visible %.0f%%'%(100*vis.mean()),'start E%+.0f N%+.0f z%.0f'%(E[0],N[0],z[0]),'snap offsets mean %.1f m'%np.abs(best).mean())
