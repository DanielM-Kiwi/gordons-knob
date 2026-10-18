import numpy as np, json, cv2, math
S='work'
T=json.load(open(S+'/db/route/trace.json')); pts=np.array(T['data']['pts'] if 'data' in T else T['pts'],float)
cb=json.load(open(S+'/calib_b.json')); CB=np.array(cb['bearing']); CE=np.array(cb['elev'])
def four(c,b):
    r=np.radians(b); v=c[0]+0*r
    for k in range(1,(len(c)-1)//2+1): v=v+c[2*k-1]*np.sin(k*r)+c[2*k]*np.cos(k*r)
    return v
tb=(pts[:,0]+four(CB,pts[:,0]))%360; te=pts[:,1]-four(CE,pts[:,0]); hid=pts[:,2]>0
d14=np.load(S+'/localdem.npy'); HALF=7000.0; CELL=14.0
CEn,CNn=59.0+15.5,-213.0-7.4
def z(e,n): return cv2.remap(d14,np.float32((e+HALF)/CELL-0.5).reshape(1,-1),np.float32((HALF-n)/CELL-0.5).reshape(1,-1),cv2.INTER_LINEAR).ravel()
eye=z(np.array([CEn]),np.array([CNn]))[0]+1.6
dist=np.concatenate([np.arange(5,600,2),np.arange(600,6000,8)])
D=np.full(len(pts),np.nan)
for i,(b,e) in enumerate(zip(tb,te)):
    br=math.radians(b); zz=z(CEn+math.sin(br)*dist,CNn+math.cos(br)*dist)
    ang=np.degrees(np.arctan2(zz-eye,dist)); h=np.where(ang>=e)[0]
    if len(h): D[i]=dist[h[0]]
print('n',len(pts),'hidden',hid.sum())
for i in range(len(pts)): print(i,'band %.2f %.2f true %.1f %.2f hid %d dist %s'%(pts[i,0],pts[i,1],tb[i],te[i],hid[i],'%.0f'%D[i] if D[i]==D[i] else '-'))
np.save(S+'/trace_raw.npy',np.column_stack([pts,tb,te,D]))
