import numpy as np, math, sys
exec(open('tools/stitch/geo.py').read())
pts=np.load(sys.argv[1]); w,h=float(sys.argv[2]),float(sys.argv[3]); f0=float(sys.argv[4])
P=pts[:,:2]; Q=pts[:,2:]
Bq,Eq=pano_to_dir(Q[:,0],Q[:,1]); vq=dir_vec(Bq,Eq)
def rays(P,fp): v=np.stack([(P[:,0]-w/2)/fp,-(P[:,1]-h/2)/fp,-np.ones(len(P))],1); return v/np.linalg.norm(v,axis=1,keepdims=True)
def kabsch(A,B):
    H=A.T@B; U,S_,Vt=np.linalg.svd(H); D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); return Vt.T@D@U.T
best=None
for fp in np.linspace(f0*0.8,f0*1.25,181):
    vp=rays(P,fp); Rm=kabsch(vp,vq); e=np.degrees(np.arccos(np.clip(np.sum((vp@Rm.T)*vq,1),-1,1)))
    m=np.median(e)
    if best is None or m<best[0]: best=(m,fp,Rm,e)
m,fp,Rm,e=best
print('f %.0f median err %.3f deg  p90 %.3f'%(fp,m,np.percentile(e,90)))
fwd=Rm@np.array([0,0,-1.]); print('optical axis bearing %.1f elev %.1f'%(math.degrees(math.atan2(fwd[0],fwd[1]))%360, math.degrees(math.asin(fwd[2]))))
np.save(sys.argv[5],np.concatenate([[fp,w/2,h/2],Rm.ravel()]))
