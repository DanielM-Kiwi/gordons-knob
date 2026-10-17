import numpy as np, json, math
from scipy.optimize import least_squares
S='work'
exec(open(S+'/calfit.py').read().split("for order in")[0])
sm=np.load(S+'/skymatch.npy'); c1=np.array([-9.776,-2.146,0.609])
def prior(b): r=np.radians(b); return c1[0]+c1[1]*np.sin(r)+c1[2]*np.cos(r)
xs=sm[:,0]; ds=prior(xs)+sm[:,1]
# landmark positions relative to the trig (E,N metres) from their true bearing+distance
pos=np.stack([np.sin(np.radians(tb))*dk*1000,np.cos(np.radians(tb))*dk*1000],1)
kind={'Nelson':'town','Richmond':'town','Motueka':'town'}
wl=np.array([0.3 if kind.get(x) else 1.5 for x in n])
def F(b,order): r=np.radians(b); return np.stack([np.ones_like(r)]+sum([[np.sin(k*r),np.cos(k*r)] for k in range(1,order+1)],[]),1)
ORDER=3; nc=2*ORDER+1
def resid(p,ws,wlan):
    c=p[:nc]; ce,cn=p[nc],p[nc+1]
    tb2=(np.degrees(np.arctan2(pos[:,0]-ce,pos[:,1]-cn)))%360
    dl=((tb2-bb+180)%360)-180
    rs=(ds-F(xs,ORDER)@c)*ws; rl=(dl-F(bb,ORDER)@c)*wlan
    return np.concatenate([rs,rl,[ce/40,cn/40]])          # weak prior: camera within ~40 m of the trig
ws=np.ones(len(xs)); p=np.zeros(nc+2)
for it in range(8):
    sol=least_squares(resid,p,args=(ws,wl)); p=sol.x
    rs=(ds-F(xs,ORDER)@p[:nc]); s=1.4826*np.median(np.abs(rs[ws>0])); ws=np.clip(1-(rs/(3*s))**2,0,1)**2
c=p[:nc]; ce,cn=p[nc],p[nc+1]
tb2=(np.degrees(np.arctan2(pos[:,0]-ce,pos[:,1]-cn)))%360
dl=((tb2-bb+180)%360)-180; rl=dl-F(bb,ORDER)@c
print('camera offset from trig: E %+.1f m  N %+.1f m   sky inliers %d/%d sigma %.2f'%(ce,cn,(ws>0).sum(),len(xs),s))
for i in range(len(n)): print('   %-13s %5.1f km  resid %+.2f'%(n[i],dk[i],rl[i]))
json.dump({'bearing':c.tolist(),'order_b':ORDER,'cam_offset_en':[ce,cn]},open(S+'/calib_b.json','w'))
