import numpy as np, json, math, re
from scipy.optimize import least_squares
S='work'
src=open('src/panorama.html').read()
pois={m[0]:(float(m[1]),float(m[2]),None if m[3]=='null' else float(m[3])) for m in re.findall(r'name: "([^"]+)",\s+lat: (-?[\d.]+), lon: (-?[\d.]+), height: (null|[\d.]+)',src)}
L=[("gordons-knob",322.2136,-5.5707),("lake-rotoiti",209.6561,-3.067),("motueka",19.5874,-2.2543),("mt-arthur",335.6843,-0.5538),("mt-owen",285.4856,0.1983),
("mt-richmond",78.3361,0.33),("mt-rintoul",77.2048,0.4077),("nelson",46.9778,-2.2362),("point-1164",14.8464,-9.4786),("point-1365",49.1923,-10.622),
("point-1371",40.8473,-4.2925),("point-1519",56.8656,-2.7804),("point-1564",68.451,-2.6971),("point-1614",337.9234,-5.6691),("red-hill",109.0476,0.9813),("richmond",44.2764,-2.6662)]
name={k:v for k,v in [(n.lower().replace(' ','-'),n) for n in pois]}
LAT0,LON0=-41.60985,172.94482; mlat=111320; mlon=111320*math.cos(math.radians(LAT0))
rows=[]
for k,bb,be in L:
    la,lo,h=pois[name[k]]; rows.append((k,bb,be,(lo-LON0)*mlon,(la-LAT0)*mlat,h if h else {'lake-rotoiti':615,'nelson':10,'richmond':20,'motueka':10}.get(k,10)))
nm=[r[0] for r in rows]; bb=np.array([r[1] for r in rows]); be=np.array([r[2] for r in rows]); PE=np.array([r[3] for r in rows]); PN=np.array([r[4] for r in rows]); H=np.array([r[5] for r in rows])
sm=np.load(S+'/skymatch.npy'); c1=np.array([-9.776,-2.146,0.609])
prior=lambda b:c1[0]+c1[1]*np.sin(np.radians(b))+c1[2]*np.cos(np.radians(b))
xs=sm[:,0]; ds=prior(xs)+sm[:,1]; es=sm[:,2]
def F(b,o): r=np.radians(b); return np.stack([np.ones_like(r)]+sum([[np.sin(k*r),np.cos(k*r)] for k in range(1,o+1)],[]),1)
town={'nelson','richmond','motueka'}; wl=np.array([0.3 if k in town else 1.5 for k in nm])
OB=4; nb=2*OB+1
def tb_of(ce,cn): return np.degrees(np.arctan2(PE-ce,PN-cn))%360
def res(p,ws):
    c=p[:nb]; ce,cn=p[nb],p[nb+1]
    dl=((tb_of(ce,cn)-bb+180)%360)-180
    return np.concatenate([(ds-F(xs,OB)@c)*ws,(dl-F(bb,OB)@c)*wl,[ce/6,cn/6]])
ws=np.ones(len(xs)); p=np.zeros(nb+2)
for it in range(8):
    p=least_squares(res,p,args=(ws,)).x; r=ds-F(xs,OB)@p[:nb]; s=1.4826*np.median(np.abs(r[ws>0])); ws=np.clip(1-(r/(3*s))**2,0,1)**2
c=p[:nb]; ce,cn=p[nb],p[nb+1]; rl=((tb_of(ce,cn)-bb+180)%360)-180-F(bb,OB)@c
print('camera from trig E %+.1f N %+.1f'%(ce,cn))
for i,k in enumerate(nm): print('  %-13s bearing resid %+.2f'%(k,rl[i]))
dd=np.hypot(PE-ce,PN-cn)
def te(h0): return np.degrees(np.arctan2(H-h0-dd*dd/(2*6371000)*0.87,dd))
def rese(q): return np.concatenate([es-F(xs,2)@q[:5],((be-te(1680+q[5]))-F(bb,2)@q[:5])*wl])
q=least_squares(rese,np.zeros(6),loss='soft_l1',f_scale=0.2).x; re_=(be-te(1680+q[5]))-F(bb,2)@q[:5]
print('eye %.1f m'%(1680+q[5])); [print('  %-13s elev resid %+.2f'%(k,re_[i])) for i,k in enumerate(nm)]
mlon0=111320*math.cos(math.radians(LAT0))
json.dump({'bearing':c.tolist(),'elev':q[:5].tolist(),'cam':[LAT0+cn/mlat,LON0+ce/mlon],'eye':1680+q[5]},open(S+'/calib_c.json','w'))
# local corrections: Gaussian bumps centred on each placement, solved so markers land on the placements
SIG=6.0
def wrap(x): return (x+180)%360-180
K=np.exp(-(wrap(bb[:,None]-bb[None,:])/SIG)**2)
wt=np.where(np.isin(nm,list(town)),0.5,1.0)
lam=0.05
wB=np.linalg.solve(K+lam*np.eye(len(bb)),rl*wt); wE=np.linalg.solve(K+lam*np.eye(len(bb)),re_*wt)
fitB=K@wB; fitE=K@wE
print('after local corrections: bearing resid max %.2f, elev resid max %.2f'%(np.abs(rl-fitB)[wt==1].max(),np.abs(re_-fitE)[wt==1].max()))
print('towns:',dict(zip([k for k in nm if k in town],np.round((rl-fitB)[np.isin(nm,list(town))],2))))
out=json.load(open(S+'/calib_c.json')); out.update({'rbfC':bb.tolist(),'rbfB':wB.tolist(),'rbfE':wE.tolist(),'rbfSig':SIG})
json.dump(out,open(S+'/calib_c.json','w')); print(out['cam'],out['eye'])
print('--- regularisation sweep')
bgrid=np.arange(0,360,0.05)
def fold_and_resid(SIG,lam):
    K=np.exp(-(wrap(bb[:,None]-bb[None,:])/SIG)**2)
    wB=np.linalg.solve(K+lam*np.diag(1/wt),rl); wE=np.linalg.solve(K+lam*np.diag(1/wt),re_)
    Kg=np.exp(-(wrap(bgrid[:,None]-bb[None,:])/SIG)**2)
    f=F(bgrid,OB)@c+Kg@wB; der=1+np.gradient(f,bgrid)
    return der.min(),np.abs(rl-K@wB),wB,wE
best=None
for SIG in (6,8,10,14):
  for lam in (0.05,0.2,0.5,1,2):
    m,r,wB,wE=fold_and_resid(SIG,lam)
    ok=m>0.6
    print('sig %2d lam %.2f  min slope %.2f  max resid (non-town) %.2f  mean %.2f'%(SIG,lam,m,r[wt==1].max(),r[wt==1].mean()))
    if ok and (best is None or r[wt==1].mean()<best[0]): best=(r[wt==1].mean(),SIG,lam,wB,wE,r)
_,SIG,lam,wB,wE,r=best
print('chosen sig %d lam %.2f'%(SIG,lam)); [print('   %-13s %+.2f'%(k,(rl-np.exp(-(wrap(bb[:,None]-bb[None,:])/SIG)**2)@wB)[i])) for i,k in enumerate(nm)]
out=json.load(open(S+'/calib_c.json')); out.update({'rbfB':wB.tolist(),'rbfE':wE.tolist(),'rbfSig':SIG}); json.dump(out,open(S+'/calib_c.json','w'))
