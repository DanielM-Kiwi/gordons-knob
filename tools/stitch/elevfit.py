import numpy as np, json, math
from scipy.optimize import least_squares
S='work'
exec(open(S+'/calfit.py').read().split("for order in")[0])
sm=np.load(S+'/skymatch.npy'); xs=sm[:,0]; es=sm[:,2]
cb=json.load(open(S+'/calib_b.json')); ce_,cn_=cb['cam_offset_en']
hts={'Gordons Knob':1592,'Point 1164':1164,'Point 1365':1365,'Point 1519':1519,'Point 1564':1564,'Mt Owen':1875,'Mt Richmond':1760,'Mt Rintoul':1731,'Lake Rotoiti':615,'Nelson':10,'Richmond':20,'Motueka':10}
pos=np.stack([np.sin(np.radians(tb))*dk*1000-ce_,np.cos(np.radians(tb))*dk*1000-cn_],1); dd=np.hypot(*pos.T)
H=np.array([hts[x] for x in n])
def true_el(h0): return np.degrees(np.arctan2(H-h0-dd*dd/(2*6371000)*0.87,dd))
def F(b,o): r=np.radians(b); return np.stack([np.ones_like(r)]+sum([[np.sin(k*r),np.cos(k*r)] for k in range(1,o+1)],[]),1)
wl=np.array([0.3 if x in ('Nelson','Richmond','Motueka') else 1.5 for x in n])
def resid(p):
    c=p[:5]; h0=1686.6+p[5]
    return np.concatenate([(es-F(xs,2)@c), ((be-true_el(h0))-F(bb,2)@c)*wl])
sol=least_squares(resid,np.zeros(6),loss='soft_l1',f_scale=0.2); p=sol.x
print('camera height change %+.1f m  -> eye at %.1f m'%(p[5],1686.6+p[5]))
r=(be-true_el(1686.6+p[5]))-F(bb,2)@p[:5]
for i in range(len(n)): print('   %-13s elev resid %+.2f'%(n[i],r[i]))
cb['elev']=p[:5].tolist(); cb['order_e']=2; cb['eye_height']=1686.6+p[5]
json.dump(cb,open(S+'/calib_b.json','w')); print(cb)
