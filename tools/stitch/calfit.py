import numpy as np, json
rows=[("Gordons Knob",322.2136,-5.5707,316.2078,-6.1035,0.88),("Lake Rotoiti",209.6561,-3.067,197.8502,-2.3945,26),
("Motueka",19.5874,-2.2543,5.9151,-1.9766,54),("Mt Owen",285.4856,0.1983,281.0025,0.1794,34),("Mt Richmond",78.3361,0.33,67.9446,-0.0539,40),
("Mt Rintoul",77.2048,0.4077,67.0902,-0.0046,26),("Nelson",46.9778,-2.2362,36.7792,-2.2158,47),("Point 1164",14.8464,-9.4786,1.6748,-9.3172,3.2),
("Point 1365",49.1923,-10.622,37.0275,-11.5518,1.6),("Point 1519",56.8656,-2.7804,45.3468,-3.3532,3.9),("Point 1564",68.451,-2.6971,58.7843,-3.7226,2.0),
("Richmond",44.2764,-2.6662,33.228,-2.7787,36)]
n=[r[0] for r in rows]; bb=np.array([r[1] for r in rows]); be=np.array([r[2] for r in rows]); tb=np.array([r[3] for r in rows]); te=np.array([r[4] for r in rows]); dk=np.array([r[5] for r in rows])
d=((tb-bb+180)%360)-180
w=np.where(dk<5,0.5,1.0)
def basis(b,order):
    r=np.radians(b); cols=[np.ones_like(r)]
    for k in range(1,order+1): cols+= [np.sin(k*r),np.cos(k*r)]
    return np.stack(cols,1)
for order in (1,2,3):
    A=basis(bb,order); c,*_=np.linalg.lstsq(A*w[:,None],d*w,rcond=None); r=d-A@c
    # leave-one-out
    loo=[]
    for i in range(len(bb)):
        m=np.arange(len(bb))!=i; ci,*_=np.linalg.lstsq(A[m]*w[m,None],d[m]*w[m],rcond=None); loo.append(d[i]-A[i]@ci)
    print('bearing order',order,'rms resid %.2f  loo rms %.2f'%(np.sqrt(np.mean(r**2)),np.sqrt(np.mean(np.array(loo)**2))), np.round(c,3))
order=2; A=basis(bb,order); cb,*_=np.linalg.lstsq(A*w[:,None],d*w,rcond=None); rb=d-A@cb
# elevation: band_el - true_el = c0 + a sin + b cos + s*true_el
de=be-te
E=np.column_stack([basis(bb,1),te]); ce,*_=np.linalg.lstsq(E*w[:,None],de*w,rcond=None); re=de-E@ce
print('elev coeffs',np.round(ce,4),'rms %.2f'%np.sqrt(np.mean(re**2)))
for i in range(len(n)): print('  %-13s bearing resid %+.2f  elev resid %+.2f'%(n[i],rb[i],re[i]))
json.dump({'bearing':cb.tolist(),'elev':ce.tolist()},open('work/calib.json','w'))
