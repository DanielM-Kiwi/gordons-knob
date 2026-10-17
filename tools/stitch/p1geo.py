import numpy as np, math
S='work'
exec(open('tools/stitch/geo.py').read())
d=np.load('work/p1p2.npy'); A=d[:,:2]; B=d[:,2:].copy()
B[B[:,0]>=18704,0]-=18704
be,el=pano_to_dir(B[:,0],B[:,1])
# unwrap bearing relative to p1 x
o=np.argsort(A[:,0]); A=A[o]; be=be[o]; el=el[o]
ref=(A[:,0]/C*360)
off=((be-ref-180)%360)-180+180  # coarse
be_u=ref+((be-ref - np.median((be-ref)%360) +180)%360-180)+np.median((be-ref)%360)
# global f1 + per-bin horizon via least squares
bins=np.arange(0,A[:,0].max()+300,300); idx=np.digitize(A[:,0],bins)-1; nb=len(bins)
T=np.tan(el)
for it in range(3):
    M=np.zeros((len(A),nb+1)); M[np.arange(len(A)),idx]=1; M[:,-1]=-T
    if it==0: keep=np.ones(len(A),bool)
    sol,*_=np.linalg.lstsq(M[keep],A[keep,1],rcond=None)
    r=A[:,1]-M@sol; keep=np.abs(r)<max(3,2.5*np.median(np.abs(r[keep])))
f1=sol[-1]; hb=sol[:-1]
print('f1 %.1f (p2 f %.1f)  resid median px %.2f  kept %d/%d'%(f1,f,np.median(np.abs(r[keep])),keep.sum(),len(A)))
# per-bin bearing
bb=np.array([np.median(be_u[keep&(idx==i)]) if (keep&(idx==i)).sum()>3 else np.nan for i in range(nb)])
hbv=np.array([hb[i] if (keep&(idx==i)).sum()>3 else np.nan for i in range(nb)])
xc=bins+150; ok=~np.isnan(bb)
# fit smooth: bearing linear+cubic, horizon cubic
pb=np.polyfit(xc[ok],bb[ok],3); ph=np.polyfit(xc[ok],hbv[ok],3)
print('bearing resid deg', np.round(bb[ok]-np.polyval(pb,xc[ok]),3))
print('hor resid px', np.round(hbv[ok]-np.polyval(ph,xc[ok]),1))
print('p1 bearing range %.1f..%.1f'%(np.polyval(pb,0),np.polyval(pb,10544)))
np.save('work/p1geo.npy',np.concatenate([[f1],pb,ph]))
from scipy.ndimage import median_filter
xs_=xc[ok]; bbs=median_filter(bb[ok],3,mode='nearest'); hbs=median_filter(hbv[ok],3,mode='nearest')
print('monotonic', np.all(np.diff(bbs)>0))
# per-point check of interpolated model
bi=np.interp(A[keep,0],xs_,bbs); hi=np.interp(A[keep,0],xs_,hbs)
eb=be_u[keep]-bi; ey=A[keep,1]-(hi-f1*T[keep])
print('interp model: bearing err median %.3f deg p90 %.3f ; y err median %.1f px p90 %.1f'%(np.median(np.abs(eb)),np.percentile(np.abs(eb),90),np.median(np.abs(ey)),np.percentile(np.abs(ey),90)))
np.savez(S+'/p1model.npz',x=xs_,b=bbs,h=hbs,f1=f1)
# bearing = b(x) + s(x)*(y-1400) per bin, then joint check
bin2=np.arange(0,A[:,0].max()+200,200); id2=np.digitize(A[:,0],bin2)-1
bx=[];bb2=[];ss=[]
for i in range(len(bin2)):
    m=keep&(id2==i)
    if m.sum()<12: continue
    Y=A[m,1]-1400; X=np.stack([np.ones(m.sum()),Y,A[m,0]-bin2[i]-100],1)
    w=np.ones(m.sum(),bool)
    for _ in range(3):
        c,*_=np.linalg.lstsq(X[w],be_u[m][w],rcond=None); r=be_u[m]-X@c; w=np.abs(r)<max(0.03,2.5*np.median(np.abs(r[w])))
    bx.append(bin2[i]+100); bb2.append(c[0]); ss.append(c[1])
bx=np.array(bx); bb2=np.array(bb2); ss=median_filter(np.array(ss),5,mode='nearest')
def p1dir(x,y):
    return np.interp(x,bx,bb2)+np.interp(x,bx,ss)*(y-1400), np.arctan((np.interp(x,xs_,hbs)-y)/f1)
bq,_=p1dir(A[keep,0],A[keep,1]); eb=be_u[keep]-bq
print('shear model bearing err median %.3f p90 %.3f deg; shear range'%(np.median(np.abs(eb)),np.percentile(np.abs(eb),90)), ss.min(),ss.max())
np.savez(S+'/p1model.npz',x=xs_,h=hbs,f1=f1,bx=bx,bb=bb2,ss=ss)
