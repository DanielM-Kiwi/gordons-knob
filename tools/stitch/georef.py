import numpy as np, math, json
S='work'
d=np.load(S+'/localdem.npy'); N=1000; CELL=14.0; HALF=7000
LAT0,LON0=-41.60794,172.94411
def en_to_ij(e,n): return int(round((HALF-n)/CELL-0.5)), int(round((e+HALF)/CELL-0.5))
def ij_to_en(i,j): return (j+0.5)*CELL-HALF, HALF-(i+0.5)*CELL
spots={'1685':(455,1050),'1614':(357,885),'1592':(282,870),'1365':(722,695),'1564':(910,773),'1519':(1030,480),'1371':(958,110),'1164':(480,150)}
PX=280.0
# initial guess: 1685 trig = DEM max near centre
i,j=np.unravel_index(d[N//2-60:N//2+60,N//2-60:N//2+60].argmax(),(120,120)); i+=N//2-60; j+=N//2-60
e0,n0=ij_to_en(i,j); print('DEM top near camera %.0f m at E%+.0f N%+.0f'%(d[i,j],e0,n0))
src=[];dst=[]
for name,(x,y) in spots.items():
    pe=e0+(x-455)*1000/PX; pn=n0-(y-1050)*1000/PX
    ii,jj=en_to_ij(pe,pn); k=int(160/CELL)
    win=d[ii-k:ii+k+1,jj-k:jj+k+1]; a,b=np.unravel_index(win.argmax(),win.shape)
    me,mn=ij_to_en(ii-k+a,jj-k+b)
    print('%s: map predicts E%+.0f N%+.0f, DEM summit %.0f m at E%+.0f N%+.0f (moved %.0f m)'%(name,pe,pn,win.max(),me,mn,math.hypot(me-pe,mn-pn)))
    src.append((x,y)); dst.append((me,mn))
src=np.array(src,float); dst=np.array(dst,float)
A=np.hstack([src,np.ones((len(src),1))]); M,*_=np.linalg.lstsq(A,dst,rcond=None)
res=np.hypot(*(A@M-dst).T); print('affine residual m:',np.round(res))
print('scale m/px %.3f %.3f'%(np.hypot(*M[0]),np.hypot(*M[1])))
route=[(370,5),(445,45),(460,90),(470,130),(480,170),(520,200),(560,230),(595,270),(620,310),(645,350),(660,380),(690,410),(720,428),(750,432),(770,440),(785,470),(790,520),(795,560),(800,575),(800,610),(775,625),(750,640),(735,660),(728,690),(715,720),(695,760),(670,800),(645,840),(625,875),(600,900),(570,915),(540,930),(510,945),(470,950),(440,960),(425,975),(422,1000),(435,1020),(455,1048)]
R=np.hstack([np.array(route,float),np.ones((len(route),1))])@M
json.dump({'M':M.tolist(),'route_en':R.tolist(),'trig_en':[e0,n0]},open(S+'/georef.json','w'))
print('route length %.1f km'%np.hypot(*np.diff(R,axis=0).T).sum()/1 if False else np.hypot(*np.diff(R,axis=0).T).sum()/1000)
good=['1685','1564','1519','1371','1164']
names=list(spots)
gi=[names.index(g) for g in good]
s_=src[gi]; t_=dst[gi]
# similarity fit (scale, rotation, shift)
sc=s_-s_.mean(0); tc=t_-t_.mean(0)
# map image (x right, y down) to (e right, n up): flip y
sc[:,1]*=-1
H=sc.T@tc; U,Sg,Vt=np.linalg.svd(H); Rm=(U@Vt).T; scale=Sg.sum()/(sc**2).sum()
def img2en(p):
    p=np.atleast_2d(np.array(p,float)); q=p-s_.mean(0); q[:,1]*=-1; return (scale*(Rm@q.T)).T+t_.mean(0)
r=np.hypot(*(img2en(s_)-t_).T); print('similarity: scale %.3f m/px rot %.2f deg resid'%(scale,math.degrees(math.atan2(Rm[1,0],Rm[0,0]))),np.round(r))
for nm in names:
    e,n=img2en(spots[nm])[0]; ii,jj=en_to_ij(e,n); print('  %s at E%+.0f N%+.0f DEM %.0f'%(nm,e,n,d[ii,jj]))
R=img2en(route)
json.dump({'route_en':R.tolist(),'trig_en':[e0,n0],'scale':scale},open(S+'/georef.json','w'))
print('route km %.2f'%(np.hypot(*np.diff(R,axis=0).T).sum()/1000))
