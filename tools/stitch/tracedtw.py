import numpy as np, json, math
S='work'
T=np.load(S+'/trace_raw.npy'); tb,te,hid=T[:,3],T[:,4],T[:,2]>0
R=json.load(open(S+'/route.json')); P=np.array(R['pts']); E,N,Z=P.T
d=np.hypot(E,N); rb=np.degrees(np.arctan2(E,N))%360; re=np.degrees(np.arctan2(Z+0.4-1.6,np.maximum(d,1)))
def ang(b1,e1,b2,e2):
    db=((b1-b2+180)%360-180)*np.cos(np.radians((e1+e2)/2)); return np.hypot(db,e1-e2)
n,m=len(tb),len(rb)
C=ang(tb[:,None],te[:,None],rb[None],re[None])
INF=1e9; A=np.full((n+1,m+1),INF); A[0,0]=0; bp=np.zeros((n+1,m+1),int)
for i in range(1,n+1):
    for j in range(1,m+1):
        opts=(A[i-1,j-1],A[i-1,j],A[i,j-1]*1.0+0)  # diag, trace-stay, route-skip(free-ish)
        k=int(np.argmin([opts[0],opts[1],A[i,j-1]+0.02])); 
        A[i,j]=C[i-1,j-1]+[opts[0],opts[1],A[i,j-1]][k]; bp[i,j]=k
# backtrack from (n, best j near end)
i,j=n,m; match=np.zeros(n,int)
while i>0 and j>0:
    match[i-1]=max(match[i-1],j-1) if match[i-1] else j-1
    k=bp[i,j]
    if k==0: i,j=i-1,j-1
    elif k==1: i-=1
    else: j-=1
D=d[match]; err=C[np.arange(n),match]
for i in range(0,n,6): print(i,'trace %.1f %.1f -> route #%d dist %.0f m (route %.1f %.1f) err %.2f'%(tb[i],te[i],match[i],D[i],rb[match[i]],re[match[i]],err[i]))
print('median angular err %.2f deg'%np.median(err))
np.save(S+'/trace_dist.npy',D)
