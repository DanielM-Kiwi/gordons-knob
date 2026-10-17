import numpy as np, cv2
S='work'
d=np.load(S+'/localdem.npy'); N=d.shape[0]; CELL=14.0; OUT=1400; sc=OUT/N
D=cv2.resize(d,(OUT,OUT),interpolation=cv2.INTER_CUBIC); cs=CELL/sc
gy,gx=np.gradient(D,cs)
L=np.array([-0.5,0.5,0.7071]); L/=np.linalg.norm(L)
nrm=np.dstack([-gx,gy,np.ones_like(D)]); nrm/=np.linalg.norm(nrm,axis=2,keepdims=True)
hs=np.clip(nrm@L,0,1)
stops=[(300,(206,222,196)),(700,(190,212,170)),(1100,(176,204,156)),(1300,(214,214,180)),(1500,(232,224,200)),(1700,(244,240,232))]
z=D; col=np.zeros(D.shape+(3,),np.float32)
for i in range(len(stops)-1):
    z0,c0=stops[i]; z1,c1=stops[i+1]; t=np.clip((z-z0)/(z1-z0),0,1)[...,None]
    m=((z>=z0)|(i==0))&((z<z1)|(i==len(stops)-2))
    col[m]=(np.array(c0)*(1-t)+np.array(c1)*t)[m]
img=col*(0.58+0.55*hs[...,None])
img=np.clip(img,0,255).astype(np.uint8)[:,:,::-1].copy()
# contours: 20 m, index 100 m
for L_ in range(300,1720,20):
    m=(D>=L_).astype(np.uint8)
    cnt,_=cv2.findContours(m,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    idx=L_%100==0
    cv2.drawContours(img,cnt,-1,(40,80,140) if idx else (70,110,160),1 if not idx else 2,lineType=cv2.LINE_AA)
# 1 km grid
for k in range(-7,8):
    p=int((k*1000+7000)/CELL*sc); cv2.line(img,(p,0),(p,OUT),(140,120,90),1); cv2.line(img,(0,p),(OUT,p),(140,120,90),1)
ok,b=cv2.imencode('.webp',img,[cv2.IMWRITE_WEBP_QUALITY,86]); open(S+'/topo_local.webp','wb').write(b.tobytes())
cv2.imwrite(S+'/st/topo_local.jpg',cv2.resize(img,(700,700)))
print(len(b)/1e6)
