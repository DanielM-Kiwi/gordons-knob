import cv2, numpy as np
from scipy.ndimage import binary_fill_holes
im=cv2.imread('Looking south I think with us in it.jpg')
X0,Y0=1500,1700; c=im[Y0:Y0+220,X0:X0+380].copy(); f=c.astype(np.float32)
B,G,R=f[...,0],f[...,1],f[...,2]
mask=np.full(c.shape[:2],cv2.GC_PR_BGD,np.uint8)
# definite background: bright hazy blue (mountains/sky) and tussock far from the people
mask[(B-R>55)&(B>175)]=cv2.GC_BGD
# probable foreground: dark clothing, saturated blue jacket, skin
dark=(f.max(2)<85); jacket=(B>140)&(R<75)&(G<110); skin=(R>G)&(G>B)&(R-B>25)&(R>110)
hair=(R>B+5)&(f.max(2)<170)&(R>60)
mask[dark|jacket|skin|hair]=cv2.GC_PR_FGD
# definite foreground seeds along every limb (crop coordinates)
limbs=[[(88,32),(105,45),(130,60),(150,62)], [(85,33),(88,32)], [(30,130),(60,125),(95,116),(115,112)],
       [(127,145),(128,170),(130,198)], [(150,55),(155,90),(140,120)],
       [(215,62),(240,75),(265,85)], [(290,90),(320,88),(340,88),(350,93)], [(157,131),(190,125),(230,122),(250,118)],
       [(248,135),(250,165),(250,192)], [(270,75),(262,105),(258,125)],
       [(143,48),(150,41),(158,42),(164,50)], [(262,72),(266,64),(274,62),(282,68),(286,80)], [(284,72),(288,84)]]
for L in limbs: cv2.polylines(mask,[np.array(L)],False,int(cv2.GC_FGD),3)
bg=np.zeros((1,65)); fg=np.zeros((1,65))
cv2.grabCut(c,mask,None,bg,fg,12,cv2.GC_INIT_WITH_MASK)
m=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
# keep components touching a seed, close small gaps, fill holes
seed=np.zeros_like(m)
for L in limbs: cv2.polylines(seed,[np.array(L)],False,1,3)
n,lab,st,_=cv2.connectedComponentsWithStats(m)
keep=np.isin(lab,[i for i in range(1,n) if (seed[lab==i]).any()]).astype(np.uint8)
keep=cv2.morphologyEx(keep,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
filled=binary_fill_holes(keep).astype(np.uint8)
hn,hl,hs,_=cv2.connectedComponentsWithStats((filled-keep).astype(np.uint8))
for i in range(1,hn):
    if hs[i,4]<400: keep[hl==i]=1
np.save('people2.npy',np.array([X0,Y0])); np.save('people2_mask.npy',keep)
vis=c.copy(); vis[keep==0]=(vis[keep==0]*0.3).astype(np.uint8)
cv2.imwrite('st/people2.png',cv2.resize(vis,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
print(keep.sum())
# soft matte on the outer band: opacity from how unlike the blue background each pixel is
core=cv2.erode(keep,np.ones((5,5),np.uint8))
band=cv2.dilate(keep,np.ones((3,3),np.uint8)).astype(bool)&~core.astype(bool)
blu=(B-R)
alpha=core.astype(np.float32)
a_band=np.clip((62-blu)/34,0,1)*np.clip((215-B)/40+0.4,0,1)
alpha[band]=a_band[band]
sky=((B>180)&(G>135)&(blu>45))|((blu>70)&(G>98)&(G-R>35))
skys=cv2.GaussianBlur(sky.astype(np.float32),(0,0),1.0)
alpha=alpha*np.clip(1-skys*1.6,0,1)
np.save('people2_alpha.npy',alpha)
vis=(c.astype(np.float32)*alpha[...,None]+np.array([40,40,40])*(1-alpha[...,None])).astype(np.uint8)
cv2.imwrite('st/people2_matte.png',cv2.resize(vis,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)[60:330,380:960])
