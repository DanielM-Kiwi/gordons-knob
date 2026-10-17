import cv2, numpy as np, math
S='work'
img=cv2.imread(S+'/band_base.png').astype(np.float32)
W,Hh,TOP,BOT=np.load(S+'/band_meta.npy'); W=int(W); Hh=int(Hh); ppd=W/360
q=np.load(S+'/rot_us.npy'); fp,cx,cy=q[:3]; Rm=q[3:].reshape(3,3)
ph=cv2.imread('Looking south I think with us in it.jpg').astype(np.float32); h,w=ph.shape[:2]
fg=np.load(S+'/crew_fg.npy').astype(np.float32)
# soft edges: people/knoll silhouette 1.5 px, plus overall soft limit around the knoll
fgs=cv2.GaussianBlur(fg,(0,0),0.5)
def to_photo(Bdeg,Edeg):
    B=np.radians(Bdeg); E=np.radians(Edeg)
    d=np.stack([np.sin(B)*np.cos(E),np.cos(B)*np.cos(E),np.sin(E)],-1); cam=d@Rm; z=-cam[...,2]
    return (cx+fp*cam[...,0]/z).astype(np.float32),(cy-fp*cam[...,1]/z).astype(np.float32),z>0
B0,B1,E0,E1=176.0,193.5,-9.5,1.0          # patch extent (old-frame bearings)
def render(ppd_out):
    nb=int(round((B1-B0)*ppd_out)); ne=int(round((E1-E0)*ppd_out))
    Bg,Eg=np.meshgrid(B0+(np.arange(nb)+0.5)/ppd_out, E1-(np.arange(ne)+0.5)/ppd_out)
    mx,my,ok=to_photo(Bg,Eg)
    col=cv2.remap(ph,mx,my,cv2.INTER_AREA if ppd_out<60 else cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REPLICATE)
    a=cv2.remap(fgs,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    inside=ok&(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
    edge=np.clip(np.minimum(np.minimum(mx,w-1-mx)/w,np.minimum(my,h-1-my)/h)/0.05,0,1)
    a=a*inside*edge
    return col,a,Bg,Eg
# low-res (band) render for gain matching and baking
col,a,Bg,Eg=render(ppd)
c0=int(round(B0*ppd)); r0=int(round((TOP-E1)*ppd)); base=img[r0:r0+col.shape[0], c0:c0+col.shape[1]]
m=(a>0.9)&(fg.max()>0)
# gain from grass overlap (people have no counterpart in the pano)
gm=(a>0.95)
gain=np.array([np.median(base[...,k][gm])/max(1,np.median(col[...,k][gm])) for k in range(3)])
print('crew gain BGR',gain)
out=base*(1-a[...,None])+col*gain*a[...,None]
img[r0:r0+col.shape[0], c0:c0+col.shape[1]]=out
cv2.imwrite(S+'/band_final.png',np.clip(img,0,255).astype(np.uint8))
# high-res patch at the photo's native angular resolution
ppd_hi=fp*math.pi/180
colh,ah,_,_=render(ppd_hi)
rgba=np.dstack([np.clip(colh*gain,0,255),np.clip(ah*255,0,255)]).astype(np.uint8)
cv2.imwrite(S+'/crew_patch.png',rgba,[cv2.IMWRITE_PNG_COMPRESSION,9])
np.save(S+'/crew_patch_meta.npy',np.array([B0,B1,E0,E1,ppd_hi]))
print('patch',rgba.shape,'ppd_hi %.1f'%ppd_hi)
