import cv2, numpy as np, math
S='work'
img=cv2.imread(S+'/band_base.png').astype(np.float32)
W,Hh,TOP,BOT=np.load(S+'/band_meta.npy'); W=int(W); Hh=int(Hh); ppd=W/360
q=np.load(S+'/rot_us.npy'); fp,cx,cy=q[:3]; Rm=q[3:].reshape(3,3)
ph=cv2.imread('Looking south I think with us in it.jpg').astype(np.float32); h,w=ph.shape[:2]
c0,c1=int(168*ppd),int(202*ppd); r0,r1=int((TOP-15)*ppd),int((TOP+10)*ppd)
B=np.radians((np.arange(c0,c1)+0.5)/ppd); E=np.radians(TOP-(np.arange(r0,r1)+0.5)/ppd)
Bg,Eg=np.meshgrid(B,E)
d=np.stack([np.sin(Bg)*np.cos(Eg),np.cos(Bg)*np.cos(Eg),np.sin(Eg)],-1)
cam=d@Rm      # = (R^T d)^T
z=-cam[...,2]
mx=(cx+fp*cam[...,0]/z).astype(np.float32); my=(cy-fp*cam[...,1]/z).astype(np.float32)
inside=(z>0)&(mx>=0)&(mx<w-1)&(my>=0)&(my<h-1)
warp=cv2.remap(ph,mx,my,cv2.INTER_AREA if False else cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
# feather: distance from photo border in photo pixels, normalised
dx=np.minimum(mx,w-1-mx)/w; dy=np.minimum(my,h-1-my)/h
fe=np.clip(np.minimum(dx/0.14,dy/0.14),0,1); fe=fe*fe*(3-2*fe); fe*=inside
base=img[r0:r1,c0:c1]
# low-frequency colour match (ratio of blurred images inside overlap)
sig=ppd*1.2
m=inside.astype(np.float32)
bw=cv2.GaussianBlur(warp*m[...,None],(0,0),sig); bb=cv2.GaussianBlur(base*m[...,None],(0,0),sig); bm=cv2.GaussianBlur(m,(0,0),sig)[...,None]
gain=(bb+1)/(bw+1)
gain=np.clip(gain,0.6,1.6)
matched=warp*gain
out=base*(1-fe[...,None])+matched*fe[...,None]
img[r0:r1,c0:c1]=out
cv2.imwrite(S+'/band_final.png',np.clip(img,0,255).astype(np.uint8))
# crops for checking
crop=np.clip(img[r0:r1,c0:c1],0,255).astype(np.uint8)
cv2.imwrite(S+'/st/ins.jpg',cv2.resize(crop,None,fx=0.6,fy=0.6))
cv2.imwrite(S+'/st/ins_base.jpg',cv2.resize(np.clip(base,0,255).astype(np.uint8),None,fx=0.6,fy=0.6))
# people location (photo px approx 1630,1810)
v=np.array([(1630-cx)/fp,-(1810-cy)/fp,-1.]); v/=np.linalg.norm(v); dw=Rm@v
print('people bearing %.2f elev %.2f'%(math.degrees(math.atan2(dw[0],dw[1]))%360, math.degrees(math.asin(dw[2]))))
pc=int(((math.degrees(math.atan2(dw[0],dw[1]))%360))*ppd)-c0; pr=int((TOP-math.degrees(math.asin(dw[2])))*ppd)-r0
cv2.imwrite(S+'/st/ins_zoom.jpg',crop[pr-160:pr+120,pc-260:pc+260])
