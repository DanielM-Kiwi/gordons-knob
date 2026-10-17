import cv2, numpy as np
S='work'
C=18032; sc=0.25
s=cv2.resize(cv2.imread('pano2.jpg')[:, :C],None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA).astype(np.float32)
H,W=s.shape[:2]
r=cv2.GaussianBlur(s[...,2],(3,3),0)
cm=np.maximum.accumulate(r,axis=0)
land=(r<cm-20)
land[:40]=False
first=np.argmax(land,axis=0).astype(np.float32)
first=cv2.medianBlur(first.reshape(1,-1),3).ravel()
np.save(S+'/skyline_px.npy',first/sc)
vis=s.astype(np.uint8).copy()
for x in range(W):
    y=int(first[x]); vis[max(0,y-1):y+1,x]=(0,0,255)
cv2.imwrite(S+'/st/skyline_a.jpg',vis[60:330,:W//2]); cv2.imwrite(S+'/st/skyline_b.jpg',vis[60:330,W//2:])
print(first.min(),first.max())
