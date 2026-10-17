import cv2, numpy as np
R=''  # run from the repo root
C=18032.0
sc=0.5
p1=cv2.resize(cv2.imread(R+'pano1.jpg',0),None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
p2=cv2.resize(cv2.imread(R+'pano2.jpg',0),None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
p2w=np.concatenate([p2,p2[:, :int(5000*sc)]],1)   # allow wrap
sift=cv2.SIFT_create(60000)
A=[];B=[]
for x0 in range(0,p1.shape[1],int(1500*sc)):
    t1=p1[:,x0:x0+int(1800*sc)]
    k1,d1=sift.detectAndCompute(t1,None)
    gx=int(((x0/sc-3770)%C)*sc)   # predicted p2 x
    lo=max(0,gx-int(900*sc)); hi=gx+int(2700*sc)
    k2,d2=sift.detectAndCompute(p2w[:,lo:hi],None)
    if d1 is None or d2 is None: continue
    m=cv2.BFMatcher().knnMatch(d1,d2,k=2); g=[a for a,b in m if a.distance<0.75*b.distance]
    if len(g)<8: continue
    a=np.float32([k1[q.queryIdx].pt for q in g])+[x0,0]; b=np.float32([k2[q.trainIdx].pt for q in g])+[lo,0]
    M,inl=cv2.estimateAffine2D(a,b,method=cv2.RANSAC,ransacReprojThreshold=3); inl=inl.ravel()==1
    A.append(a[inl]/sc); B.append(b[inl]/sc); print(x0/sc,len(g),inl.sum(),flush=True)
A=np.concatenate(A); B=np.concatenate(B)
np.save('work/p1p2.npy',np.concatenate([A,B],1)); print('total',len(A))
