R=''  # run from the repo root
import cv2, numpy as np, sys

p2=cv2.imread(R+'pano2.jpg',0); u=cv2.imread(R+'Looking south I think with us in it.jpg',0)
cl=cv2.createCLAHE(2.0,(8,8))
X0,Y0=8800,600; P=cl.apply(p2[Y0:1500,X0:10900].copy())
sift=cv2.SIFT_create(40000)
kp,dp=sift.detectAndCompute(P,None)
for s in [0.45,0.55,0.62,0.7]:
    U=cl.apply(cv2.resize(u[1200:,:],None,fx=s,fy=s,interpolation=cv2.INTER_AREA))
    ku,du=sift.detectAndCompute(U,None)
    m=cv2.BFMatcher().knnMatch(du,dp,k=2); g=[a for a,b in m if a.distance<0.8*b.distance]
    if len(g)<4: print(s,len(g)); continue
    A=np.float32([ku[a.queryIdx].pt for a in g])/s+[0,1200]; B=np.float32([kp[a.trainIdx].pt for a in g])+[X0,Y0]
    M,inl=cv2.estimateAffinePartial2D(A,B,method=cv2.RANSAC,ransacReprojThreshold=4)
    inl=inl.ravel()==1
    sc=np.hypot(M[0,0],M[1,0]); rot=np.degrees(np.arctan2(M[1,0],M[0,0]))
    print('s',s,'matches',len(g),'inl',inl.sum(),'scale %.3f rot %.2f'%(sc,rot),'photo y range',A[inl,1].min() if inl.any() else 0,A[inl,1].max() if inl.any() else 0)
    if inl.sum()>20: np.save('work/us_pts_%.2f.npy'%s,np.concatenate([A[inl],B[inl]],1))
