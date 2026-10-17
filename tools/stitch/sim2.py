R=''  # run from the repo root
import cv2, numpy as np, sys

name,X0,X1,s,out=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
p2=cv2.imread(R+'pano2.jpg',0); u=cv2.imread(R+name,0)
cl=cv2.createCLAHE(2.0,(8,8))
P=cl.apply(p2[:, X0:X1].copy())
sift=cv2.SIFT_create(60000)
kp,dp=sift.detectAndCompute(P,None)
U=cl.apply(cv2.resize(u,None,fx=s,fy=s,interpolation=cv2.INTER_AREA) if s<1 else u)
ku,du=sift.detectAndCompute(U,None)
m=cv2.BFMatcher().knnMatch(du,dp,k=2); g=[a for a,b in m if a.distance<0.8*b.distance]
A=np.float32([ku[a.queryIdx].pt for a in g])/min(s,1); B=np.float32([kp[a.trainIdx].pt for a in g])+[X0,0]
M,inl=cv2.estimateAffinePartial2D(A,B,method=cv2.RANSAC,ransacReprojThreshold=5); inl=inl.ravel()==1
print('matches',len(g),'inl',inl.sum(),'scale %.3f'%np.hypot(M[0,0],M[1,0]),'photo x',A[inl,0].min(),A[inl,0].max(),'y',A[inl,1].min(),A[inl,1].max(),'pano x',B[inl,0].min(),B[inl,0].max())
# keep wider inlier set using homography (wide-angle)
Hm,inl2=cv2.findHomography(A,B,cv2.RANSAC,8); inl2=inl2.ravel()==1
print('homog inl',inl2.sum())
np.save(out,np.concatenate([A[inl2],B[inl2]],1))
