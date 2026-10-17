import cv2, numpy as np
S='work'; ppd=12288/360; TOP=36
al=np.load(S+'/alpha.npy').astype(np.float32); H,W=al.shape
rowel=TOP-(np.arange(H)+0.5)/ppd
# synthetic ground visible for a uniform lowest-view limit (exact, every pixel)
for L in (-26,-25,-24.5,-24):
    rows=(rowel>=L)&(rowel<0); syn=1-al[rows]
    print('lowest view %.1f: max synthetic weight in any visible pixel %.4f, mean %.5f%%'%(L,syn.max(),100*syn.mean()))
# every blend zone, full res: gradient energy inside transition zones vs the real image just outside them
g=cv2.cvtColor(cv2.imread(S+'/band_final.png'),cv2.COLOR_BGR2GRAY).astype(np.float32)
gx=np.abs(np.diff(g,axis=1,append=g[:,:1])); gy=np.abs(np.diff(g,axis=0,append=g[-1:]))
grad=gx+gy
vis=(rowel>=-24.5)[:,None]&np.ones((1,W),bool)
zone=(al>0.02)&(al<0.98)&vis
# also the crew-photo feather zone
q=np.load(S+'/rot_us.npy')
tiles=[]
for c0 in range(0,W,256):
  for r0 in range(0,H,256):
    z=zone[r0:r0+256,c0:c0+256]
    if z.sum()<500: continue
    ring=cv2.dilate(z.astype(np.uint8),np.ones((61,61),np.uint8)).astype(bool)&~z&vis[r0:r0+256,c0:c0+256]
    if ring.sum()<500: continue
    G=grad[r0:r0+256,c0:c0+256]; tiles.append((G[z].mean()/G[ring].mean(),c0/ppd,TOP-r0/ppd))
tiles.sort(reverse=True)
print('blend-zone tiles checked: %d; zone/neighbour gradient ratio max %.2f (bearing %.0f, elev %.0f), median %.2f'%(len(tiles),tiles[0][0],tiles[0][1],tiles[0][2],np.median([t[0] for t in tiles])))
print('min ratio %.2f at bearing %.0f elev %.0f'%(tiles[-1][0],tiles[-1][1],tiles[-1][2]))
