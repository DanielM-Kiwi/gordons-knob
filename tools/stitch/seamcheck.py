import cv2, numpy as np
from scipy.ndimage import median_filter
a=cv2.imread('work/band_final.png').astype(np.float32); H,W=a.shape[:2]; ppd=W/360; TOP=36
g=cv2.cvtColor(a.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(np.float32)
def rows(e0,e1): return int((TOP-e0)*ppd),int((TOP-e1)*ppd)
# vertical seams: column-to-column step, over ground+scenery rows (+5..-45 deg)
r0,r1=rows(5,-45)
dc=np.abs(np.diff(np.concatenate([g[r0:r1],g[r0:r1,:1]],1),axis=1)).mean(0)      # W values, wraps
ref=median_filter(dc,size=61,mode='wrap')
ratio=dc/np.maximum(ref,1e-3)
top=np.argsort(ratio)[::-1][:6]
print('vertical: median step %.2f; worst columns (bearing, ratio):'%np.median(dc),[(round(c/ppd,1),round(float(ratio[c]),2)) for c in top])
# named seams
for name,b in [('pano2 loop seam',347.3),('pano1 start',269.7),('pano1 end',116.9),('crew photo L',171.5),('crew photo R',197.7)]:
    c=int(b*ppd); w=int(1.5*ppd); print('  %-16s max ratio within +-1.5deg: %.2f'%(name,ratio[np.arange(c-w,c+w)%W].max()))
# horizontal seams: row-to-row step per row, averaged over all columns
dr=np.abs(np.diff(g,axis=0)).mean(1); refr=median_filter(dr,size=41,mode='nearest'); rr=dr/np.maximum(refr,1e-3)
for name,e in [('sky synth start',11.7),('p2 top edge',13.7),('p2 bottom edge',-25.3),('band limit',-50)]:
    r=int((TOP-e)*ppd); w=int(1.5*ppd); print('  %-16s max row ratio within +-1.5deg: %.2f'%(name,rr[max(0,r-w):r+w].max()))
print('worst rows (elev, ratio):',[(round(TOP-r/ppd,1),round(float(rr[r]),2)) for r in np.argsort(rr)[::-1][:5]])
