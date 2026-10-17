# true skyline from the 1685 trig, hi-res near field + z10 far field
import numpy as np, math, cv2
exec(open('tools/stitch/demsky.py').read().split("z0=elev")[0])   # z10 mosaic elev(e,n) about GRID_ORIGIN (-41.6039,172.9369)
S='work'
loc=np.load(S+'/localdem.npy'); LCELL=14.0; LHALF=7000.0     # centred at (-41.60794,172.94411)
LAT_T,LON_T=-41.60985,172.94482                                 # trig
mlon0=111320*math.cos(math.radians(-41.6039))
# offsets of the trig relative to each grid's origin
tE0=(LON_T-172.9369)*mlon0; tN0=(LAT_T+41.6039)*111320
tEL=(LON_T-172.94411)*mlon0; tNL=(LAT_T+41.60794)*111320
def elev_hi(e,n):   # e,n relative to the trig
    fx=(e+tEL+LHALF)/LCELL-0.5; fy=(LHALF-(n+tNL))/LCELL-0.5
    return cv2.remap(loc,fx.astype(np.float32),fy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=np.nan)
Z_EYE=1685.0+1.6
B=np.arange(0,360,0.05); br=np.radians(B)
dn=np.concatenate([np.arange(15,300,3),np.arange(300,6500,12)]); df=np.geomspace(6500,155000,1500)
En=np.sin(br)[:,None]*dn; Nn=np.cos(br)[:,None]*dn
zn=elev_hi(En,Nn)
# DEM summit is ~13 m low (smoothing): scale near-field relief so the trig reads its surveyed height
zt=elev_hi(np.zeros((1,1)),np.zeros((1,1)))[0,0]
Ef=np.sin(br)[:,None]*df; Nf=np.cos(br)[:,None]*df
zf=elev(Ef+tE0,Nf+tN0)
def ang(z,d): return np.degrees(np.arctan2(z-Z_EYE-d*d/(2*6371000)*0.87,d))
an=ang(zn+(1685-zt),dn); af=ang(zf,df)
sky=np.maximum(np.nanmax(an,1),af.max(1))
dist=np.where(np.nanmax(an,1)>=af.max(1), dn[np.nanargmax(an,1)], df[af.argmax(1)])
np.save(S+'/sky_trig.npy',np.stack([B,sky,dist]))
print('trig DEM height %.0f; skyline %.2f..%.2f deg'%(zt,sky.min(),sky.max()))
