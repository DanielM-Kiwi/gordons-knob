# band (beta deg, eps rad) -> pano2 pixel, with the loop-closure warp spread over the circle,
# plus the matching "old" direction used to sample pano1 / Motueka / crew layers.
exec(open('tools/stitch/geo.py').read())
_sm=np.load('work/seam_map.npy')
def _feats(x,y): x=x/1000; y=y/1000; return np.stack([np.ones_like(x),x,y,x*y,x*x,y*y,y**3],-1)
def band_to_p2(beta,eps):
    x=x_of_brg(beta); y=hor_of_x(x)-f*np.tan(eps)
    s=x/C; ys=y-K*s
    F=_feats(np.zeros_like(ys),ys)
    ex=F@_sm[0]-C; ey=F@_sm[1]-ys-K
    return x+s*ex, y+s*ey
def band_to_olddir(beta,eps):
    x2,y2=band_to_p2(beta,eps); b,e=pano_to_dir(x2,y2); return b,e,x2,y2
