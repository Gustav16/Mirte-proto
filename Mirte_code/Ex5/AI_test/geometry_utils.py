"""Exact planar segment/convex-rectangle distances (metres)."""
import numpy as np

def point_segment_distance(p,a,b):
    p,a,b=map(lambda q:np.asarray(q,float),(p,a,b));v=b-a;vv=float(v@v)
    if vv<1e-20:return float(np.linalg.norm(p-a))
    t=float(np.clip((p-a)@v/vv,0,1));return float(np.linalg.norm(p-(a+t*v)))

def cross(a,b):return float(a[0]*b[1]-a[1]*b[0])

def segment_distance(a,b,c,d):
    a,b,c,d=map(lambda p:np.asarray(p,float),(a,b,c,d))
    ab=b-a;cd=d-c;den=cross(ab,cd)
    if abs(den)>1e-12:
        t=cross(c-a,cd)/den;u=cross(c-a,ab)/den
        if -1e-12<=t<=1+1e-12 and -1e-12<=u<=1+1e-12:return 0.
    return min(point_segment_distance(a,c,d),point_segment_distance(b,c,d),
               point_segment_distance(c,a,b),point_segment_distance(d,a,b))

def contains(p,corners):
    signs=[cross(b-a,np.asarray(p)-a) for a,b in zip(corners,np.roll(corners,-1,axis=0))]
    return min(signs)>=-1e-12 or max(signs)<=1e-12

def rectangle_distance(a,b):
    ac,bc=np.asarray(a['corners']),np.asarray(b['corners'])
    if contains(ac[0],bc) or contains(bc[0],ac):return 0.
    return min(segment_distance(p,q,r,s) for p,q in zip(ac,np.roll(ac,-1,axis=0))
               for r,s in zip(bc,np.roll(bc,-1,axis=0)))

def segment_rectangle_distance(a,b,box):
    corners=np.asarray(box['corners'])
    if contains(a,corners) or contains(b,corners):return 0.
    return min(segment_distance(a,b,c,d) for c,d in zip(corners,np.roll(corners,-1,axis=0)))
