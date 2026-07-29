import numpy as np
from PIL import Image
U='/sessions/ecstatic-serene-ptolemy/mnt/uploads/'

# correspondences in 1000-wide grid coords: (x2026,y2026) <-> (x1854,y1854)
pairs=[((215,105),(358,150)),
       ((575,455),(760,600)),
       ((245,265),(290,300)),
       ((168,170),(198,180)),
       ((470,180),(505,195))]
S26=2.6; S18=1.264
A=[];B=[]
for (p,q) in pairs:
    A.append([p[0]*S26,p[1]*S26,1]); B.append([q[0]*S18,q[1]*S18])
A=np.array(A,float); B=np.array(B,float)
M,res,rk,sv=np.linalg.lstsq(A,B,rcond=None)   # 3x2 : B = A @ M
M=M.T  # 2x3 mapping 2026 -> 1854
print("M=",M)
for (p,q) in pairs:
    v=M@np.array([p[0]*S26,p[1]*S26,1.0])
    print(p,"->",np.round(v,1),"target",(q[0]*S18,q[1]*S18))
np.save('M.npy',M)
