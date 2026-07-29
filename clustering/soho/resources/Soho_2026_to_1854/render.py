import numpy as np, os
from PIL import Image, ImageFilter

P=np.asarray(Image.open('photo_aligned.png').convert('RGB'),np.float32)
E=np.asarray(Image.open('eng_aligned.png').convert('RGB'),np.float32)
H,W,_=P.shape
lum=(0.299*P[...,0]+0.587*P[...,1]+0.114*P[...,2])
mono=Image.fromarray(np.clip(lum,0,255).astype(np.uint8))
mono=mono.filter(ImageFilter.UnsharpMask(radius=2,percent=110,threshold=2))
M=np.asarray(mono,np.float32)
M=np.clip((M-128)*1.22+132,0,255)          # contrast/print look
M=np.repeat(M[...,None],3,axis=2)

yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
u=(xx/W)*0.62+(yy/H)*0.38                  # diagonal sweep coordinate
FE=0.45                                     # feather width

def ss(x):
    x=np.clip(x,0,1); return x*x*(3-2*x)

FPS=30; DUR=9.0; N=int(FPS*DUR)
os.makedirs('frames',exist_ok=True)
for i in range(N):
    t=i/(N-1)
    d=ss((t-0.14)/0.24)                     # desaturation ramp
    g=ss((t-0.28)/0.42)                     # dissolve progress
    A=P*(1-d)+M*d
    w=np.clip((g*(1+FE)-u)/FE,0,1)[...,None]
    out=A*(1-w)+E*w
    edge=np.exp(-((g*(1+FE)-u)/(FE*0.30))**2)*(1-abs(2*t-1))**0.5
    out=np.clip(out+edge[...,None]*26,0,255)
    Image.fromarray(out.astype(np.uint8)).save('frames/f%04d.png'%i)
print('frames',N)
