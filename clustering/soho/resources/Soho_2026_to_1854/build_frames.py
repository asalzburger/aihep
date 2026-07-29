import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
U='/sessions/ecstatic-serene-ptolemy/mnt/uploads/'
CW,CH=1264,832; CROP=32
sx,sy=0.691,0.767; tx,ty=-73.1,-149.5+28
photo=Image.open(U+'Soho_2026.png').convert('RGB')
eng=Image.open(U+'Soho_1854.png').convert('RGB')
a,b=1/sx,1/sy
pw=photo.transform((CW,CH), Image.AFFINE, (a,0,-tx*a,0,b,-ty*b), resample=Image.BICUBIC)
pw=pw.crop((0,CROP,CW,CH)); ew=eng.crop((0,CROP,CW,CH))
pw.save('photo_aligned.png'); ew.save('eng_aligned.png')
Image.blend(pw,ew,0.5).save('preview_final.png')
print(pw.size)
