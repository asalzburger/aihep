import numpy as np
from PIL import Image
U='/sessions/ecstatic-serene-ptolemy/mnt/uploads/'
W,H=1264,832
sx,sy=0.691,0.767; tx,ty=-73.1,-149.5
# canvas->source inverse
a=1/sx; b=1/sy
photo=Image.open(U+'Soho_2026.png').convert('RGB')
# PIL AFFINE: source = (a*x + b*y + c, d*x + e*y + f)
warped=photo.transform((W,H), Image.AFFINE, (a,0,-tx*a, 0,b,-ty*b), resample=Image.BICUBIC)
warped.save('photo_aligned.png')
eng=Image.open(U+'Soho_1854.png').convert('RGB'); eng.save('eng_aligned.png')
Image.blend(warped,eng,0.5).save('preview_blend.png')
