import os
from PIL import Image
src = "out/batch1"
dst = "out/preview1"
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    im = Image.open(os.path.join(src, f))
    w, h = im.size
    scale = 0.18
    im2 = im.resize((int(w*scale), int(h*scale)))
    im2.save(os.path.join(dst, f))
    print(f, im.size, "->", im2.size)
