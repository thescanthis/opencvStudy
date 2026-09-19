import os, fitz
from extract_main_image import extract_main

base = "../../TestData"
folders = sorted(os.listdir(base))
done = {"50073721_5995-37-501-9359","50073731_5995-37-501-9361","60309822_5995-37-502-9688",
        "A20016147_6150-37-520-5295","A20016148_6145-37-520-5297","A20016149_6145-37-520-5278"}
outdir = "out/batch1"
os.makedirs(outdir, exist_ok=True)
for f in folders:
    if f in done: continue
    pdf = os.path.join(base, f, "Cable.pdf")
    if not os.path.exists(pdf):
        print("MISSING", f); continue
    name = f.split("_")[0]
    outpng = os.path.join(outdir, f"{name}.png")
    try:
        w,h = extract_main(pdf, outpng)
    except Exception as e:
        print("ERR", f, e)
