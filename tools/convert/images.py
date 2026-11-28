"""Collect the public evaluation images into <out_dir>: the images pinned in sources.json (fetched by `fetch.py data`)
plus synthetic images generated here. Usage: images.py <sources_dir> <out_dir>"""
import os
import shutil
import sys

import numpy as np
from PIL import Image, ImageDraw

PUBLIC = [  # files in sources_dir; origin and license in sources.json
    "china.jpg", "flower.jpg",                                   # scikit-learn sample images
    "test-1.jpeg", "llama1-banner.png", "matmul.png",            # llama.cpp
    "cat_with_sd_cpp_20184.png", "newton_0.jpg",                 # stable-diffusion.cpp
    "test_house_with_colors.png",                                # networkx test baseline (plot)
]


def synthetic(out):
    w, h = 900, 200
    g = np.zeros((h, w, 3), np.uint8)
    g[..., 0] = np.linspace(0, 255, w)[None, :]
    g[..., 1] = np.linspace(255, 0, h)[:, None]
    g[..., 2] = 128
    Image.fromarray(g).save(f"{out}/syn_wide_gradient.png")
    Image.fromarray(np.ascontiguousarray(g.transpose(1, 0, 2))).save(f"{out}/syn_tall_gradient.png")
    rng = np.random.default_rng(7)
    Image.fromarray(rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)).save(f"{out}/syn_noise.png")
    im = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(im)
    for i in range(8):
        d.rectangle([20 + 70 * i, 40 + 20 * i, 80 + 70 * i, 200 + 20 * i], fill=(30 * i, 200 - 20 * i, 90))
    d.ellipse([300, 250, 600, 460], fill=(240, 180, 20), outline="black", width=5)
    d.text((30, 400), "ANEEmbed synthetic test image 123", fill="black")
    im.save(f"{out}/syn_shapes_text.png")
    Image.new("RGB", (40, 30), (200, 30, 30)).save(f"{out}/syn_tiny.png")


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for name in PUBLIC:
        shutil.copyfile(os.path.join(src, name), os.path.join(out, name))
    synthetic(out)
    print(len(os.listdir(out)), "images in", out)
