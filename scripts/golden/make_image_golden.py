"""Image preprocessing golden data for the Swift port, from synthetic images only.

1. JPEG decoding: synthetic JPEGs (baseline / progressive, 4:2:0 / 4:2:2 / 4:4:4 / grey, restart markers,
   odd and tiny sizes, CMYK for the ImageIO fallback) and the SHA-256 of PIL's decode (libjpeg-turbo).
2. Resizing: synthetic PNG/JPEG sources run through transformers' Gemma4ImageProcessor (torchvision backend,
   280 soft tokens), recording the target size and the SHA-256 of the resized uint8 canvas, and of PIL's
   decoded source.

Hashes instead of pixel dumps keep the fixtures small; any difference fails the test.

Usage (isolated venv with tokenizers transformers torch torchvision pillow numpy):
    python -I scripts/golden/make_image_golden.py Tests/Fixtures/images
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(94)


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array, np.uint8).tobytes()).hexdigest()


def picture(w, h, noise=40):
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255 // max(w - 1, 1)).astype(np.int32)
    g = (y * 255 // max(h - 1, 1)).astype(np.int32)
    b = (((x // 7 + y // 5) % 2) * 200 + 20).astype(np.int32)
    img = np.stack([r, g, b], -1)
    if noise:
        img = img + rng.integers(-noise, noise, (h, w, 3))
    img[h // 3: h // 2, w // 4: w // 2] = [250, 10, 10]  # hard colour edges
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")


# 1. JPEG decoder fixtures
jpeg = []
for w, h in [(97, 61), (33, 17), (17, 9), (1, 1), (3, 2), (301, 3), (5, 77), (64, 64)]:
    for sub in (0, 1, 2):
        for prog in (False, True):
            jpeg.append((f"rgb_{w}x{h}_s{sub}_{'p' if prog else 'b'}", picture(w, h), dict(subsampling=sub, progressive=prog, quality=87)))
jpeg.append(("rgb_q100_s2_p_opt", picture(120, 90), dict(subsampling=2, progressive=True, quality=100, optimize=True)))
jpeg.append(("rgb_q20_s2_b", picture(120, 90), dict(subsampling=2, quality=20)))
jpeg.append(("rgb_restart_blocks_s2_b", picture(101, 73), dict(subsampling=2, quality=85, restart_marker_blocks=5)))
jpeg.append(("rgb_restart_rows_s1_p", picture(101, 73), dict(subsampling=1, quality=85, progressive=True, restart_marker_rows=1)))
jpeg.append(("grey_b", picture(57, 29).convert("L"), dict(quality=90)))
jpeg.append(("grey_p", picture(57, 29).convert("L"), dict(quality=90, progressive=True)))
jpeg.append(("cmyk_b", picture(40, 30).convert("CMYK"), dict(quality=90)))
jpeg_records = []
for name, img, kw in jpeg:
    path = out / f"{name}.jpg"
    try:
        img.save(path, "JPEG", **kw)
    except OSError as e:  # libjpeg rejects some tiny progressive images
        print("skip", name, e, file=sys.stderr)
        path.unlink(missing_ok=True)
        continue
    dec = Image.open(path)
    dec.load()
    dec = dec.convert("RGB")
    jpeg_records.append({"file": path.name, "width": dec.size[0], "height": dec.size[1], "sha256": sha(np.asarray(dec))})

# 2. Resize fixtures through the official processor
from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor  # noqa: E402
import transformers  # noqa: E402
import torch  # noqa: E402
import torchvision  # noqa: E402

processor = Gemma4ImageProcessor()
sources = [
    ("up_320x240.png", picture(320, 240, noise=0), "PNG"),
    ("down_1600x1200.png", picture(1600, 1200, noise=0), "PNG"),
    ("tiny_1x1.png", picture(1, 1, noise=0), "PNG"),
    ("strip_3000x10.png", picture(3000, 10, noise=0), "PNG"),
    ("tall_10x3000.png", picture(10, 3000, noise=0), "PNG"),
    ("odd_47x1000.png", picture(47, 1000, noise=0), "PNG"),
    ("exact_960x672.png", picture(960, 672, noise=0), "PNG"),  # 20 x 14 cells = 280 tokens, no resize
    ("noise_97x61.png", picture(97, 61), "PNG"),
    ("noise_401x299.jpg", picture(401, 299), "JPEG"),
]
resize_records = []
for name, img, fmt in sources:
    path = out / name
    img.save(path, fmt, **({"quality": 87} if fmt == "JPEG" else {"optimize": True}))
    src = Image.open(path)
    src.load()
    src = src.convert("RGB")
    feats = processor(images=[src], return_tensors="pt")
    pv = feats["pixel_values"][0]
    pos = feats["image_position_ids"][0]
    real = pos[:, 0] >= 0
    pw, ph = int(pos[real, 0].max()) + 1, int(pos[real, 1].max()) + 1
    patches = pv[real].float().numpy()  # (ph * pw, 16 * 16 * 3) as (y, x, c)
    canvas = patches.reshape(ph, pw, 16, 16, 3).transpose(0, 2, 1, 3, 4).reshape(ph * 16, pw * 16, 3)
    u8 = np.round(canvas * 255.0)
    assert np.abs(canvas * 255.0 - u8).max() < 1e-3
    resize_records.append({
        "file": name, "source_width": src.size[0], "source_height": src.size[1],
        "source_sha256": sha(np.asarray(src)), "width": pw * 16, "height": ph * 16,
        "tokens": (pw // 3) * (ph // 3), "sha256": sha(u8),
    })

(out / "golden.json").write_text(json.dumps({
    "generator": f"transformers {transformers.__version__}, torch {torch.__version__}, torchvision {torchvision.__version__}, "
                 f"Pillow {Image.__version__}",
    "jpeg": jpeg_records, "resize": resize_records,
}, indent=1) + "\n")
print(len(jpeg_records), "jpeg,", len(resize_records), "resize fixtures", file=sys.stderr)
