"""Image input formats: the same public images as other files, checked against the official fp32 path.

    python -I formats.py convert <images_dir> <out_dir>
        Writes every image of images_dir (tools/convert images.py) as: PNG (control), WebP lossy (quality 90) and
        lossless, HEIC (`sips`, macOS's ImageIO encoder), GIF (256-colour adaptive palette), TIFF, BMP, and a
        3-frame animated GIF / WebP / PNG (APNG) whose first frame is the image (frames 2-3: inverted, flipped).
        Sources are read with transformers' load_image (RGB, EXIF orientation applied).
    python -I formats.py reference --hf-dir DIR <out_dir> <reference.json>
        Official fp32 reference for every file: transformers.image_utils.load_image(path) (PIL; HEIC through the
        pillow-heif plugin, which must be installed) -> the model's processor -> EmbeddingGemma2Model fp32 on the
        CPU, mean pooling, L2. Groups "format_<ext>" in compare.py's format: send them to `aneembed serve` with
        `compare.py --reference reference.json --images <out_dir>`.
    python -I formats.py pixels <out_dir> <decodedump>
        Pixel-level decode differences: aneembed's decoder (ImagePrep.decode, built into the `decodedump` helper:
        swiftc -parse-as-library -module-name EmbedCore Sources/EmbedCore/Image/*.swift <main.swift>) against PIL's
        load_image, per format: share of differing channel values, mean and max absolute difference.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from PIL import Image

FORMATS = ("png", "webp_lossy", "webp_lossless", "heic", "gif", "tiff", "bmp", "anim_gif", "anim_webp", "anim_png")
EXT = {"webp_lossy": "lossy.webp", "webp_lossless": "lossless.webp", "anim_gif": "anim.gif", "anim_webp": "anim.webp",
       "anim_png": "anim.png"}


def fmt_of(name):
    for f, ext in EXT.items():
        if name.endswith("." + ext):
            return f
    return name.rsplit(".", 1)[1]


def heif():
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
        return pillow_heif.__version__
    except ImportError:
        return None


def convert(src, out):
    from transformers.image_utils import load_image
    os.makedirs(out, exist_ok=True)
    for path in sorted(glob.glob(f"{src}/*")):
        stem = os.path.splitext(os.path.basename(path))[0]
        img = load_image(path)
        img.save(f"{out}/{stem}.png")
        img.save(f"{out}/{stem}.lossy.webp", quality=90, method=6)
        img.save(f"{out}/{stem}.lossless.webp", lossless=True)
        img.save(f"{out}/{stem}.gif")
        img.save(f"{out}/{stem}.tiff")
        img.save(f"{out}/{stem}.bmp")
        subprocess.run(["sips", "-s", "format", "heic", f"{out}/{stem}.png", "--out", f"{out}/{stem}.heic"],
                       check=True, capture_output=True)
        frames = [img, Image.eval(img, lambda v: 255 - v), img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)]
        frames[0].save(f"{out}/{stem}.anim.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
        frames[0].save(f"{out}/{stem}.anim.webp", save_all=True, append_images=frames[1:], duration=100, loop=0,
                       lossless=True)
        frames[0].save(f"{out}/{stem}.anim.png", save_all=True, append_images=frames[1:], duration=100, loop=0)
        print(stem, img.size, flush=True)


def reference(hf_dir, out, target):
    import torch
    from transformers import EmbeddingGemma2Model, AutoProcessor
    from transformers.image_utils import load_image
    plugin = heif()
    torch.set_num_threads(8)
    model = EmbeddingGemma2Model.from_pretrained(hf_dir, dtype=torch.float32, use_safetensors=True, local_files_only=True).eval()
    proc = AutoProcessor.from_pretrained(hf_dir, local_files_only=True, trust_remote_code=False)
    cases = []
    for path in sorted(glob.glob(f"{out}/*")):
        name = os.path.basename(path)
        if name.endswith(".rgb") or (fmt_of(name) == "heic" and not plugin):
            continue
        x = proc(images=[load_image(path)], return_tensors="pt")
        with torch.no_grad():
            h = model(**{k: x[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_position_ids")})
        e = h.last_hidden_state[0].mean(0)
        cases.append({"group": f"format_{fmt_of(name)}", "kind": "image", "images": [name],
                      "tokens": [int(x["input_ids"].shape[1])], "reference": [(e / e.norm()).numpy().round(7).tolist()]})
        print(name, cases[-1]["tokens"], flush=True)
    import PIL
    json.dump({"format": "aneembed-accuracy/1", "precision": "fp32", "device": "cpu", "loader": "transformers load_image",
               "pillow": PIL.__version__, "pillow_heif": plugin, "torch": torch.__version__, "cases": cases}, open(target, "w"))


def pixels(out, tool):
    from transformers.image_utils import load_image
    heif()
    files = [p for p in sorted(glob.glob(f"{out}/*")) if not p.endswith(".rgb")]
    dims = {}
    for line in subprocess.run([tool] + files, capture_output=True, text=True, check=True).stdout.splitlines():
        parts = line.rsplit(" ", 2)
        dims[parts[0]] = (int(parts[1]), int(parts[2])) if parts[1] != "ERROR" else None
    stats = {}
    for p in files:
        f = fmt_of(os.path.basename(p))
        ref = np.asarray(load_image(p), np.int16)
        if dims.get(p) is None:
            stats.setdefault(f, []).append(None)
            continue
        w, h = dims[p]
        ours = np.fromfile(p + ".rgb", np.uint8).reshape(h, w, 3).astype(np.int16)
        if ours.shape != ref.shape:
            stats.setdefault(f, []).append(("shape", ours.shape, ref.shape))
            continue
        d = np.abs(ours - ref)
        stats.setdefault(f, []).append((float((d > 0).mean()), float(d.mean()), int(d.max())))
    for f in FORMATS:
        v = stats.get(f, [])
        ok = [x for x in v if isinstance(x, tuple) and len(x) == 3 and not isinstance(x[0], str)]
        bad = len(v) - len(ok)
        if not ok:
            print(f"{f:14s} files={len(v)} not comparable={bad}")
            continue
        print(f"{f:14s} files={len(v)} differing={max(x[0] for x in ok) * 100:.2f}% (max over files) "
              f"mean|d|={max(x[1] for x in ok):.3f} max|d|={max(x[2] for x in ok)}" + (f" not comparable={bad}" if bad else ""))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "convert":
        convert(sys.argv[2], sys.argv[3])
    elif cmd == "reference":
        reference(sys.argv[3], sys.argv[4], sys.argv[5]) if sys.argv[2] == "--hf-dir" else sys.exit("usage")
    elif cmd == "pixels":
        pixels(sys.argv[2], sys.argv[3])
