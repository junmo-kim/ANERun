"""Print or verify Swift thresholds for round(127 * tanh(x)) in the pinned PyTorch runtime."""

import argparse
from pathlib import Path
import re

import numpy as np
import torch


def thresholds(quantize):
    codes = np.arange(127, dtype=np.int32)
    analytic = np.arctanh((codes.astype(np.float64) + 0.5) / 127)
    centers = analytic.astype(np.float32).view(np.uint32).astype(np.int64)
    bits = (centers[:, None] + np.arange(-24, 25)).astype(np.uint32)
    grid = torch.from_numpy(bits.view(np.float32))
    actual = quantize(grid).numpy().astype(np.int32)
    assert (actual[:, 0] == codes).all() and (actual[:, -1] == codes + 1).all()
    crossing = np.argmax(actual == codes[:, None] + 1, axis=1)
    cuts = bits[codes, crossing]
    assert (np.diff(cuts) > 0).all()
    assert np.array_equal(actual, np.searchsorted(cuts, bits, side="right"))
    assert np.array_equal(quantize(-grid).numpy(), -actual)
    # The scalar path and the vectorized path must agree at every boundary.
    for code, cut in enumerate(cuts):
        for bits, expected in ((cut - 1, code), (cut, code + 1)):
            scalar = np.array([bits], dtype=np.uint32).view(np.float32)
            assert quantize(torch.from_numpy(scalar)).item() == expected, hex(int(bits))
    return cuts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--check", type=Path, metavar="SWIFT_TABLE")
    args = cli.parse_args()
    if torch.__version__.split("+")[0] != "2.7.1":
        raise SystemExit("use PyTorch 2.7.1 to reproduce Swift's Float32 threshold table")
    torch.set_num_threads(1)
    cuts = list(thresholds(lambda x: torch.round(127 * torch.tanh(x))))
    if args.check:
        actual = [int(digits, 16) for digits in re.findall(r"0x[0-9a-f]{8}\b", args.check.read_text())]
        assert actual == cuts, "Swift table differs from the pinned quantizer"
        print(f"Verified {len(cuts)} Float32 thresholds")
    else:
        for start in range(0, len(cuts), 8):
            print("        " + ", ".join(f"0x{cut:08x}" for cut in cuts[start:start + 8]) + ",")


if __name__ == "__main__":
    main()
