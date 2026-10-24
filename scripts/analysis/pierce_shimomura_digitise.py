#!/usr/bin/env python
r"""List the black strokes in Pierce-Shimomura et al. 1999's Figure 8, as rates, at chosen dC/dt.

The digitisation behind ``data/chemotaxis/pierce_shimomura1999_pirouettes.json``. The paper's PDF
(PMC6782915, ``ns009557.pdf``, (c) Society for Neuroscience) is not vendored; download it by browser.
Page 8 is rendered at 500 dpi, panel A (biotin) and B (NH4Cl) clipped, and the axes calibrated from
the tick marks' pixel positions. For each requested dC/dt the script prints every thin black stroke
in that pixel column as a rate, with its thickness in pixels; the curve is read from these by
setting aside the dashed spontaneous-rate line (about 0.029) and the point-count histogram (the
lowest strokes).

Usage::

    uv run --with pymupdf --with pillow python scripts/analysis/pierce_shimomura_digitise.py \
        --pdf build/ns009557.pdf --panel A -0.18 -0.08 -0.04 0 0.04 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DESCRIPTION = "List Figure 8's black strokes as rates at chosen dC/dt."
CLIPS = {"A": (55, 55, 300, 215), "B": (55, 215, 300, 380)}  # PDF points, page 8
DPI = 500
# Pixel positions of the tick marks at 500 dpi, found from the rendered axes.
CALIBRATION = {
    "A": {
        "rate_zero_row": 970,
        "rate_014_row": 230,
        "dcdt": [(316, -0.2), (550, -0.1), (787, 0.0), (1022, 0.1)],
    },
    "B": {
        "rate_zero_row": 1020,
        "rate_014_row": 280,
        "dcdt": [(423, -0.1), (772, 0.0), (1120, 0.1)],
    },
}
BLACK = 80


def strokes(pdf: Path, panel: str, dcdt: list[float]) -> dict[float, list[tuple[float, int]]]:
    """Return, for each dC/dt, every black stroke's rate and thickness in that pixel column."""
    import pymupdf  # pyright: ignore[reportMissingImports]
    from PIL import Image

    page = pymupdf.open(pdf)[7]
    pixmap = page.get_pixmap(dpi=DPI, clip=pymupdf.Rect(*CLIPS[panel]))
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples).convert("L")
    black = np.asarray(image).astype(int) < BLACK
    cal = CALIBRATION[panel]
    slope, intercept = np.polyfit([v for _, v in cal["dcdt"]], [p for p, _ in cal["dcdt"]], 1)
    per_rate = (cal["rate_zero_row"] - cal["rate_014_row"]) / 0.14
    out: dict[float, list[tuple[float, int]]] = {}
    for x in dcdt:
        column = round(slope * x + intercept)
        rows = np.flatnonzero(black[:, column - 1 : column + 2].any(axis=1))
        runs: list[list[int]] = []
        for r in rows:
            if runs and r - runs[-1][-1] <= 1:
                runs[-1].append(int(r))
            else:
                runs.append([int(r)])
        out[x] = [
            ((cal["rate_zero_row"] - float(np.mean(r))) / per_rate, len(r))
            for r in runs
            if cal["rate_014_row"] - 10 < np.mean(r) < cal["rate_zero_row"] - 3
        ]
    return out


def main() -> int:
    """CLI: print the strokes at each requested dC/dt."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--panel", choices=tuple(CLIPS), required=True)
    parser.add_argument("dcdt", type=float, nargs="+")
    args = parser.parse_args()
    for x, found in strokes(args.pdf, args.panel, args.dcdt).items():
        print(f"{x:+.3f}", " ".join(f"{rate:.4f}({width})" for rate, width in found))
    return 0


if __name__ == "__main__":
    sys.exit(main())
