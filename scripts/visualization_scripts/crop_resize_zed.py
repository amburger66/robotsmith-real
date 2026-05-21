"""Crop the ZED capture to the VLM-relevant top-right region and resize to 256x256.

Loads rgb.png / rgb.npy / depth_m.npy / xyz_m.npy / camera_intrinsics.npz from
the input directory, applies the same square crop to all of them, resizes to
256x256, updates the intrinsics for the crop+resize, and writes everything to
an output directory (default: <input>_256).

Defaults crop to the 620x620 square used for VLM query images
(x=380:1000, y=0:620). Override with --x0/--x1/--y0/--y1 if you want a different window.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    p.add_argument(
        "--input-dir",
        default="data/zed_captures/zed1_20260516_170417",
        help="Directory containing rgb.png/rgb.npy/depth_m.npy/xyz_m.npy/camera_intrinsics.npz",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Where to write cropped+resized outputs (default: <input-dir>_256)",
    )
    p.add_argument("--x0", type=int, default=380)
    p.add_argument("--x1", type=int, default=1000)
    p.add_argument("--y0", type=int, default=0)
    p.add_argument("--y1", type=int, default=620)
    p.add_argument("--size", type=int, default=256)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = args.input_dir
    out_dir = args.output_dir or f"{in_dir.rstrip(os.sep)}_{args.size}"
    os.makedirs(out_dir, exist_ok=True)

    x0, x1, y0, y1 = args.x0, args.x1, args.y0, args.y1
    crop_w, crop_h = x1 - x0, y1 - y0
    if crop_w != crop_h:
        print(f"[warn] crop is non-square ({crop_w}x{crop_h}); 256x256 resize will distort")
    S = args.size

    rgb_np = np.load(os.path.join(in_dir, "rgb.npy"))
    depth = np.load(os.path.join(in_dir, "initial_depth.npy"))
    xyz = np.load(os.path.join(in_dir, "xyz_m.npy"))
    intr = dict(np.load(os.path.join(in_dir, "camera_intrinsics.npz")))

    rgb_crop = rgb_np[y0:y1, x0:x1]
    depth_crop = depth[y0:y1, x0:x1]
    xyz_crop = xyz[y0:y1, x0:x1]

    # PIL handles RGB resize; for depth/xyz keep float32 + use BILINEAR.
    rgb_img = Image.fromarray(rgb_crop).resize((S, S), Image.BILINEAR)

    def resize_float(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            im = Image.fromarray(arr, mode="F").resize((S, S), Image.BILINEAR)
            return np.asarray(im, dtype=np.float32)
        # (H, W, 3)
        chans = [
            np.asarray(
                Image.fromarray(arr[..., c], mode="F").resize((S, S), Image.BILINEAR),
                dtype=np.float32,
            )
            for c in range(arr.shape[-1])
        ]
        return np.stack(chans, axis=-1)

    depth_out = resize_float(depth_crop)
    xyz_out = resize_float(xyz_crop)
    rgb_out = np.asarray(rgb_img, dtype=np.uint8)

    sx = S / crop_w
    sy = S / crop_h
    fx_new = float(intr["fx"]) * sx
    fy_new = float(intr["fy"]) * sy
    cx_new = (float(intr["cx"]) - x0) * sx
    cy_new = (float(intr["cy"]) - y0) * sy

    rgb_img.save(os.path.join(out_dir, "initial.png"))
    np.save(os.path.join(out_dir, "rgb.npy"), rgb_out)
    np.save(os.path.join(out_dir, "initial_depth.npy"), depth_out)
    np.save(os.path.join(out_dir, "xyz_m.npy"), xyz_out)
    np.savez(
        os.path.join(out_dir, "camera_intrinsics.npz"),
        fx=np.float32(fx_new),
        fy=np.float32(fy_new),
        cx=np.float32(cx_new),
        cy=np.float32(cy_new),
        width=np.int32(S),
        height=np.int32(S),
    )

    print(f"Wrote {out_dir}")
    print(f"  crop=[x {x0}:{x1}, y {y0}:{y1}] ({crop_w}x{crop_h}) -> {S}x{S}")
    print(
        f"  intrinsics: fx={fx_new:.2f} fy={fy_new:.2f} cx={cx_new:.2f} cy={cy_new:.2f}"
    )


if __name__ == "__main__":
    main()
