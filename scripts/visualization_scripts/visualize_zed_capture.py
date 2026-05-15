import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from robo_utils.visualization.plotting import plot_pcd


def load_zed_capture(
    capture_dir: Path,
    min_depth_m: float,
    max_depth_m: float,
    max_distance_m: float,
):
    xyz = np.load(capture_dir / "xyz_m.npy")
    rgb = np.load(capture_dir / "rgb.npy")
    depth = np.load(capture_dir / "depth_m.npy")

    if (
        xyz.shape[:2] != rgb.shape[:2]
        or xyz.shape[:2] != depth.shape
        or xyz.shape[-1] != 3
        or rgb.shape[-1] != 3
    ):
        raise ValueError(
            f"Expected xyz HxWx3, rgb HxWx3, and depth HxW with matching image "
            f"size; got xyz={xyz.shape}, rgb={rgb.shape}, depth={depth.shape}"
        )

    points = xyz.reshape(-1, 3)
    colors = rgb.reshape(-1, 3).astype(np.float32) / 255.0
    depths = depth.reshape(-1)

    mask = np.isfinite(points).all(axis=1)
    mask &= np.isfinite(depths)
    mask &= depths >= min_depth_m
    if max_depth_m > 0.0:
        mask &= depths <= max_depth_m
    if max_distance_m > 0.0:
        mask &= np.linalg.norm(points, axis=1) <= max_distance_m

    return points[mask], colors[mask]


def downsample(points, colors, voxel_size_m: float):
    if voxel_size_m <= 0.0:
        return points, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = pcd.voxel_down_sample(voxel_size_m)

    return np.asarray(pcd.points), np.asarray(pcd.colors)


def save_ply(path: Path, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(path), pcd)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize a saved ZED capture as an RGB point cloud."
    )
    parser.add_argument(
        "capture_dir",
        type=Path,
        nargs="?",
        default=Path("data/zed_captures/zed1_20260515_093719"),
    )
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=0.0)
    parser.add_argument("--max-distance-m", type=float, default=1.5)
    parser.add_argument("--voxel-size-m", type=float, default=0.005)
    parser.add_argument("--save-ply", type=Path, default=None)
    parser.add_argument("--no-view", action="store_true")
    args = parser.parse_args()

    points, colors = load_zed_capture(
        args.capture_dir,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        max_distance_m=args.max_distance_m,
    )
    points, colors = downsample(points, colors, args.voxel_size_m)

    print(f"Loaded {len(points)} valid points from {args.capture_dir}")
    print(f"Bounds min: {points.min(axis=0)}")
    print(f"Bounds max: {points.max(axis=0)}")

    if args.save_ply is not None:
        args.save_ply.parent.mkdir(parents=True, exist_ok=True)
        save_ply(args.save_ply, points, colors)
        print(f"Saved point cloud to {args.save_ply}")

    if not args.no_view:
        plot_pcd(points, colors=colors, base_frame=True)


if __name__ == "__main__":
    main()
