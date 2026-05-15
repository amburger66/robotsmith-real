"""Execute a VLM waypoint plan on the real Franka using a ZED capture.

The VLM plan is expected to contain waypoints of the form:
    [x_norm, y_norm, dx, dy, dz, gripper_action]

The normalized image coordinates are interpreted on the 256x256 image produced
by scripts/visualization_scripts/crop_resize_zed.py. This script recreates the
same crop+resize for rgb/depth/xyz, transforms camera-frame targets into the
Franka base frame, writes dry-run visualizations, plans all segments, and only
moves the robot when --execute is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROBOTSMITH_ROOT = Path(__file__).resolve().parents[1]
# Let this script run both from the repository root and after editable installs.
for path in (ROBOTSMITH_ROOT, ROBOTSMITH_ROOT / "robo_utils"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


DEFAULT_SCENE_DIR = ROBOTSMITH_ROOT / "data/task08_cutting/vlm_traj_queries/cutter"
DEFAULT_EXTRINSICS = ROBOTSMITH_ROOT / "data/calibration/eye_to_hand/cam0_calibration.npz"
DEFAULT_ZED_ROOT = ROBOTSMITH_ROOT / "data/zed_captures"
DEFAULT_CROP = (380, 1000, 0, 620)
DEFAULT_GRIPPER_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

GRIPPER_CLOSE = 0
GRIPPER_OPEN = 1


@dataclass
class TargetRecord:
    index: int
    source_pixel: list[int]
    normalized_xy: list[float]
    offset_robot_m: list[float]
    camera_point_m: list[float] | None
    base_anchor_m: list[float] | None
    target_position_unclipped_m: list[float] | None
    target_position_m: list[float] | None
    target_pose_wxyz: list[float] | None
    gripper_action: int
    clipped_delta_m: float
    used_fallback_backprojection: bool
    valid: bool
    error: str | None = None
    planning_success: bool | None = None


def resolve_path(path: str | os.PathLike, *, base: Path = ROBOTSMITH_ROOT) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / p).resolve()


def parse_quat(raw: str) -> np.ndarray:
    values = [float(v.strip()) for v in raw.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--gripper-quat must contain 4 comma-separated values")
    quat = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-8:
        raise argparse.ArgumentTypeError("--gripper-quat must be a nonzero quaternion")
    return quat / norm


def latest_zed_capture(root: Path = DEFAULT_ZED_ROOT) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"ZED capture root does not exist: {root}")
    candidates = sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and p.name.startswith("zed")
        and not p.name.endswith("_256")
        and (p / "rgb.npy").is_file()
        and (p / "depth_m.npy").is_file()
        and (p / "xyz_m.npy").is_file()
        and (p / "camera_intrinsics.npz").is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No unpacked ZED capture directories found under {root}")
    return candidates[-1]


def load_plan(plan_file: Path) -> list[list[float]]:
    with plan_file.open("r") as f:
        plan = json.load(f)
    if not isinstance(plan, list):
        raise ValueError(f"Expected top-level list in plan file, got {type(plan).__name__}")
    for i, waypoint in enumerate(plan):
        if not isinstance(waypoint, list) or len(waypoint) != 6:
            raise ValueError(
                f"Waypoint {i} must be [x_norm, y_norm, dx, dy, dz, gripper], got {waypoint!r}"
            )
    return plan


def resolve_plan_file(args: argparse.Namespace) -> Path:
    if args.plan_file is not None:
        return resolve_path(args.plan_file)
    scene_dir = resolve_path(args.scene_dir)
    if args.plan_id is None:
        raise ValueError("Provide --plan-id or --plan-file")
    return scene_dir / args.plan_id / "initial_0_robot.json"


def resize_float_array(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr, mode="F").resize((size, size), Image.BILINEAR), dtype=np.float32)
    if arr.ndim == 3:
        channels = [
            np.asarray(Image.fromarray(arr[..., c], mode="F").resize((size, size), Image.BILINEAR), dtype=np.float32)
            for c in range(arr.shape[-1])
        ]
        return np.stack(channels, axis=-1)
    raise ValueError(f"Expected 2D or 3D float array, got shape {arr.shape}")


def crop_resize_zed_capture(capture_dir: Path, crop: tuple[int, int, int, int], size: int):
    x0, x1, y0, y1 = crop
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop {crop}; expected x1>x0 and y1>y0")

    rgb = np.load(capture_dir / "rgb.npy")
    depth = np.load(capture_dir / "depth_m.npy")
    xyz = np.load(capture_dir / "xyz_m.npy")
    intr_raw = dict(np.load(capture_dir / "camera_intrinsics.npz"))

    if rgb.shape[:2] != depth.shape or xyz.shape[:2] != depth.shape or xyz.shape[-1] != 3:
        raise ValueError(f"ZED arrays have incompatible shapes: rgb={rgb.shape}, depth={depth.shape}, xyz={xyz.shape}")

    H, W = depth.shape
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise ValueError(f"Crop {crop} is outside capture image size {(W, H)}")

    crop_w, crop_h = x1 - x0, y1 - y0
    rgb_crop = rgb[y0:y1, x0:x1]
    depth_crop = depth[y0:y1, x0:x1]
    xyz_crop = xyz[y0:y1, x0:x1]

    rgb_256 = np.asarray(Image.fromarray(rgb_crop).resize((size, size), Image.BILINEAR), dtype=np.uint8)
    depth_256 = resize_float_array(depth_crop, size)
    xyz_256 = resize_float_array(xyz_crop, size)

    sx = size / crop_w
    sy = size / crop_h
    intr = {
        "fx": float(intr_raw["fx"]) * sx,
        "fy": float(intr_raw["fy"]) * sy,
        "cx": (float(intr_raw["cx"]) - x0) * sx,
        "cy": (float(intr_raw["cy"]) - y0) * sy,
        "width": int(size),
        "height": int(size),
    }
    return rgb_256, depth_256, xyz_256, intr


def load_extrinsics_matrix(extrinsics_path: Path) -> np.ndarray:
    data = np.load(extrinsics_path)
    if "T" in data.files and np.asarray(data["T"]).shape == (4, 4):
        T = np.asarray(data["T"], dtype=np.float64)
    elif "R" in data.files and "T" in data.files:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.asarray(data["R"], dtype=np.float64)
        T[:3, 3] = np.asarray(data["T"], dtype=np.float64).reshape(3)
    else:
        raise ValueError(f"Extrinsics file must contain a 4x4 T or R plus 3x1 T keys: {extrinsics_path}")
    if T.shape != (4, 4):
        raise ValueError(f"Extrinsics T must resolve to 4x4, got {T.shape}")
    return T


def resolve_camera_to_base(extrinsics_path: Path, extrinsics_direction: str) -> np.ndarray:
    T = load_extrinsics_matrix(extrinsics_path)
    if extrinsics_direction == "camera_to_base":
        return T
    if extrinsics_direction == "base_to_camera":
        return np.linalg.inv(T)
    raise ValueError(f"Unsupported extrinsics direction: {extrinsics_direction}")


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    flat = points.reshape(-1, 3)
    hom = np.concatenate([flat, np.ones((flat.shape[0], 1), dtype=np.float64)], axis=1)
    out = hom @ T.T
    return out[:, :3].reshape(points.shape)


def fallback_backproject(u: int, v: int, depth: np.ndarray, intr: dict[str, float]) -> np.ndarray | None:
    z = float(depth[v, u])
    if not np.isfinite(z) or z <= 0.0:
        return None
    x = (float(u) - float(intr["cx"])) / float(intr["fx"]) * z
    y = (float(v) - float(intr["cy"])) / float(intr["fy"]) * z
    return np.array([x, y, z], dtype=np.float32)


def clip_position(position: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    if not args.clip_targets:
        return position.copy(), 0.0
    lower = np.array([args.workspace_x[0], args.workspace_y[0], args.workspace_z[0]], dtype=np.float64)
    upper = np.array([args.workspace_x[1], args.workspace_y[1], args.workspace_z[1]], dtype=np.float64)
    clipped = np.clip(position, lower, upper)
    return clipped, float(np.linalg.norm(clipped - position))


def build_targets(
    plan: list[list[float]],
    depth: np.ndarray,
    xyz: np.ndarray,
    intr: dict[str, float],
    T_cam_to_base: np.ndarray,
    gripper_quat: np.ndarray,
    args: argparse.Namespace,
) -> list[TargetRecord]:
    H, W = depth.shape
    targets: list[TargetRecord] = []

    for i, waypoint in enumerate(plan):
        x_norm, y_norm, dx, dy, dz, gripper_action = waypoint
        u = int(np.clip(round(float(x_norm) * (W - 1)), 0, W - 1))
        v = int(np.clip(round(float(y_norm) * (H - 1)), 0, H - 1))
        action = int(gripper_action)

        if action not in (GRIPPER_CLOSE, GRIPPER_OPEN):
            targets.append(
                TargetRecord(
                    index=i,
                    source_pixel=[u, v],
                    normalized_xy=[float(x_norm), float(y_norm)],
                    offset_robot_m=[float(dx), float(dy), float(dz)],
                    camera_point_m=None,
                    base_anchor_m=None,
                    target_position_unclipped_m=None,
                    target_position_m=None,
                    target_pose_wxyz=None,
                    gripper_action=action,
                    clipped_delta_m=0.0,
                    used_fallback_backprojection=False,
                    valid=False,
                    error=f"Unsupported gripper action {action}; expected 0 close or 1 open",
                )
            )
            continue

        camera_point = np.asarray(xyz[v, u], dtype=np.float64)
        used_fallback = False
        if not np.isfinite(camera_point).all() or np.linalg.norm(camera_point) <= 1e-8:
            fallback = fallback_backproject(u, v, depth, intr)
            used_fallback = True
            if fallback is None:
                targets.append(
                    TargetRecord(
                        index=i,
                        source_pixel=[u, v],
                        normalized_xy=[float(x_norm), float(y_norm)],
                        offset_robot_m=[float(dx), float(dy), float(dz)],
                        camera_point_m=None,
                        base_anchor_m=None,
                        target_position_unclipped_m=None,
                        target_position_m=None,
                        target_pose_wxyz=None,
                        gripper_action=action,
                        clipped_delta_m=0.0,
                        used_fallback_backprojection=True,
                        valid=False,
                        error=f"Invalid depth/xyz at pixel {(u, v)}",
                    )
                )
                continue
            camera_point = fallback.astype(np.float64)

        base_anchor = transform_points(camera_point.reshape(1, 3), T_cam_to_base)[0]
        plan_offset = np.array([float(dx), float(dy), float(dz)], dtype=np.float64)
        offset = np.zeros(3, dtype=np.float64) if (args.ignore_xyz_offsets or args.anchor_points_only) else plan_offset
        target_position = base_anchor + offset
        if not args.anchor_points_only:
            target_position[2] += float(args.finger_tip_offset)
        target_position_clipped, clipped_delta = clip_position(target_position, args)

        pose = np.concatenate([target_position_clipped, gripper_quat.astype(np.float64)])
        targets.append(
            TargetRecord(
                index=i,
                source_pixel=[u, v],
                normalized_xy=[float(x_norm), float(y_norm)],
                offset_robot_m=plan_offset.tolist(),
                camera_point_m=camera_point.tolist(),
                base_anchor_m=base_anchor.tolist(),
                target_position_unclipped_m=target_position.tolist(),
                target_position_m=target_position_clipped.tolist(),
                target_pose_wxyz=pose.tolist(),
                gripper_action=action,
                clipped_delta_m=clipped_delta,
                used_fallback_backprojection=used_fallback,
                valid=True,
                error=None,
            )
        )

    return targets


def finite_point_mask(points: np.ndarray, *, max_camera_distance_m: float | None = None) -> np.ndarray:
    mask = np.isfinite(points).all(axis=1)
    if max_camera_distance_m is not None and max_camera_distance_m > 0.0:
        mask &= np.linalg.norm(points, axis=1) <= max_camera_distance_m
    return mask


def make_scene_pointcloud(
    xyz_256: np.ndarray,
    rgb_256: np.ndarray,
    T_cam_to_base: np.ndarray,
    max_points: int,
    max_camera_distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    points_cam = xyz_256.reshape(-1, 3)
    colors = rgb_256.reshape(-1, 3).astype(np.float32) / 255.0
    mask = finite_point_mask(points_cam, max_camera_distance_m=max_camera_distance_m)
    points_base = transform_points(points_cam[mask], T_cam_to_base)
    colors = colors[mask]

    if max_points > 0 and len(points_base) > max_points:
        idx = np.linspace(0, len(points_base) - 1, max_points, dtype=np.int64)
        points_base = points_base[idx]
        colors = colors[idx]

    return points_base.astype(np.float32), colors.astype(np.float32)


def save_overlay(rgb: np.ndarray, targets: Iterable[TargetRecord], path: Path) -> None:
    img = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    valid_targets = [t for t in targets if t.valid]
    for a, b in zip(valid_targets, valid_targets[1:]):
        draw.line([tuple(a.source_pixel), tuple(b.source_pixel)], fill=(255, 0, 0), width=2)

    for target in targets:
        u, v = target.source_pixel
        color = (0, 220, 0) if target.valid else (255, 0, 0)
        r = 5
        draw.ellipse((u - r, v - r, u + r, v + r), outline=color, width=2)
        draw.text((u + 7, v - 7), str(target.index), fill=color, font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def pose_to_transform_wxyz(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    q = pose[3:] / (np.linalg.norm(pose[3:]) + 1e-12)
    w, x, y, z = q
    R = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = pose[:3]
    return T


def sample_line(start: np.ndarray, end: np.ndarray, count: int = 50) -> np.ndarray:
    t = np.linspace(0.0, 1.0, count, dtype=np.float64)[:, None]
    return start[None, :] * (1.0 - t) + end[None, :] * t


def target_marker_points(position: np.ndarray, radius: float = 0.015) -> np.ndarray:
    axes = np.eye(3, dtype=np.float64)
    lines = []
    for axis in axes:
        lines.append(sample_line(position - radius * axis, position + radius * axis, count=20))
    return np.vstack(lines)


def gripper_glyph_points(pose: np.ndarray) -> np.ndarray:
    """Return a simple point-sampled gripper glyph in the target pose frame."""
    T = pose_to_transform_wxyz(pose)
    # Local convention for visualization only: palm opens along local y, fingers
    # extend along local -z. This makes the fixed scraper orientation visible.
    local_segments = [
        (np.array([0.0, -0.04, 0.0]), np.array([0.0, 0.04, 0.0])),
        (np.array([0.0, -0.04, 0.0]), np.array([0.0, -0.04, -0.08])),
        (np.array([0.0, 0.04, 0.0]), np.array([0.0, 0.04, -0.08])),
        (np.array([-0.02, 0.0, 0.0]), np.array([0.02, 0.0, 0.0])),
    ]
    pts = []
    for start, end in local_segments:
        line = sample_line(start, end, count=70)
        hom = np.concatenate([line, np.ones((len(line), 1))], axis=1)
        pts.append((hom @ T.T)[:, :3])
    return np.vstack(pts)


def target_visual_points(targets: list[TargetRecord]) -> tuple[np.ndarray, np.ndarray]:
    palette = [
        np.array([1.0, 0.05, 0.05]),
        np.array([0.1, 0.5, 1.0]),
        np.array([0.1, 0.85, 0.2]),
        np.array([1.0, 0.75, 0.05]),
        np.array([0.85, 0.1, 1.0]),
    ]
    all_points = []
    all_colors = []

    for target in targets:
        if not target.valid or target.target_pose_wxyz is None:
            continue
        pose = np.asarray(target.target_pose_wxyz, dtype=np.float64)
        color = palette[target.index % len(palette)]
        pts = np.vstack([target_marker_points(pose[:3]), gripper_glyph_points(pose)])
        all_points.append(pts)
        all_colors.append(np.tile(color[None, :], (len(pts), 1)))

    if not all_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    return np.vstack(all_points).astype(np.float32), np.vstack(all_colors).astype(np.float32)


def write_ascii_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.float64).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError(f"PLY points/colors length mismatch: {len(points)} != {len(colors)}")
    colors_u8 = np.clip(np.round(colors * 255.0), 0, 255).astype(np.uint8)
    with path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]:.7f} {p[1]:.7f} {p[2]:.7f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def save_3d_visualization(points: np.ndarray, colors: np.ndarray, targets: list[TargetRecord], path: Path, show: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_points, target_colors = target_visual_points(targets)
    merged_points = np.vstack([points, target_points]) if len(target_points) else points
    merged_colors = np.vstack([colors, target_colors]) if len(target_colors) else colors
    write_ascii_ply(path, merged_points, merged_colors)

    if show:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("--show-vis requires open3d to be installed") from exc
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(merged_points)
        pcd.colors = o3d.utility.Vector3dVector(merged_colors)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, frame])


def save_records_json(
    path: Path,
    args: argparse.Namespace,
    plan_file: Path,
    zed_capture_dir: Path,
    extrinsics: Path,
    intr: dict[str, float],
    targets: list[TargetRecord],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "plan_file": str(plan_file),
        "zed_capture_dir": str(zed_capture_dir),
        "extrinsics": str(extrinsics),
        "extrinsics_direction": args.extrinsics_direction,
        "crop": list(args.crop),
        "size": int(args.size),
        "intrinsics_after_crop_resize": intr,
        "gripper_quat_wxyz": args.gripper_quat.tolist(),
        "finger_tip_offset": float(args.finger_tip_offset),
        "ignore_xyz_offsets": bool(args.ignore_xyz_offsets),
        "anchor_points_only": bool(args.anchor_points_only),
        "clip_targets": bool(args.clip_targets),
        "workspace_bounds": {
            "x": list(args.workspace_x),
            "y": list(args.workspace_y),
            "z": list(args.workspace_z),
        },
        "targets": [asdict(t) for t in targets],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def choose_plan_config(motion_planner, start_pose: np.ndarray | None, goal_pose: np.ndarray):
    if start_pose is None:
        return None
    delta = goal_pose[:3] - start_pose[:3]
    abs_delta = np.abs(delta)
    if abs_delta[2] > 0.03 and abs_delta[2] >= 2.0 * max(abs_delta[0], abs_delta[1], 1e-6):
        return motion_planner.lift_plan_config
    if max(abs_delta[0], abs_delta[1]) > 0.03 and abs_delta[2] < 0.03:
        return motion_planner.only_xy_translation_plan_config
    return None


def plan_segments(
    targets: list[TargetRecord],
    scene_points: np.ndarray,
    execute: bool,
    args: argparse.Namespace,
):
    import torch
    from frankapanda import FrankaPandaController
    from frankapanda.motionplanner import MotionPlanner

    controller = FrankaPandaController()
    if execute and args.home_first:
        controller.open_gripper()
        controller.move_to_joints(controller.home_joints, controller.open_gripper_action)

    current_joints_np = controller.get_robot_joints()
    current_joints = torch.tensor(current_joints_np, dtype=torch.float32, device=args.torch_device)
    motion_planner = MotionPlanner(scene_points)

    trajectories = []
    prev_pose = None
    for target in targets:
        if not target.valid or target.target_pose_wxyz is None:
            target.planning_success = False
            continue

        goal_pose_np = np.asarray(target.target_pose_wxyz, dtype=np.float32)
        goal_pose = torch.tensor(goal_pose_np, dtype=torch.float32, device=args.torch_device)
        plan_config = choose_plan_config(motion_planner, prev_pose, goal_pose_np)
        plan_kwargs = {}
        if plan_config is not None:
            plan_kwargs["plan_config"] = plan_config

        # Allow the hand/fingers to occupy the target contact region while still
        # checking the rest of the arm against the ZED collision cloud.
        plan_kwargs["disable_collision_links"] = motion_planner.links[-5:]

        traj, success = motion_planner.plan_to_goal_poses(
            current_joints=current_joints.unsqueeze(0),
            goal_poses=goal_pose.unsqueeze(0),
            **plan_kwargs,
        )
        target.planning_success = bool(success.item())
        if not target.planning_success:
            trajectories.append(None)
            continue

        trajectories.append(traj[0].detach().cpu().numpy())
        current_joints = traj[0, -1].detach()
        prev_pose = goal_pose_np

    all_success = all(t.valid and bool(t.planning_success) for t in targets)
    if not all_success:
        failed = [t.index for t in targets if not (t.valid and bool(t.planning_success))]
        raise RuntimeError(f"Planning failed for target indices: {failed}")

    clipped = [t.index for t in targets if t.clipped_delta_m > args.clip_tolerance_m]
    if args.clip_targets and clipped and not args.allow_clipped_targets:
        raise RuntimeError(
            f"Targets clipped by more than {args.clip_tolerance_m:.3f} m: {clipped}. "
            "Inspect dry-run output or pass --allow-clipped-targets."
        )

    if not execute:
        return trajectories

    for target, trajectory in zip(targets, trajectories):
        assert trajectory is not None
        current_gripper = controller.close_gripper_action if controller.get_gripper_state() == controller.close_gripper_action else controller.open_gripper_action
        controller.move_along_trajectory(trajectory, gripper_state=current_gripper)
        if target.gripper_action == GRIPPER_CLOSE:
            controller.close_gripper(num_steps=args.gripper_steps)
        elif target.gripper_action == GRIPPER_OPEN:
            controller.open_gripper(num_steps=args.gripper_steps)

    return trajectories


def print_target_summary(targets: list[TargetRecord]) -> None:
    print("\nResolved targets:")
    for t in targets:
        if not t.valid:
            print(f"  [{t.index}] INVALID pixel={t.source_pixel} action={t.gripper_action}: {t.error}")
            continue
        pos = np.asarray(t.target_position_m)
        clip = f", clipped {t.clipped_delta_m:.3f} m" if t.clipped_delta_m > 1e-6 else ""
        fallback = ", fallback-depth" if t.used_fallback_backprojection else ""
        print(
            f"  [{t.index}] pixel={t.source_pixel} "
            f"target=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
            f"action={t.gripper_action}{clip}{fallback}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-dir", default=str(DEFAULT_SCENE_DIR))
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--plan-file", default=None)
    parser.add_argument("--zed-capture-dir", default=None)
    parser.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    parser.add_argument(
        "--extrinsics-direction",
        choices=("base_to_camera", "camera_to_base"),
        default="camera_to_base",
        help="Frame direction of the matrix stored in --extrinsics. Default matches cam1_calibration.npz.",
    )
    parser.add_argument("--crop", nargs=4, type=int, default=list(DEFAULT_CROP), metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--execute", action="store_true", help="Move the real robot after a successful dry-run/planning pass.")
    parser.add_argument("--dry-run", action="store_true", help="Explicitly request dry-run behavior; this is the default unless --execute is set.")
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--show-vis", action="store_true")
    parser.add_argument("--gripper-quat", type=parse_quat, default=DEFAULT_GRIPPER_QUAT_WXYZ)
    parser.add_argument("--finger-tip-offset", type=float, default=0.08)
    parser.add_argument(
        "--ignore-xyz-offsets",
        action="store_true",
        help="Debug mode: ignore VLM dx/dy/dz offsets but still apply --finger-tip-offset.",
    )
    parser.add_argument(
        "--anchor-points-only",
        action="store_true",
        help="Debug mode: use raw backprojected VLM anchor points only; ignores dx/dy/dz and --finger-tip-offset.",
    )
    parser.add_argument("--workspace-x", nargs=2, type=float, default=[0.15, 0.85])
    parser.add_argument("--workspace-y", nargs=2, type=float, default=[-0.60, 0.60])
    parser.add_argument("--workspace-z", nargs=2, type=float, default=[0.02, 0.70])
    parser.add_argument("--clip-tolerance-m", type=float, default=0.02)
    parser.add_argument("--allow-clipped-targets", action="store_true")
    parser.add_argument(
        "--clip-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip projected targets to workspace bounds. Use --no-clip-targets to keep raw projected targets.",
    )
    parser.add_argument("--max-camera-distance-m", type=float, default=1.5)
    parser.add_argument("--max-planning-points", type=int, default=12000)
    parser.add_argument("--skip-planning", action="store_true", help="Only project targets and write visualizations.")
    parser.add_argument("--home-first", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gripper-steps", type=int, default=80)
    parser.add_argument("--torch-device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.crop = tuple(args.crop)

    plan_file = resolve_plan_file(args)
    if not plan_file.is_file():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")

    plan_dir = plan_file.parent
    vis_dir = resolve_path(args.vis_dir, base=plan_dir) if args.vis_dir is not None else plan_dir / "dry_run_vis"

    zed_capture_dir = resolve_path(args.zed_capture_dir) if args.zed_capture_dir else latest_zed_capture()
    extrinsics = resolve_path(args.extrinsics)
    if not zed_capture_dir.is_dir():
        raise FileNotFoundError(f"ZED capture directory not found: {zed_capture_dir}")
    if not extrinsics.is_file():
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics}")

    print(f"Plan:       {plan_file}")
    print(f"ZED:        {zed_capture_dir}")
    print(f"Extrinsics: {extrinsics} ({args.extrinsics_direction})")
    print(f"Vis dir:    {vis_dir}")
    print(f"Mode:       {'EXECUTE' if args.execute else 'DRY RUN'}")

    plan = load_plan(plan_file)
    rgb_256, depth_256, xyz_256, intr = crop_resize_zed_capture(zed_capture_dir, args.crop, args.size)
    T_cam_to_base = resolve_camera_to_base(extrinsics, args.extrinsics_direction)
    targets = build_targets(plan, depth_256, xyz_256, intr, T_cam_to_base, args.gripper_quat, args)
    print_target_summary(targets)

    scene_points, scene_colors = make_scene_pointcloud(
        xyz_256,
        rgb_256,
        T_cam_to_base,
        max_points=args.max_planning_points,
        max_camera_distance_m=args.max_camera_distance_m,
    )

    save_overlay(rgb_256, targets, vis_dir / "targets_2d_overlay.png")
    save_3d_visualization(scene_points, scene_colors, targets, vis_dir / "targets_3d.ply", show=args.show_vis)

    invalid = [t.index for t in targets if not t.valid]
    if invalid:
        save_records_json(vis_dir / "targets_3d.json", args, plan_file, zed_capture_dir, extrinsics, intr, targets)
        raise RuntimeError(f"Invalid targets: {invalid}. See {vis_dir / 'targets_3d.json'}")

    clipped = [t.index for t in targets if t.clipped_delta_m > args.clip_tolerance_m]
    if args.clip_targets and clipped and not args.allow_clipped_targets:
        save_records_json(vis_dir / "targets_3d.json", args, plan_file, zed_capture_dir, extrinsics, intr, targets)
        raise RuntimeError(
            f"Targets clipped by more than {args.clip_tolerance_m:.3f} m: {clipped}. "
            f"See {vis_dir / 'targets_3d.json'}"
        )

    try:
        if not args.skip_planning:
            plan_segments(targets, scene_points, execute=args.execute, args=args)
        else:
            print("Skipping MotionPlanner planning because --skip-planning was passed.")
    except Exception:
        save_records_json(vis_dir / "targets_3d.json", args, plan_file, zed_capture_dir, extrinsics, intr, targets)
        raise

    save_records_json(vis_dir / "targets_3d.json", args, plan_file, zed_capture_dir, extrinsics, intr, targets)

    print("\nWrote dry-run artifacts:")
    print(f"  {vis_dir / 'targets_2d_overlay.png'}")
    print(f"  {vis_dir / 'targets_3d.ply'}")
    print(f"  {vis_dir / 'targets_3d.json'}")
    if not args.execute:
        print("\nRobot was not moved. Pass --execute after inspecting the dry-run outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
