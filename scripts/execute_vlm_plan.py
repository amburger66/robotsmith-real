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
WORKSPACE_ROOT = ROBOTSMITH_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "data"
# Let this script run both from the repository root and after editable installs.
for path in (ROBOTSMITH_ROOT, ROBOTSMITH_ROOT / "robo_utils"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from frankapanda.motionplanner import EE_LINK_CENTER_TO_GRIPPER_TIP


TASK_TO_TOOL = {
    "task08_cutting": "cutter",
    "task03_flatten": "paddle",
}
DEFAULT_TASK_NAME = "task08_cutting"
DEFAULT_EXTRINSICS = DATA_ROOT / "calibration/eye_to_hand/cam0_calibration.npz"
ZED_CAPTURE_DIR = {
    "task08_cutting": "data/zed_captures/zed1_20260515_131532",
    "task03_flatten": "data/zed_captures/zed1_20260516_154329",
}

DEFAULT_CROP = (380, 1000, 0, 620)
DEFAULT_GRIPPER_QUAT_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

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
    target_position_m: list[float] | None
    target_pose_wxyz: list[float] | None
    gripper_action: int
    used_fallback_backprojection: bool
    valid: bool
    z_clipped_for_grip: bool = False
    error: str | None = None
    planning_success: bool | None = None


def resolve_path(path: str | os.PathLike, *, base: Path = WORKSPACE_ROOT) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base / p).resolve()


def task_tool(task_name: str) -> str:
    try:
        return TASK_TO_TOOL[task_name]
    except KeyError as exc:
        supported = ", ".join(sorted(TASK_TO_TOOL))
        raise ValueError(f"Unsupported task_name {task_name!r}; expected one of: {supported}") from exc


def default_scene_dir(task_name: str) -> Path:
    return DATA_ROOT / task_name / "vlm_traj_queries" / task_tool(task_name)


def default_episode_dir(task_name: str) -> Path:
    return DATA_ROOT / "real_world_episodes_wm" / task_name / task_tool(task_name)


def default_zed_capture_dir(task_name: str) -> Path:
    capture = ZED_CAPTURE_DIR.get(task_name)
    if not capture:
        raise ValueError(f"ZED_CAPTURE_DIR[{task_name!r}] is not configured")
    return resolve_path(capture)

def parse_quat(raw: str) -> np.ndarray:
    values = [float(v.strip()) for v in raw.split(",")]
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--gripper-quat must contain 4 comma-separated values")
    quat = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm < 1e-8:
        raise argparse.ArgumentTypeError("--gripper-quat must be a nonzero quaternion")
    return quat / norm


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
    scene_dir = resolve_path(args.scene_dir) if args.scene_dir is not None else default_scene_dir(args.task_name)
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


def crop_resize_rgb_array(rgb: np.ndarray, crop: tuple[int, int, int, int], size: int) -> np.ndarray:
    x0, x1, y0, y1 = crop
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with shape HxWx3, got {rgb.shape}")
    H, W = rgb.shape[:2]
    if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
        raise ValueError(f"Crop {crop} is outside image size {(W, H)}")
    rgb_crop = rgb[y0:y1, x0:x1]
    return np.asarray(Image.fromarray(rgb_crop).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def resize_intrinsics(intr_raw: dict, crop: tuple[int, int, int, int], size: int) -> dict[str, float]:
    x0, x1, y0, y1 = crop
    crop_w, crop_h = x1 - x0, y1 - y0
    sx = size / crop_w
    sy = size / crop_h
    return {
        "fx": float(intr_raw["fx"]) * sx,
        "fy": float(intr_raw["fy"]) * sy,
        "cx": (float(intr_raw["cx"]) - x0) * sx,
        "cy": (float(intr_raw["cy"]) - y0) * sy,
        "width": int(size),
        "height": int(size),
    }


def intrinsics_dict_to_matrix(intr: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            [float(intr["fx"]), 0.0, float(intr["cx"])],
            [0.0, float(intr["fy"]), float(intr["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


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

    depth_crop = depth[y0:y1, x0:x1]
    xyz_crop = xyz[y0:y1, x0:x1]

    rgb_256 = crop_resize_rgb_array(rgb, crop, size)
    depth_256 = resize_float_array(depth_crop, size)
    xyz_256 = resize_float_array(xyz_crop, size)
    intr = resize_intrinsics(intr_raw, crop, size)

    coord_file = capture_dir / "coordinate_system.txt"
    coordinate_system = coord_file.read_text().strip() if coord_file.is_file() else "unknown"
    return rgb_256, depth_256, xyz_256, intr, coordinate_system


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


def resolve_episode_id(args: argparse.Namespace, plan_file: Path) -> str:
    if args.episode_id:
        return str(args.episode_id)
    if args.plan_id:
        return str(args.plan_id)
    return plan_file.stem


class LiveZedWmRecorder:
    def __init__(self, args: argparse.Namespace, T_cam_to_base: np.ndarray, episode_dir: Path, episode_id: str):
        self.args = args
        self.crop = tuple(args.crop)
        self.size = int(args.size)
        self.T_base_to_camera = np.linalg.inv(np.asarray(T_cam_to_base, dtype=np.float64))
        self.episode_dir = episode_dir
        self.episode_id = episode_id
        self.states: list[np.ndarray] = []
        self.positions: list[np.ndarray] = []
        self.grippers: list[float] = []
        self.zed = None
        self.runtime_params = None
        self.image_mat = None
        self.intrinsics: np.ndarray | None = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def open(self) -> None:
        import pyzed.sl as sl

        resolution_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1200": sl.RESOLUTION.HD1200,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution_map[self.args.zed_resolution]
        init_params.camera_fps = int(self.args.zed_fps)
        init_params.depth_mode = getattr(sl.DEPTH_MODE, self.args.zed_depth_mode)
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        init_params.set_from_camera_id(int(self.args.zed_camera_id))

        self.zed = sl.Camera()
        status = self.zed.open(init_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera {self.args.zed_camera_id}: {status}")

        self.runtime_params = sl.RuntimeParameters()
        self.image_mat = sl.Mat()
        for _ in range(max(0, int(self.args.zed_warmup_frames))):
            self.zed.grab(self.runtime_params)

        cam_info = self.zed.get_camera_information()
        left = cam_info.camera_configuration.calibration_parameters.left_cam
        intr_raw = {
            "fx": left.fx,
            "fy": left.fy,
            "cx": left.cx,
            "cy": left.cy,
        }
        resized_intr = resize_intrinsics(intr_raw, self.crop, self.size)
        self.intrinsics = intrinsics_dict_to_matrix(resized_intr)

    def close(self) -> None:
        if self.zed is not None:
            self.zed.close()
            self.zed = None

    def capture_rgb(self) -> np.ndarray:
        import pyzed.sl as sl

        if self.zed is None or self.runtime_params is None or self.image_mat is None:
            raise RuntimeError("ZED recorder is not open")
        status = self.zed.grab(self.runtime_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to grab ZED frame for WM data: {status}")
        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        bgra = np.array(self.image_mat.get_data())
        rgb = np.ascontiguousarray(bgra[..., [2, 1, 0]])
        return crop_resize_rgb_array(rgb, self.crop, self.size)

    def start_episode(self) -> None:
        self.states.append(self.capture_rgb())

    def record_step(self, ee_before: np.ndarray, ee_after: np.ndarray, gripper_action: int) -> None:
        self.positions.append(np.stack([ee_before, ee_after], axis=0).astype(np.float32))
        self.grippers.append(float(gripper_action))
        self.states.append(self.capture_rgb())

    def save(self) -> Path:
        if self.intrinsics is None:
            raise RuntimeError("Cannot save WM episode before opening ZED recorder")
        if len(self.states) - 1 != len(self.positions) or len(self.positions) != len(self.grippers):
            raise RuntimeError(
                "Inconsistent WM episode lengths: "
                f"states={len(self.states)}, positions={len(self.positions)}, grippers={len(self.grippers)}"
            )
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        path = self.episode_dir / f"{self.episode_id}.npz"
        positions = np.asarray(self.positions, dtype=np.float32)
        np.savez(
            path,
            states=np.asarray(self.states, dtype=np.uint8),
            positions=positions,
            valid_mask=np.ones(positions.shape[:2], dtype=bool),
            grippers=np.asarray(self.grippers, dtype=np.float32),
            penalty=np.asarray([0.0, 0.0], dtype=np.float32),
            metric=np.asarray(0.0, dtype=np.float32),
            intrinsics=self.intrinsics,
            extrinsics=self.T_base_to_camera,
        )
        return path


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
    gripping_tool = False
    grip_z_floor_m: float | None = None

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
                    target_position_m=None,
                    target_pose_wxyz=None,
                    gripper_action=action,
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
                        target_position_m=None,
                        target_pose_wxyz=None,
                        gripper_action=action,
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
            target_position[2] += float(EE_LINK_CENTER_TO_GRIPPER_TIP)

        z_clipped = False
        if gripping_tool and grip_z_floor_m is not None:
            clipped_z = max(target_position[2], grip_z_floor_m)
            z_clipped = bool(clipped_z > target_position[2] + 1e-9)
            target_position[2] = clipped_z

        pose = np.concatenate([target_position, gripper_quat.astype(np.float64)])
        targets.append(
            TargetRecord(
                index=i,
                source_pixel=[u, v],
                normalized_xy=[float(x_norm), float(y_norm)],
                offset_robot_m=plan_offset.tolist(),
                camera_point_m=camera_point.tolist(),
                base_anchor_m=base_anchor.tolist(),
                target_position_m=target_position.tolist(),
                target_pose_wxyz=pose.tolist(),
                gripper_action=action,
                used_fallback_backprojection=used_fallback,
                z_clipped_for_grip=z_clipped,
                valid=True,
                error=None,
            )
        )

        if action == GRIPPER_CLOSE and not gripping_tool:
            gripping_tool = True
            grip_z_floor_m = float(target_position[2])
            print("setting grip_z_floor_m to ", grip_z_floor_m)
        elif action == GRIPPER_OPEN:
            gripping_tool = False
            grip_z_floor_m = None

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
        "crop": list(args.crop),
        "size": int(args.size),
        "intrinsics_after_crop_resize": intr,
        "zed_coordinate_system": getattr(args, "zed_coordinate_system", "unknown"),
        "gripper_quat_wxyz": args.gripper_quat.tolist(),
        "ee_link_center_to_gripper_tip": float(EE_LINK_CENTER_TO_GRIPPER_TIP),
        "ignore_xyz_offsets": bool(args.ignore_xyz_offsets),
        "anchor_points_only": bool(args.anchor_points_only),
        "targets": [asdict(t) for t in targets],
    }
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def plan_segments(
    targets: list[TargetRecord],
    scene_points: np.ndarray,
    T_cam_to_base: np.ndarray,
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
    # NOTE: point cloud is not being used for motion planning, but is used for visualization
    motion_planner = MotionPlanner(scene_points)

    # motion_planner.visualize_world_and_robot(current_joints)

    trajectories = []
    for target in targets:
        print("Planning target: ", target.index)
        if not target.valid or target.target_pose_wxyz is None:
            target.planning_success = False
            continue

        goal_pose_np = np.asarray(target.target_pose_wxyz, dtype=np.float32)
        goal_pose = torch.tensor(goal_pose_np, dtype=torch.float32, device=args.torch_device)

        if args.dry_run:
            motion_planner.visualize_world_and_robot(current_joints, goal_pose)

        # Allow the hand/fingers to occupy the target contact region while still
        # checking the rest of the arm against the ZED collision cloud.
        traj, success = motion_planner.plan_to_goal_poses(
            current_joints=current_joints.unsqueeze(0),
            goal_poses=goal_pose.unsqueeze(0),
            plan_config=motion_planner.fixed_gripper_orientation_plan_config,
            disable_collision_links=motion_planner.links[-5:],
        )
        print("Planning success: ", success.item())
        target.planning_success = bool(success.item())
        if not target.planning_success:
            break

        trajectories.append(traj[0].detach().cpu().numpy())
        current_joints = traj[0, -1].detach()

    all_success = all(t.valid and bool(t.planning_success) for t in targets)
    if not all_success:
        failed = [t.index for t in targets if not (t.valid and bool(t.planning_success))]
        raise RuntimeError(f"Planning failed for target indices: {failed}")

    if not execute:
        return trajectories

    recorder = None
    if args.collect_wm_data:
        recorder = LiveZedWmRecorder(args, T_cam_to_base, args.resolved_episode_dir, args.resolved_episode_id)

    try:
        if recorder is not None:
            recorder.open()
            recorder.start_episode()

        for target, trajectory in zip(targets, trajectories):
            assert trajectory is not None
            ee_before = controller.get_gripper_pose()[:3].astype(np.float32)
            current_gripper = controller.close_gripper_action if controller.get_gripper_state() == controller.close_gripper_action else controller.open_gripper_action
            controller.move_along_trajectory(trajectory, gripper_state=current_gripper)
            if target.gripper_action == GRIPPER_CLOSE:
                controller.close_gripper(num_steps=args.gripper_steps)
            elif target.gripper_action == GRIPPER_OPEN:
                controller.open_gripper(num_steps=args.gripper_steps)
            ee_after = controller.get_gripper_pose()[:3].astype(np.float32)

            if recorder is not None:
                if args.capture_settle_s > 0.0:
                    import time

                    time.sleep(float(args.capture_settle_s))
                recorder.record_step(ee_before, ee_after, target.gripper_action)

        if recorder is not None:
            episode_path = recorder.save()
            print(f"Saved WM episode: {episode_path}")

        return trajectories
    finally:
        if recorder is not None:
            recorder.close()


def print_target_summary(targets: list[TargetRecord]) -> None:
    print("\nResolved targets:")
    for t in targets:
        if not t.valid:
            print(f"  [{t.index}] INVALID pixel={t.source_pixel} action={t.gripper_action}: {t.error}")
            continue
        pos = np.asarray(t.target_position_m)
        fallback = ", fallback-depth" if t.used_fallback_backprojection else ""
        clipped = ", z-clipped (gripping)" if t.z_clipped_for_grip else ""
        print(
            f"  [{t.index}] pixel={t.source_pixel} "
            f"target=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] "
            f"action={t.gripper_action}{fallback}{clipped}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", choices=tuple(TASK_TO_TOOL), default=DEFAULT_TASK_NAME)
    parser.add_argument("--scene-dir", default=None, help="Plan root override. Defaults to data/<task-name>/vlm_traj_queries/<tool>.")
    parser.add_argument("--plan-id", default=None)
    parser.add_argument("--plan-file", default=None)
    parser.add_argument("--extrinsics", default=str(DEFAULT_EXTRINSICS))
    parser.add_argument(
        "--collect-wm-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="During --execute, save a WM-compatible real-world episode .npz. Use --no-collect-wm-data to disable.",
    )
    parser.add_argument("--episode-dir", default=None, help="Directory for collected WM episode .npz files.")
    parser.add_argument("--episode-id", default=None, help="Episode file stem. Defaults to --plan-id, otherwise the plan file stem.")
    parser.add_argument("--zed-camera-id", type=int, default=1)
    parser.add_argument("--zed-resolution", choices=("HD2K", "HD1200", "HD1080", "HD720", "VGA"), default="HD720")
    parser.add_argument("--zed-fps", type=int, default=30)
    parser.add_argument(
        "--zed-depth-mode",
        choices=("NEURAL_PLUS", "NEURAL", "NEURAL_LIGHT", "ULTRA", "QUALITY", "PERFORMANCE"),
        default="NEURAL",
    )
    parser.add_argument("--zed-warmup-frames", type=int, default=5)
    parser.add_argument("--capture-settle-s", type=float, default=0.25)
    parser.add_argument("--crop", nargs=4, type=int, default=list(DEFAULT_CROP), metavar=("X0", "X1", "Y0", "Y1"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--execute", action="store_true", help="Move the real robot after a successful dry-run/planning pass.")
    parser.add_argument("--dry-run", action="store_true", help="Explicitly request dry-run behavior; this is the default unless --execute is set.")
    parser.add_argument("--vis-dir", default=None)
    parser.add_argument("--show-vis", action="store_true")
    parser.add_argument("--gripper-quat", type=parse_quat, default=DEFAULT_GRIPPER_QUAT_WXYZ)
    parser.add_argument(
        "--ignore-xyz-offsets",
        action="store_true",
        help="Debug mode: ignore VLM dx/dy/dz offsets but still apply the gripper-tip offset.",
    )
    parser.add_argument(
        "--anchor-points-only",
        action="store_true",
        help="Debug mode: use raw backprojected VLM anchor points only; ignores dx/dy/dz and the gripper-tip offset.",
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
    if args.execute and args.collect_wm_data and args.skip_planning:
        raise ValueError("WM data collection requires planning/execution; remove --skip-planning or pass --no-collect-wm-data")

    plan_file = resolve_plan_file(args)
    if not plan_file.is_file():
        raise FileNotFoundError(f"Plan file not found: {plan_file}")

    plan_dir = plan_file.parent
    vis_dir = resolve_path(args.vis_dir, base=plan_dir) if args.vis_dir is not None else plan_dir / "dry_run_vis"
    args.resolved_episode_dir = (
        resolve_path(args.episode_dir) if args.episode_dir is not None else default_episode_dir(args.task_name)
    )
    args.resolved_episode_id = resolve_episode_id(args, plan_file)

    zed_capture_dir = default_zed_capture_dir(args.task_name)
    extrinsics = resolve_path(args.extrinsics)
    if not zed_capture_dir.is_dir():
        raise FileNotFoundError(f"ZED capture directory not found: {zed_capture_dir}")
    if not extrinsics.is_file():
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsics}")

    print(f"Task:       {args.task_name} ({task_tool(args.task_name)})")
    print(f"Plan:       {plan_file}")
    print(f"ZED:        {zed_capture_dir}")
    print(f"Extrinsics: {extrinsics} (camera_to_base)")
    print(f"Vis dir:    {vis_dir}")
    print(f"Mode:       {'EXECUTE' if args.execute else 'DRY RUN'}")
    if args.execute and args.collect_wm_data:
        print(f"WM episode: {args.resolved_episode_dir / (args.resolved_episode_id + '.npz')}")

    plan = load_plan(plan_file)
    rgb_256, depth_256, xyz_256, intr, zed_coordinate_system = crop_resize_zed_capture(zed_capture_dir, args.crop, args.size)
    args.zed_coordinate_system = zed_coordinate_system
    if zed_coordinate_system == "unknown":
        print(
            "WARNING: ZED capture has no coordinate_system.txt metadata. "
            "Old captures from capture_zed_scene.py may have used RIGHT_HANDED_Z_UP, "
            "while calibration expects OpenCV/ZED IMAGE camera coordinates."
        )
    elif zed_coordinate_system != "IMAGE":
        print(
            f"WARNING: ZED capture coordinate system is {zed_coordinate_system!r}; "
            "calibration expects OpenCV/ZED IMAGE camera coordinates."
        )
    T_cam_to_base = load_extrinsics_matrix(extrinsics)
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

    try:
        if not args.skip_planning:
            plan_segments(targets, scene_points, T_cam_to_base, execute=args.execute, args=args)
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
