import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pyzed.sl as sl


def sl_mat_to_numpy(mat: sl.Mat) -> np.ndarray:
    arr = mat.get_data()
    return np.array(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_id", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default="data/zed_captures")
    parser.add_argument("--resolution", type=str, default="HD720")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--coordinate-system",
        choices=["IMAGE", "RIGHT_HANDED_Z_UP"],
        default="IMAGE",
        help="ZED XYZ coordinate system. IMAGE matches the OpenCV camera frame used by calibration.",
    )
    args = parser.parse_args()

    resolution_map = {
        "HD2K": sl.RESOLUTION.HD2K,
        "HD1200": sl.RESOLUTION.HD1200,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD720": sl.RESOLUTION.HD720,
        "VGA": sl.RESOLUTION.VGA,
    }

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution_map[args.resolution]
    init_params.camera_fps = args.fps
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = getattr(sl.COORDINATE_SYSTEM, args.coordinate_system)

    # For multiple ZEDs, this selects by camera index.
    init_params.set_from_camera_id(args.camera_id)

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open ZED camera {args.camera_id}: {status}")

    runtime_params = sl.RuntimeParameters()

    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()

    status = zed.grab(runtime_params)
    if status != sl.ERROR_CODE.SUCCESS:
        zed.close()
        raise RuntimeError(f"Failed to grab frame: {status}")

    zed.retrieve_image(image, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

    rgb_bgra = sl_mat_to_numpy(image)
    depth_m = sl_mat_to_numpy(depth)
    xyzrgba = sl_mat_to_numpy(point_cloud)

    # ZED image is BGRA. Convert to RGB.
    rgb = cv2.cvtColor(rgb_bgra, cv2.COLOR_BGRA2RGB)

    # Point cloud is H x W x 4: X, Y, Z, packed RGBA.
    xyz = xyzrgba[..., :3]

    cam_info = zed.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters
    left = calib.left_cam

    intrinsics = {
        "fx": left.fx,
        "fy": left.fy,
        "cx": left.cx,
        "cy": left.cy,
        "width": rgb.shape[1],
        "height": rgb.shape[0],
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) / f"zed{args.camera_id}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(out_dir / "rgb.npy", rgb)
    np.save(out_dir / "depth_m.npy", depth_m)
    np.save(out_dir / "xyz_m.npy", xyz)
    np.savez(out_dir / "camera_intrinsics.npz", **intrinsics)
    (out_dir / "coordinate_system.txt").write_text(args.coordinate_system + "\n")

    print(f"Saved ZED observation to: {out_dir}")
    print(f"RGB:   {rgb.shape}")
    print(f"Depth: {depth_m.shape}, meters")
    print(f"XYZ:   {xyz.shape}, meters")
    print(f"Intrinsics: {intrinsics}")
    print(f"Coordinate system: {args.coordinate_system}")

    zed.close()


if __name__ == "__main__":
    main()