"""Stable ZED camera identity by serial number.

OS camera indices can change after reboot; use logical ids mapped to serials.
"""

from __future__ import annotations

import pyzed.sl as sl


class ZedCams:
    # cam 0, 1: eye-to-hand fixed ZED 2i cameras in this lab.
    # cam 2:    wrist-mounted ZED Mini, used for eye-in-hand calibration.
    id2serial = {
        0: 34858067,
        1: 31240214,
        2: 14636260,
    }


DEFAULT_ZED_CAMERA_ID = 0


def serial_for_camera_id(camera_id: int) -> int:
    try:
        return ZedCams.id2serial[camera_id]
    except KeyError as exc:
        known = ", ".join(str(k) for k in sorted(ZedCams.id2serial))
        raise ValueError(
            f"Unknown logical ZED camera id {camera_id}. Known ids: {known}"
        ) from exc


def apply_camera_to_init_params(init_params: sl.InitParameters, camera_id: int) -> int:
    """Select a ZED by serial number. Returns the serial used."""
    serial = serial_for_camera_id(camera_id)
    init_params.set_from_serial_number(serial)
    return serial
