"""Load the calibrated four-LiDAR transforms used by map localization."""
import json
from pathlib import Path

import numpy as np


def load_lidar_transforms(calibration_dir):
    calibration_dir = Path(calibration_dir)
    rig = json.loads((calibration_dir / 'results/rig_cam0_with_lidars_paired_1203.json').read_text())
    mount = json.loads((calibration_dir / 'config/imu_mount.json').read_text())
    imu_from_camera = np.eye(4)
    imu_from_camera[:3, :3] = np.column_stack([
        mount['camera0_axes_in_imu'][axis] for axis in ['x', 'y', 'z']
    ])
    imu_from_camera[:3, 3] = mount['camera0_origin_in_imu_m']
    imu_from_lidars = [
        imu_from_camera @ np.array(rig['lidars'][str(index)]['T_reference_from_lidar'])
        for index in range(4)
    ]
    origin = imu_from_lidars[0][:3, 3].copy()
    for transform in imu_from_lidars:
        transform[:3, 3] -= origin
    return imu_from_lidars
