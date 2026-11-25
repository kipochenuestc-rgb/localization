"""Export every recorded camera pose with explicit coverage and consistency flags.

This checks existing SLAM poses; it does not turn them into independent NDT fixes.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from rgb_core import Poses, load_rig
from gnss_core import LocalENU


def sample(times, xyz, rotations, query):
    result = np.tile(np.eye(4), (len(query), 1, 1))
    result[:, :3, :3] = Slerp(times-times[0], rotations)(query-times[0]).as_matrix()
    for j in range(3):
        result[:, j, 3] = np.interp(query, times, xyz[:, j])
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--calibration', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ndt-run', type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = np.load(args.audit/'raw_timestamps.npz')
    stamps = raw['/Data_Cam_Global'][:, 0]
    t0 = json.loads((args.audit/'raw_audit.json').read_text())['bag_start_s']
    key = Poses(args.run/'map/transformations.pcd')
    rows = np.genfromtxt(args.run/'trajectory.csv', delimiter=',', names=True)
    if not np.isfinite(np.column_stack([rows[n] for n in rows.dtype.names])).all():
        raise ValueError('Nonfinite input odometry')
    times = rows['stamp_s']
    if np.any(np.diff(times) <= 0):
        raise ValueError('Nonmonotonic odometry')
    xyz = np.column_stack([rows[n] for n in ['x', 'y', 'z']])
    rotations = Rotation.from_quat(np.column_stack([rows[n] for n in ['qx', 'qy', 'qz', 'qw']]))
    kt = key.times[(key.times >= times[0]) & (key.times <= times[-1])]
    correction = key.at(kt) @ np.linalg.inv(sample(times, xyz, rotations, kt))
    cr = Rotation.from_matrix(correction[:, :3, :3])
    covered = (stamps >= kt[0]) & (stamps <= kt[-1])
    query = stamps[covered]
    final = sample(kt, correction[:, :3, 3], cr, query) @ sample(times, xyz, rotations, query)
    sparse = key.at(query)
    position_difference = np.linalg.norm(final[:, :3, 3]-sparse[:, :3, 3], axis=1)
    angle_difference = np.degrees(Rotation.from_matrix(np.transpose(final[:, :3, :3], (0, 2, 1)) @ sparse[:, :3, :3]).magnitude())
    bracket = np.searchsorted(times, query, side='right').clip(1, len(times)-1)
    gaps = times[bracket]-times[bracket-1]
    key_index = np.searchsorted(kt, query, side='right').clip(1, len(kt)-1)
    key_gaps = kt[key_index]-kt[key_index-1]
    dt = np.diff(query)
    speed = np.r_[0, np.linalg.norm(np.diff(final[:, :3, 3], axis=0), axis=1)/dt]
    rate = np.r_[0, np.degrees(Rotation.from_matrix(np.transpose(final[:-1, :3, :3], (0, 2, 1)) @ final[1:, :3, :3]).magnitude())/dt]
    cameras, _, _, calibration_status = load_rig(args.calibration)
    base = np.full((len(stamps), 4, 4), np.nan)
    base[covered] = final
    camera_poses = np.full((len(stamps), 4, 4, 4), np.nan)
    for cid in range(4):
        camera_poses[covered, cid] = final @ cameras[cid]['reference_from_camera']
    flags = [[] for _ in stamps]
    for i in np.flatnonzero(~covered):
        flags[i].append('outside_final_map_pose_support')
    for i, gap, diff, angle, v, omega in zip(np.flatnonzero(covered), gaps, position_difference, angle_difference, speed, rate):
        if gap > .5: flags[i].append('odometry_bracket_over_0.5s')
        if diff > .1 or angle > 1.: flags[i].append('interpolation_methods_disagree')
        if v > 25. or omega > 90.: flags[i].append('motion_outlier')
    gnss = np.loadtxt(args.audit/'raw_gnss.csv', delimiter=',', skiprows=1, ndmin=2)
    gnss = gnss[np.isfinite(gnss[:, :5]).all(1) & (gnss[:, 5] >= 0)]
    alignment = json.loads((args.run/'gnss_alignment.json').read_text())
    local = LocalENU(*alignment['reference_wgs84'])
    gxyz = np.array([local.convert(r[3], r[2], r[4]) for r in gnss]) @ np.array(alignment['enu_to_map_rotation']).T + np.array(alignment['enu_to_map_translation'])
    mount = json.loads((args.calibration/'config/antenna_mount.json').read_text())
    C = cameras[0]['reference_from_camera']
    lever = C[:3, :3] @ np.array(mount['antenna_origin_in_camera0_m']) + C[:3, 3]
    valid_g = (gnss[:, 0] >= kt[0]) & (gnss[:, 0] <= kt[-1])
    gt = gnss[valid_g, 0]
    gm = sample(kt, correction[:, :3, 3], cr, gt) @ sample(times, xyz, rotations, gt)
    residual = gm[:, :3, 3] + np.einsum('nij,j->ni', gm[:, :3, :3], lever) - gxyz[valid_g]
    np.savetxt(args.output/'gnss_residuals.csv', np.column_stack([gt, residual]), delimiter=',', header='stamp_s,dx,dy,dz', comments='')
    near = np.searchsorted(gnss[:, 0], stamps).clip(0, len(gnss)-1)
    left = np.maximum(near-1, 0)
    near = np.where(abs(gnss[left, 0]-stamps) < abs(gnss[near, 0]-stamps), left, near)
    gnss_age = abs(gnss[near, 0]-stamps)
    for i in np.flatnonzero(gnss_age > 2.): flags[i].append('no_fixed_gnss_within_2s')
    gap_all = np.full(len(stamps), np.nan); gap_all[covered] = gaps
    diff_all = np.full(len(stamps), np.nan); diff_all[covered] = position_difference
    angle_all = np.full(len(stamps), np.nan); angle_all[covered] = angle_difference
    with (args.output/'all_camera_poses.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group_index', 'camera_id', 'stamp_s', 'bag_time_s', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw', 'pose_available', 'source', 'odometry_bracket_s', 'gnss_nearest_age_s', 'interpolation_difference_m', 'interpolation_difference_deg', 'review_flags'])
        for i, t in enumerate(stamps):
            for cid in range(4):
                pose = camera_poses[i, cid]
                values = [*pose[:3, 3], *Rotation.from_matrix(pose[:3, :3]).as_quat()] if covered[i] else ['']*7
                writer.writerow([i, cid, f'{t:.9f}', t-t0, *values, int(covered[i]), 'slam_odometry_with_final_graph_correction' if covered[i] else 'missing', gap_all[i], gnss_age[i], diff_all[i], angle_all[i], '|'.join(flags[i])])
    np.savez_compressed(args.output/'all_camera_poses.npz', stamps=stamps, bag_start_s=t0, base=base, cameras=camera_poses, covered=covered, review=np.array([bool(f) for f in flags]), gnss_stamps=gnss[:, 0], gnss_xyz=gxyz)
    from collections import Counter
    counts = Counter(flag for group in flags for flag in group)
    ndt = None
    if args.ndt_run:
        e = np.genfromtxt(args.ndt_run/'ekf_trajectory.csv', delimiter=',', names=True)
        ndt_covered = (stamps >= e['stamp_s'][0]) & (stamps <= e['stamp_s'][-1])
        ndt = {'image_groups_in_time_support': int(ndt_covered.sum()), 'time_interval_bag_s': [float(e['stamp_s'][0]-t0), float(e['stamp_s'][-1]-t0)], 'note': 'Time coverage only; not a registration accuracy pass.'}
    q = lambda a: np.quantile(a, [.5, .95, 1]).tolist()
    report = {'image_groups': len(stamps), 'camera_images': len(stamps)*4, 'groups_with_pose': int(covered.sum()), 'groups_missing_pose': int((~covered).sum()), 'groups_with_review_flags': sum(bool(f) for f in flags), 'flag_counts': dict(counts), 'support_bag_seconds': [float(kt[0]-t0), float(kt[-1]-t0)], 'odometry_bracket_p50_p95_max_s': q(gaps), 'keyframe_bracket_p50_p95_max_s': q(key_gaps), 'interpolation_position_difference_p50_p95_max_m': q(position_difference), 'interpolation_angle_difference_p50_p95_max_deg': q(angle_difference), 'speed_p50_p95_max_m_s': q(speed), 'angular_rate_p50_p95_max_deg_s': q(rate), 'gnss_comparison_count': len(gt), 'same_source_gnss_horizontal_rmse_m': float(np.sqrt(np.mean(np.sum(residual[:, :2]**2, axis=1)))), 'ndt_ekf': ndt, 'calibration_status': calibration_status, 'limitations': ['These are existing SLAM poses, not full-route NDT localization.', 'Per-image poses are interpolated, not independently measured.', 'GNSS was used to construct the map; residuals are consistency checks, not independent accuracy.', 'Four images in each mosaic share a message timestamp; individual exposure timestamps were not supplied.', 'Extrinsics remain candidate calibration; no full-route image reprojection validation is claimed.', 'No pose is extrapolated outside final graph and odometry support.']}
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
