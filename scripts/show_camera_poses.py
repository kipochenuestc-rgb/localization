"""Show audited camera centers, orientations and an animated image timestamp in RViz."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from pcd_io import read_pcd
from localization_paths import SLAM_ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--map', type=Path, required=True)
    ap.add_argument('--rate', type=float, default=2.)
    ap.add_argument('--bag', type=Path, help='Play original camera images on the same timeline')
    ap.add_argument('--calibration', type=Path)
    args = ap.parse_args()
    if args.rate <= 0: raise ValueError('rate must be positive')
    import rclpy
    from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2, PointField, Image
    from geometry_msgs.msg import Point, Pose, PoseArray
    from visualization_msgs.msg import Marker, MarkerArray
    from nav_msgs.msg import Path as RosPath
    from geometry_msgs.msg import PoseStamped
    rclpy.init()
    node = rclpy.create_node('audited_camera_pose_viewer')
    from tf2_ros import StaticTransformBroadcaster
    from geometry_msgs.msg import TransformStamped
    static_tf = StaticTransformBroadcaster(node)
    anchor = TransformStamped(); anchor.header.frame_id = 'map'; anchor.child_frame_id = 'pose_audit_origin'
    anchor.header.stamp = node.get_clock().now().to_msg(); anchor.transform.rotation.w = 1.
    static_tf.sendTransform(anchor)
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    mp = node.create_publisher(PointCloud2, '/pose_audit/map', qos)
    markers_pub = node.create_publisher(MarkerArray, '/pose_audit/all_cameras', qos)
    current_pub = node.create_publisher(MarkerArray, '/pose_audit/current_cameras', qos)
    path_pub = node.create_publisher(RosPath, '/pose_audit/slam_path', qos)
    arrays = [node.create_publisher(PoseArray, f'/pose_audit/camera{cid}/poses', qos) for cid in range(4)]
    data = np.load(args.audit/'all_camera_poses.npz')
    report = json.loads((args.audit/'report.json').read_text())
    source_label = 'NDT + GNSS + EKF poses' if report.get('pose_source') == 'ndt_gnss_ekf' else 'Existing SLAM poses (not full-route NDT)'
    valid = np.flatnonzero(data['covered']); stamps = data['stamps']; start = float(data['bag_start_s'])
    image_source = None
    image_index = None
    last_image_stamp = None
    if args.bag:
        from bag_camera_images import BagCameraImages, labeled_mosaic
        from rgb_core import load_rig
        rig, _, quadrants, _ = load_rig(args.calibration or args.bag.parent/'calibration')
        timestamps_file = SLAM_ROOT/'runs/parameter_accuracy_audit/raw_timestamps.npz'
        recorded = np.load(timestamps_file)['/Data_Cam_Global'] if timestamps_file.exists() else None
        image_source = BagCameraImages(args.bag, rig, quadrants, recorded)
        image_pubs = [node.create_publisher(Image, f'/pose_audit/camera{cid}/image', qos) for cid in range(4)]
        mosaic_pub = node.create_publisher(Image, '/pose_audit/cameras/image', qos)
        def publish_image(pub, pixels, stamp, frame):
            pixels = np.ascontiguousarray(pixels)
            m = Image(); m.header.stamp = stamp; m.header.frame_id = frame
            m.height, m.width = pixels.shape[:2]; m.encoding = 'bgr8'; m.step = m.width*3; m.data = pixels.tobytes()
            pub.publish(m)
    colors = [(1., .35, .2), (.25, 1., .35), (.2, .55, 1.), (1., .2, 1.)]
    def point(x): return Point(x=float(x[0]), y=float(x[1]), z=float(x[2]))
    def marker(ns, mid, kind, color, width):
        m = Marker(); m.header.frame_id = 'map'; m.ns = ns; m.id = mid; m.type = kind; m.action = Marker.ADD
        m.pose.orientation.w = 1.; m.scale.x = width; m.scale.y = width; m.scale.z = width
        m.color.r, m.color.g, m.color.b = color; m.color.a = 1.
        return m
    mapdata = read_pcd(args.map)
    xyz = np.column_stack([mapdata[n] for n in ['x', 'y', 'z']]); xyz = xyz[np.isfinite(xyz).all(1)]
    xyz = np.ascontiguousarray(xyz[::max(1, int(np.ceil(len(xyz)/300000)))], dtype='<f4')
    cloud = PointCloud2(); cloud.header.frame_id = 'map'; cloud.height = 1; cloud.width = len(xyz)
    cloud.fields = [PointField(name=n, offset=i*4, datatype=7, count=1) for i, n in enumerate(['x', 'y', 'z'])]
    cloud.point_step = 12; cloud.row_step = len(xyz)*12; cloud.is_dense = True; cloud.data = xyz.tobytes()
    mp.publish(cloud)
    all_markers = MarkerArray()
    for cid in range(4):
        m = marker('camera_centers', cid, Marker.POINTS, colors[cid], .22)
        m.points = [point(x) for x in data['cameras'][valid, cid, :3, 3]]
        all_markers.markers.append(m)
        pa = PoseArray(); pa.header.frame_id = 'map'
        for T in data['cameras'][valid, cid]:
            p = Pose(); p.position = point(T[:3, 3]); q = Rotation.from_matrix(T[:3, :3]).as_quat()
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q.tolist(); pa.poses.append(p)
        arrays[cid].publish(pa)
    warning = marker('review_groups', 0, Marker.POINTS, (1., .65, 0.), .38)
    warning.points = [point(x) for x in data['base'][data['covered'] & data['review'], :3, 3]]
    all_markers.markers.append(warning)
    gps = marker('gnss_antenna', 0, Marker.POINTS, (.2, 1., 1.), .65)
    gps.points = [point(x) for x in data['gnss_xyz']]; all_markers.markers.append(gps)
    markers_pub.publish(all_markers)
    path = RosPath(); path.header.frame_id = 'map'
    for T in data['base'][valid]:
        p = PoseStamped(); p.header.frame_id = 'map'; p.pose.position = point(T[:3, 3]); p.pose.orientation.w = 1.; path.poses.append(p)
    path_pub.publish(path)
    epoch = time.monotonic(); period = stamps[-1]-stamps[0]+2.
    corners = np.array([[-1., -.7, 1.4], [1., -.7, 1.4], [1., .7, 1.4], [-1., .7, 1.4]])
    def update():
        nonlocal image_index, last_image_stamp
        tick = time.monotonic()
        target = stamps[0] + ((time.monotonic()-epoch)*args.rate) % period
        i = int(np.clip(np.searchsorted(stamps, target, side='right')-1, 0, len(stamps)-1))
        image_stamp = None
        if image_source is not None and i != image_index:
            image_stamp, tiles = image_source.at(stamps[i])
            last_image_stamp = image_stamp
            for cid in range(4):
                publish_image(image_pubs[cid], tiles[cid], image_stamp, f'camera{cid}_optical')
            publish_image(mosaic_pub, labeled_mosaic(tiles), image_stamp, 'camera_mosaic')
            image_index = i
            if i % 100 == 0:
                print('IMAGE_POSE_SYNC', i, f'{stamps[i]:.9f}', flush=True)
            (args.audit/'image_playback_status.json').write_text(json.dumps({'group': i, 'stamp_sec': image_stamp.sec, 'stamp_nanosec': image_stamp.nanosec, 'image_read_publish_s': time.monotonic()-tick}))
        covered = bool(data['covered'][i]); msgs = MarkerArray()
        for cid in range(4):
            m = marker('current_frustum', cid, Marker.LINE_LIST, colors[cid], .08)
            label = marker('current_label', cid, Marker.TEXT_VIEW_FACING, colors[cid], .5)
            if covered:
                T = data['cameras'][i, cid]; c = T[:3, 3]; quad = corners @ T[:3, :3].T + c
                for j in range(4): m.points.extend([point(c), point(quad[j]), point(quad[j]), point(quad[(j+1)%4])])
                label.pose.position = point(c + [0, 0, 2.5+cid*.5]); label.text = f'cam{cid}'
            else:
                m.action = Marker.DELETE; label.action = Marker.DELETE
            msgs.markers.extend([m, label])
        text = marker('status', 0, Marker.TEXT_VIEW_FACING, (1., 1., 1.), 1.8)
        text.pose.position = point(np.nanmedian(data['base'][valid, :3, 3], axis=0) + [0, 0, 16])
        text.text = (f'{source_label}\n'
                     f'Image group {i+1}/{len(stamps)} | bag {stamps[i]-start:.2f}s | {args.rate:g}x\n'
                     f'{"POSE AVAILABLE" if covered else "MISSING POSE - NO EXTRAPOLATION"}\n'
                     f'Available {report["groups_with_pose"]} | missing {report["groups_missing_pose"]}\n'
                     'Orange: review flags | Cyan: GNSS antenna\n'
                     'Cam0 red / Cam1 green / Cam2 blue / Cam3 purple')
        msgs.markers.append(text)
        ns = int(round(stamps[i]*1e9))
        for m in msgs.markers:
            if image_source is not None:
                # Retain the original integer timestamp; float epoch conversion loses nanoseconds.
                m.header.stamp = last_image_stamp
            else:
                m.header.stamp.sec = ns//10**9; m.header.stamp.nanosec = ns%10**9
        current_pub.publish(msgs)
    node.create_timer(.1, update)
    # Republishing also makes reconnecting RViz reliable without a ROS clock.
    node.create_timer(5., lambda: (mp.publish(cloud), markers_pub.publish(all_markers), path_pub.publish(path)))
    print('POSE_VIEWER_READY', json.dumps({'groups':len(stamps), 'poses':len(valid)*4, 'rate':args.rate}), flush=True)
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
