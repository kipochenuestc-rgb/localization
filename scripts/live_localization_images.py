"""Display original images beside the currently estimated localization pose."""
import time
import threading
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from localization_paths import SLAM_ROOT


class LiveLocalizationImages:
    def __init__(self, node, manifest):
        from sensor_msgs.msg import Image
        from visualization_msgs.msg import MarkerArray
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        from bag_camera_images import BagCameraImages
        from rgb_core import load_rig
        from builtin_interfaces.msg import Time
        self.Image=Image;self.Time=Time;self.node=node
        self.cameras,_,quadrants,_=load_rig(manifest['calibration_dir'])
        bag=Path(manifest['bag'])
        raw=np.load(SLAM_ROOT/'runs/parameter_accuracy_audit/raw_timestamps.npz')['/Data_Cam_Global']
        self.stamps=raw[:,0];self.reader=BagCameraImages(bag,self.cameras,quadrants,raw)
        qos=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE)
        self.publisher=node.create_publisher(Image,'/localization/cameras/image',qos)
        self.markers=node.create_publisher(MarkerArray,'/localization/cameras',qos)
        self.last_wall=-1.;self.t0=manifest['bag_start_s']
        self.pending=None;self.event=threading.Event();self.closed=False
        self.thread=threading.Thread(target=self.worker,daemon=True);self.thread.start()

    def update(self, status, ekf_rows):
        self.pending=(dict(status),list(ekf_rows[-1]));self.event.set()

    def close(self):
        self.closed=True;self.event.set();self.thread.join(timeout=5.)

    def worker(self):
        while not self.closed:
            self.event.wait(timeout=.2)
            if self.closed:return
            if self.pending is None:continue
            status,row=self.pending;self.pending=None;self.event.clear()
            try:self.publish(status,[row])
            except Exception as exc:self.node.get_logger().warning(f'Image preview failed: {exc}')
            self.event.wait(timeout=.3)

    def publish(self, status, ekf_rows):
        if time.monotonic()-self.last_wall<.7:return
        from bag_camera_images import labeled_mosaic
        from visualization_msgs.msg import Marker, MarkerArray
        from geometry_msgs.msg import Point
        stamp=status['stamp_s'];i=int(np.argmin(abs(self.stamps-stamp)))
        if abs(self.stamps[i]-stamp)>.02:return
        image_stamp,tiles=self.reader.at(self.stamps[i]);pixels=np.ascontiguousarray(labeled_mosaic(tiles))
        image=self.Image();image.header.stamp=image_stamp;image.header.frame_id='camera_mosaic';image.height,image.width=pixels.shape[:2];image.encoding='bgr8';image.step=image.width*3;image.data=pixels.tobytes();self.publisher.publish(image)
        # Live poses refer to the NDT scan, typically about 2 ms after the image.
        # Final exports interpolate EKF at the actual image timestamp.
        pose=status['pose'] if status['accepted'] else ekf_rows[-1][1:8]
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(pose[3:]).as_matrix();T[:3,3]=pose[:3]
        corners=np.array([[-1.,-.7,1.4],[1.,-.7,1.4],[1.,.7,1.4],[-1.,.7,1.4]])
        colors=[(1.,.35,.2),(.25,1.,.35),(.2,.55,1.),(1.,.2,1.)];msg=MarkerArray()
        def point(p):return Point(x=float(p[0]),y=float(p[1]),z=float(p[2]))
        for cid in range(4):
            C=T@self.cameras[cid]['reference_from_camera'];center=C[:3,3];quad=corners@C[:3,:3].T+center
            m=Marker();m.header.frame_id='map';m.ns='live_camera';m.id=cid;m.type=Marker.LINE_LIST;m.action=Marker.ADD;m.pose.orientation.w=1.;m.scale.x=.1;m.color.r,m.color.g,m.color.b=colors[cid];m.color.a=1.
            for j in range(4):m.points.extend([point(center),point(quad[j]),point(quad[j]),point(quad[(j+1)%4])])
            msg.markers.append(m)
        text=Marker();text.header.frame_id='map';text.ns='live_status';text.id=0;text.type=Marker.TEXT_VIEW_FACING;text.pose.orientation.w=1.;text.pose.position=point(T[:3,3]+[0,0,6]);text.scale.z=1.;text.color.r=text.color.g=text.color.b=text.color.a=1.
        text.text=f'NDT + EKF | bag {stamp-self.t0:.1f}s\nNDT {"accepted" if status["accepted"] else "REJECTED: EKF prediction"} | inliers {status["inlier_fraction"]:.2f}\nImage to scan offset {(stamp-self.stamps[i])*1000:.2f} ms'
        msg.markers.append(text);self.markers.publish(msg);self.last_wall=time.monotonic()


def main():
    import argparse
    import rclpy
    from nav_msgs.msg import Odometry
    from std_msgs.msg import String
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=Path,required=True);args=ap.parse_args()
    manifest=json.loads((args.input/'manifest.json').read_text());rclpy.init();node=rclpy.create_node('localization_camera_preview')
    preview=LiveLocalizationImages(node,manifest);latest=None
    def odom(m):
        nonlocal latest
        p=m.pose.pose.position;q=m.pose.pose.orientation
        latest=[m.header.stamp.sec+m.header.stamp.nanosec*1e-9,p.x,p.y,p.z,q.x,q.y,q.z,q.w]
    def quality(m):
        s=json.loads(m.data)
        if s['accepted']:preview.update(s,[[s['stamp_s'],*s['pose']]])
        elif latest is not None:preview.update(s,[latest])
    node.create_subscription(Odometry,'/localization/ekf',odom,1)
    node.create_subscription(String,'/localization/ndt_status',quality,1)
    try:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:preview.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
