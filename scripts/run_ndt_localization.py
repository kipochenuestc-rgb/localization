"""Run quality-gated NDT + robot_localization EKF and show fixed-map replay in RViz."""
import argparse,csv,json,os,pathlib,signal,subprocess,time
import numpy as np,yaml
from scipy.spatial.transform import Rotation
from pcd_io import read_pcd
from rgb_core import load_rig
from gnss_core import LocalENU

ROOT=pathlib.Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',type=pathlib.Path,required=True);ap.add_argument('--run',default='ndt_ekf_local');ap.add_argument('--no-gnss',action='store_true');ap.add_argument('--no-rviz',action='store_true');ap.add_argument('--images',action='store_true');ap.add_argument('--exit-after-replay',action='store_true');ap.add_argument('--pace',type=float,default=.5,help='Minimum wall time per scan; waits for NDT if slower');args=ap.parse_args()
    if pathlib.Path(args.run).name!=args.run:raise ValueError('run must be a folder name')
    out=ROOT/'runs'/args.run;out.mkdir(exist_ok=False);manifest=json.loads((args.input/'manifest.json').read_text());M=np.array(manifest['initial_map_from_reference']);euler=Rotation.from_matrix(M[:3,:3]).as_euler('xyz');params={'/**':{'ros__parameters':{'map_file':manifest['map_file'],'initial_pose':[*M[:3,3].tolist(),*euler.tolist()],'use_sim_time':True}}};(out/'ndt.yaml').write_text(yaml.safe_dump(params))
    ekf=yaml.safe_load((ROOT/'config/ndt_ekf.yaml').read_text());cfg=ekf['ekf_filter_node']['ros__parameters'];cfg.update({'odom_frame':'localization_odom','world_frame':'localization_odom'})
    if not args.no_gnss:cfg.update({'odom1':'/localization/gnss','odom1_config':[True,True,False,False,False,False,False,False,False,False,False,False,False,False,False],'odom1_queue_size':20,'odom1_pose_rejection_threshold':5.})
    (out/'ekf.yaml').write_text(yaml.safe_dump(ekf));children={};handles=[]
    def launch(name,cmd):
        log=(out/f'{name}.log').open('w');handles.append(log);p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);children[name]=p;return p
    ndt_exe=str(pathlib.Path(os.environ.get('SLAM_NDT_WORKSPACE',str(pathlib.Path.home()/'ndt_ws')))/'install/ndt_localization/lib/ndt_localization/ndt_localizer')
    launch('ndt',[ndt_exe,'--ros-args','--params-file',str(out/'ndt.yaml')]);launch('ekf',['ros2','run','robot_localization','ekf_node','--ros-args','--params-file',str(out/'ekf.yaml'),'-r','odometry/filtered:=/localization/ekf'])
    launch('map_odom_tf',['ros2','run','tf2_ros','static_transform_publisher','--x','0','--y','0','--z','0','--roll','0','--pitch','0','--yaw','0','--frame-id','map','--child-frame-id','localization_odom'])
    if not args.no_rviz:launch('rviz',['python3','-u',str(ROOT/'scripts/launch_rviz.py'),'--repair','--config',str(ROOT/'config/ndt_localization.rviz')])
    (out/'processes.json').write_text(json.dumps({k:p.pid for k,p in children.items()},indent=2))
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import PointCloud2,PointField,Imu
    from nav_msgs.msg import Odometry,Path
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg import String
    from rosgraph_msgs.msg import Clock
    from builtin_interfaces.msg import Time
    rclpy.init();node=rclpy.create_node('ndt_replay_controller');clock_pub=node.create_publisher(Clock,'/clock',100);imu_pub=node.create_publisher(Imu,'/localization/imu',QoSProfile(depth=2000,reliability=ReliabilityPolicy.RELIABLE));cloud_pub=node.create_publisher(PointCloud2,'/localization/input_scan',2);gnss_pub=node.create_publisher(Odometry,'/localization/gnss',20);path_pub=node.create_publisher(Path,'/localization/path',10);gps_path_pub=node.create_publisher(Path,'/localization/gnss_path',10)
    trajectory=Path();trajectory.header.frame_id='map';gps_path=Path();gps_path.header.frame_id='map';latest=None;ndt_status=[];ekf_rows=[];path_wall=0.;statusfile=(out/'ndt_quality.jsonl').open('w');csvfile=(out/'ekf_trajectory.csv').open('w',newline='');writer=csv.writer(csvfile);writer.writerow(['stamp_s','x','y','z','qx','qy','qz','qw','var_x','var_y','var_z'])
    def stamp(t):
        ns=int(round(t*1e9));return Time(sec=ns//10**9,nanosec=ns%10**9)
    def set_clock(t):m=Clock();m.clock=stamp(t);clock_pub.publish(m)
    def odom(m):
        nonlocal latest,path_wall
        latest=m;p=m.pose.pose.position;q=m.pose.pose.orientation;t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9;row=[t,p.x,p.y,p.z,q.x,q.y,q.z,q.w,*[m.pose.covariance[k] for k in [0,7,14]]];writer.writerow(row);ekf_rows.append(row)
        if not trajectory.poses or t-(trajectory.poses[-1].header.stamp.sec+trajectory.poses[-1].header.stamp.nanosec*1e-9)>=.095:
            pose=PoseStamped();pose.header=m.header;pose.header.frame_id='map';pose.pose=m.pose.pose;trajectory.poses.append(pose);trajectory.header.stamp=m.header.stamp
        if time.monotonic()-path_wall>.5:path_pub.publish(trajectory);path_wall=time.monotonic()
    def quality(m):
        s=json.loads(m.data);ndt_status.append(s);statusfile.write(json.dumps(s)+'\n');statusfile.flush();print(m.data,flush=True)
    node.create_subscription(Odometry,'/localization/ekf',odom,1000);node.create_subscription(String,'/localization/ndt_status',quality,100)
    live_images=None
    if args.images:
        # MCAP reads can retain Python's GIL. Keep previews in another process
        # so slow disk access cannot stall sensor delivery or EKF clock updates.
        launch('images',['python3','-u',str(ROOT/'scripts/live_localization_images.py'),'--input',str(args.input)])
    imu=np.load(args.input/'imu.npz')['imu'];frames=manifest['frames'];first=frames[0]['stamp_s'];k=int(np.searchsorted(imu[:,0],first));fixes=json.loads((args.input/'gnss.json').read_text());gk=0;alignment=json.loads((pathlib.Path(manifest['map_run'])/'gnss_alignment.json').read_text());local=LocalENU(*alignment['reference_wgs84']);R=np.array(alignment['enu_to_map_rotation']);translation=np.array(alignment['enu_to_map_translation']);cameras,_,_,_=load_rig(manifest['calibration_dir']);mount=json.loads((pathlib.Path(manifest['calibration_dir'])/'config/antenna_mount.json').read_text());C=cameras[0]['reference_from_camera'];lever=C[:3,:3]@np.array(mount['antenna_origin_in_camera0_m'])+C[:3,3];gps_counts={'fused':0,'rejected':0};start=time.monotonic()
    try:
        set_clock(first-.02);deadline=time.monotonic()+30
        while cloud_pub.get_subscription_count()==0 or imu_pub.get_subscription_count()==0:
            rclpy.spin_once(node,timeout_sec=.1)
            if any(p.poll() is not None for name,p in children.items() if name in ['ndt','ekf']):raise RuntimeError('A localization node exited; inspect logs')
            if time.monotonic()>deadline:raise RuntimeError('Localization subscriptions not ready')
        for index,frame in enumerate(frames):
            wall=time.monotonic();t=frame['stamp_s']
            deadline=time.monotonic()+180
            while True:
                try:pcd=read_pcd(args.input/frame['file']);break
                except (FileNotFoundError,ValueError):
                    if not manifest.get('streaming_input'):raise
                    if time.monotonic()>deadline:raise RuntimeError('Timed out waiting for prepared scan '+frame['file'])
                    rclpy.spin_once(node,timeout_sec=.1)
            while k<len(imu) and imu[k,0]<=t:
                set_clock(imu[k,0]);m=Imu();m.header.stamp=stamp(imu[k,0]);m.header.frame_id='localization_base';m.orientation_covariance[0]=-1.;m.linear_acceleration_covariance[0]=-1.;m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z=imu[k,1:].tolist();m.angular_velocity_covariance=[.0001,0.,0.,0.,.0001,0.,0.,0.,.0001];imu_pub.publish(m);k+=1;rclpy.spin_once(node,timeout_sec=.001)
            set_clock(t)
            while gk<len(fixes) and fixes[gk][0]<=t:
                fix=fixes[gk];gk+=1
                if fix[0]<first or latest is None:continue
                q=latest.pose.pose.orientation;rotation=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix();antenna=R@local.convert(*fix[1:4])+translation;position=antenna-rotation@lever;p=latest.pose.pose.position;innovation=np.linalg.norm(position[:2]-[p.x,p.y]);pose=PoseStamped();pose.header.frame_id='map';pose.header.stamp=stamp(fix[0]);pose.pose.position.x,pose.pose.position.y,pose.pose.position.z=antenna.tolist();pose.pose.orientation.w=1.;gps_path.poses.append(pose);gps_path.header.stamp=pose.header.stamp;gps_path_pub.publish(gps_path)
                if args.no_gnss:continue
                if innovation>3.:gps_counts['rejected']+=1;continue
                gps=Odometry();gps.header=pose.header;gps.child_frame_id='localization_base';gps.pose.pose.position.x,gps.pose.pose.position.y,gps.pose.pose.position.z=position.tolist();gps.pose.pose.orientation.w=1.;gps.pose.covariance[0]=gps.pose.covariance[7]=.25;gps.pose.covariance[14]=4.;gnss_pub.publish(gps);gps_counts['fused']+=1
            points=np.column_stack([pcd[n] for n in ['x','y','z']]).astype('<f4');m=PointCloud2();m.header.stamp=stamp(t);m.header.frame_id='localization_base';m.height=1;m.width=len(points);m.fields=[PointField(name=n,offset=j*4,datatype=7,count=1) for j,n in enumerate(['x','y','z'])];m.point_step=12;m.row_step=len(points)*12;m.is_dense=True;m.data=points.tobytes();before=len(ndt_status);cloud_pub.publish(m);deadline=time.monotonic()+45
            while len(ndt_status)==before:
                rclpy.spin_once(node,timeout_sec=.02)
                if time.monotonic()>deadline:raise RuntimeError('NDT did not return a quality status within 45s')
            # Let the EKF consume the same-stamp NDT measurement before moving the replay clock.
            for _ in range(8):rclpy.spin_once(node,timeout_sec=.01)
            if live_images is not None and ekf_rows:live_images.update(ndt_status[-1],ekf_rows)
            remaining=args.pace-(time.monotonic()-wall)
            if remaining>0:
                until=time.monotonic()+remaining
                while time.monotonic()<until:rclpy.spin_once(node,timeout_sec=min(.05,until-time.monotonic()))
            csvfile.flush();(out/'status.json').write_text(json.dumps({'state':'replaying','scan':index+1,'total':len(frames),'accepted':sum(s['accepted'] for s in ndt_status),'last':ndt_status[-1]},indent=2))
        for _ in range(20):rclpy.spin_once(node,timeout_sec=.02)
        report={'state':'complete','scans':len(frames),'accepted':sum(s['accepted'] for s in ndt_status),'rejected':sum(not s['accepted'] for s in ndt_status),'ekf_poses':len(ekf_rows),'gnss':gps_counts,'elapsed_s':time.monotonic()-start,'initialization':manifest['initialization'],'limitations':manifest['limitations']};(out/'status.json').write_text(json.dumps(report,indent=2));print('LOCALIZATION_COMPLETE',json.dumps(report),flush=True)
        # Keep the map, final trajectory and RViz alive for inspection; no more measurements.
        node.create_timer(1.,lambda:(path_pub.publish(trajectory),gps_path_pub.publish(gps_path)))
        if not args.exit_after_replay:rclpy.spin(node)
    except KeyboardInterrupt:pass
    finally:
        if live_images is not None:live_images.close()
        csvfile.close();statusfile.close()
        for p in children.values():
            if p.poll() is None:os.killpg(p.pid,signal.SIGINT)
        node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
