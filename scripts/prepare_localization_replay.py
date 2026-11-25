"""Prepare raw four-LiDAR scans and bias-corrected gyro for map localization (no SLAM poses after initialization)."""
import argparse,bisect,json,pathlib
import numpy as np,yaml
from scipy.spatial.transform import Rotation,Slerp
from rgb_core import Poses,cloud_arrays,transform
from sensor_geometry import load_lidar_transforms


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--bag',type=pathlib.Path,required=True);ap.add_argument('--map-run',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--start',type=float,default=275);ap.add_argument('--duration',type=float,default=20);args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True);(args.output/'scans').mkdir()
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    metadata=yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information'];t0=metadata['starting_time']['nanoseconds_since_epoch']*1e-9;lo=t0+args.start;hi=lo+args.duration
    topics=[f'/point_cloud_hasco{i}' for i in range(1,5)]+['/imu/data','/Data_Cam_Global','/gnss/fix_fixed'];clouds={i:[] for i in range(4)};imu=[];images=[];gnss=[]
    for file in metadata['files']:
        begin=file['starting_time']['nanoseconds_since_epoch']*1e-9;end=begin+file['duration']['nanoseconds']*1e-9
        if begin>hi+1 or end<lo-1:continue
        reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(args.bag/file['path']),storage_id='mcap'),rosbag2_py.ConverterOptions('',''));types={t.name:get_message(t.type) for t in reader.get_all_topics_and_types()};reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
        while reader.has_next():
            topic,buf,_=reader.read_next();m=deserialize_message(buf,types[topic]);t=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
            if not lo-.3<=t<=hi+.3:continue
            if topic=='/imu/data':imu.append([t,m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z])
            elif topic=='/Data_Cam_Global':images.append(t)
            elif topic=='/gnss/fix_fixed':gnss.append([t,m.longitude,m.latitude,m.altitude,*m.position_covariance])
            else:clouds[int(topic[-1])-1].append((t,m))
        print('READ',file['path'],flush=True)
    imu=np.array(sorted(imu,key=lambda x:x[0]));bias=json.loads((pathlib.Path(__file__).resolve().parents[1]/'config/profiles/session_20261124_imu_bias.json').read_text());imu[:,1:]-=bias['gyro_bias_rad_s'];gyro=[Rotation.identity()]
    for i in range(1,len(imu)):gyro.append(gyro[-1]*Rotation.from_rotvec((imu[i,1:]+imu[i-1,1:])*.5*(imu[i,0]-imu[i-1,0])))
    rotations=Slerp(imu[:,0]-imu[0,0],Rotation.from_quat([r.as_quat() for r in gyro]));transforms=load_lidar_transforms(args.bag.parent/'calibration');times={i:np.array([t for t,m in v]) for i,v in clouds.items()};frames=[]
    for stamp,m in clouds[0]:
        if not lo<=stamp<=hi:continue
        group=[(stamp,m)]
        for i in range(1,4):
            k=int(np.argmin(abs(times[i]-stamp)))
            if abs(times[i][k]-stamp)>.02:break
            group.append(clouds[i][k])
        if len(group)!=4:continue
        parts=[];reference=rotations(stamp-imu[0,0]).as_matrix()
        for i,(t,message) in enumerate(group):
            xyz,relative=cloud_arrays(message);distance=np.linalg.norm(xyz,axis=1);valid=np.isfinite(xyz).all(1)&(distance>1.5)&(distance<70)&(relative>=0)&(relative<.15);xyz=transform(transforms[i],xyz[valid]);pt=t+relative[valid];R=rotations(pt-imu[0,0]).as_matrix();xyz=np.einsum('ij,njk,nk->ni',reference.T,R,xyz);parts.append(xyz)
        x=np.concatenate(parts);_,k=np.unique(np.floor(x/.3).astype('i4'),axis=0,return_index=True);x=x[k].astype('<f4');name=f'scans/{len(frames):05d}.pcd'
        with (args.output/name).open('wb') as f:f.write(f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(x)}\nHEIGHT 1\nPOINTS {len(x)}\nDATA binary\n'.encode());f.write(x.tobytes())
        frames.append({'stamp_s':stamp,'file':name,'points':len(x)})
    if not frames:raise RuntimeError('No complete scans')
    initial=Poses(args.map_run/'map/transformations.pcd').at(frames[0]['stamp_s']);np.savez_compressed(args.output/'imu.npz',imu=imu);(args.output/'images.json').write_text(json.dumps(images));(args.output/'gnss.json').write_text(json.dumps(gnss));(args.output/'manifest.json').write_text(json.dumps({'frames':frames,'initial_map_from_reference':initial.tolist(),'map_file':str(args.map_run/'map/SurfMap.pcd'),'map_run':str(args.map_run),'bag':str(args.bag),'bag_start_s':t0,'calibration_dir':str(args.bag.parent/'calibration'),'initialization':'One saved SLAM pose at first scan only: warm-start localization, not global kidnapped-robot localization.','deskew':'Bias-corrected IMU rotation only; no SLAM trajectory or translation used to deskew.','limitations':['Same-data map replay is not independent accuracy.','No translation deskew yet; motion within a scan can contribute residual error.']},indent=2));print('READY',len(frames),'scans',len(imu),'IMUs',flush=True)


if __name__=='__main__':main()
