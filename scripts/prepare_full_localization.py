"""Stream a whole recording into rotation-deskewed localization scans with bounded memory."""
import argparse
import json
import pathlib
import struct
from collections import deque

import numpy as np
import yaml
from scipy.spatial.transform import Rotation, Slerp
from rgb_core import Poses, cloud_arrays, transform
from sensor_geometry import load_lidar_transforms
from localization_paths import SLAM_ROOT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bag', type=pathlib.Path, required=True)
    ap.add_argument('--map-run', type=pathlib.Path, required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--start-offset', type=float, default=0., help='Seconds from the bag start')
    ap.add_argument('--duration', type=float, help='Optional localization window length in seconds')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    if args.start_offset < 0 or (args.duration is not None and args.duration <= 0):
        ap.error('start offset must be nonnegative and duration must be positive')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    (args.output/'scans').mkdir(exist_ok=args.resume)
    import rosbag2_py as rb
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Imu, NavSatFix, PointCloud2
    root = SLAM_ROOT
    metadata = yaml.safe_load((args.bag/'metadata.yaml').read_text())['rosbag2_bagfile_information']
    t0 = metadata['starting_time']['nanoseconds_since_epoch']*1e-9
    bag_end = max(
        file['starting_time']['nanoseconds_since_epoch']*1e-9
        + file['duration']['nanoseconds']*1e-9
        for file in metadata['files']
    )
    window_start = t0 + args.start_offset
    window_end = min(window_start + args.duration, bag_end) if args.duration is not None else bag_end
    if window_start >= bag_end or window_end <= window_start:
        ap.error('selected time window does not overlap the bag')
    window_signature = {
        'bag': str(args.bag.resolve()),
        'map_run': str(args.map_run.resolve()),
        'start_offset_s': args.start_offset,
        'duration_s': args.duration,
    }
    signature_path = args.output/'preparation.json'
    if args.resume and signature_path.exists():
        if json.loads(signature_path.read_text()) != window_signature:
            raise ValueError('Resume settings differ from the existing preparation')
    elif args.resume and any((args.output/'scans').glob('*.pcd')) and args.start_offset != 0:
        raise ValueError('Cannot resume an older preparation with a nonzero start offset')
    else:
        signature_path.write_text(json.dumps(window_signature, indent=2))

    def reader_for(file, topics):
        reader = rb.SequentialReader()
        reader.open(rb.StorageOptions(uri=str(args.bag/file['path']), storage_id='mcap'), rb.ConverterOptions('', ''))
        reader.set_filter(rb.StorageFilter(topics=topics))
        begin = file['starting_time']['nanoseconds_since_epoch']*1e-9
        if begin < window_start-1.0:
            reader.seek(int((window_start-1.0)*1e9))
        return reader
    def header(buf):
        sec, ns = struct.unpack_from(('<' if buf[1] == 1 else '>')+'iI', buf, 4)
        return sec+ns*1e-9
    imu, images, gnss = [], [], []
    cached_imu=(args.output/'imu.npz').exists() and args.resume
    if cached_imu:
        imu=np.load(args.output/'imu.npz')['imu']
        images=json.loads((args.output/'images.json').read_text());gnss=json.loads((args.output/'gnss.json').read_text())
    selected_files = []
    for file in metadata['files']:
        begin = file['starting_time']['nanoseconds_since_epoch']*1e-9
        end = begin + file['duration']['nanoseconds']*1e-9
        if end >= window_start-1.0 and begin <= window_end+0.2:
            selected_files.append(file)
    for i, file in enumerate([] if cached_imu else selected_files):
        reader = reader_for(file, ['/imu/data', '/gnss/fix_fixed', '/Data_Cam_Global'])
        while reader.has_next():
            topic, buf, _ = reader.read_next(); t = header(buf)
            if topic == '/Data_Cam_Global' and window_start <= t <= window_end:
                images.append(t)
            elif topic == '/imu/data' and window_start-1.0 <= t <= window_end+0.2:
                m = deserialize_message(buf, Imu); imu.append([t, m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
            elif topic == '/gnss/fix_fixed' and window_start <= t <= window_end:
                m = deserialize_message(buf, NavSatFix)
                if m.status.status >= 0 and np.isfinite([m.longitude, m.latitude, m.altitude]).all():
                    gnss.append([t, m.longitude, m.latitude, m.altitude, *m.position_covariance])
        (args.output/'status.json').write_text(json.dumps({'phase':'imu_and_image_timestamps','file':i+1,'total_files':len(selected_files)}))
        print('READ_IMU', i+1, len(imu), flush=True)
    imu = np.array(sorted(imu, key=lambda x:x[0]))
    if len(imu)<2 or np.any(np.diff(imu[:,0])<=0): raise ValueError('Invalid gyro timeline')
    bias = json.loads((root/'config/profiles/session_20261124_imu_bias.json').read_text())
    if not cached_imu:imu[:,1:] -= bias['gyro_bias_rad_s']
    gyro = [Rotation.identity()]
    for i in range(1,len(imu)):
        gyro.append(gyro[-1]*Rotation.from_rotvec((imu[i,1:]+imu[i-1,1:])*.5*(imu[i,0]-imu[i-1,0])))
    rotation = Slerp(imu[:,0]-imu[0,0], Rotation.from_quat([r.as_quat() for r in gyro]))
    transforms = load_lidar_transforms(args.bag.parent/'calibration')
    np.savez_compressed(args.output/'imu.npz', imu=imu)
    (args.output/'images.json').write_text(json.dumps(sorted(images)))
    (args.output/'gnss.json').write_text(json.dumps(sorted(gnss, key=lambda x:x[0])))
    frames = []; queues = [deque() for _ in range(4)]; skipped = 0
    resume_receipt=-np.inf
    if args.resume:
        files=sorted((args.output/'scans').glob('*.pcd'))
        audit=root/'runs/parameter_accuracy_audit'
        if abs(json.loads((audit/'raw_audit.json').read_text())['bag_start_s']-t0)>.001:raise ValueError('Timestamp cache is for another bag')
        old_status=json.loads((args.output/'status.json').read_text())
        if old_status.get('skipped_incomplete',0):raise ValueError('Cannot infer resume times when incomplete groups were skipped')
        times=np.load(audit/'raw_timestamps.npz')['/point_cloud_hasco1']
        times=times[(times[:,0]>=window_start)&(times[:,0]<=window_end)]
        from pcd_io import read_pcd
        for i,file in enumerate(files):
            if file.name!=f'{i:05d}.pcd':raise ValueError('Noncontiguous scan files')
            cloud=read_pcd(file)
            frames.append({'stamp_s':float(times[i,0]),'file':f'scans/{file.name}','points':len(cloud)})
        if files:resume_receipt=float(times[len(files)-1,1])
        print('RESUME_SCANS',len(frames),flush=True)
    def voxel_indices(xyz,leaf):
        grid=np.floor(xyz/leaf).astype(np.int64)+1024
        if np.any(grid<0) or np.any(grid>=2048):raise ValueError('Voxel hash extent exceeded')
        code=grid[:,0]+(grid[:,1]<<11)+(grid[:,2]<<22)
        return np.unique(code,return_index=True)[1]
    def process(group):
        stamp = group[0][0]
        if not window_start <= stamp <= window_end:return
        if frames and stamp<=frames[-1]['stamp_s']:return
        if not imu[0,0]<=stamp<=imu[-1,0]: return
        reference = rotation(stamp-imu[0,0]).as_matrix(); parts = []; empty_lidars=[]
        for lid, (t, m) in enumerate(group):
            xyz, relative = cloud_arrays(m); distance = np.linalg.norm(xyz,axis=1); pt = t+relative
            valid = np.isfinite(xyz).all(1)&(distance>1.5)&(distance<70)&(relative>=0)&(relative<.15)&(pt>=imu[0,0])&(pt<=imu[-1,0])
            # Retain one measured point per small voxel before costly per-point SLERP.
            # Its own acquisition time is retained; no time averaging is introduced.
            selected=np.flatnonzero(valid);selected=selected[voxel_indices(xyz[selected],.2)]
            if not len(selected):
                empty_lidars.append(lid+1)
                continue
            p = transform(transforms[lid],xyz[selected]); R = rotation(pt[selected]-imu[0,0]).as_matrix()
            parts.append(np.einsum('ij,njk,nk->ni',reference.T,R,p))
        xyz = np.concatenate(parts) if parts else np.empty((0,3))
        keep = voxel_indices(xyz,.3)
        xyz = xyz[keep].astype('<f4'); name = f'scans/{len(frames):05d}.pcd'
        with (args.output/name).open('wb') as f:
            f.write(f'VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\nWIDTH {len(xyz)}\nHEIGHT 1\nPOINTS {len(xyz)}\nDATA binary\n'.encode()); f.write(xyz.tobytes())
        frames.append({'stamp_s':stamp,'file':name,'points':len(xyz),'empty_lidars':empty_lidars})
        if empty_lidars:print('EMPTY_LIDARS',stamp-t0,empty_lidars,flush=True)
        if len(frames)%100==0:
            status={'phase':'scans','scans':len(frames),'bag_time_s':stamp-t0,'skipped_incomplete':skipped}
            (args.output/'status.json').write_text(json.dumps(status));print('PREPARED',json.dumps(status),flush=True)
    def drain(final=False):
        nonlocal skipped
        while queues[0]:
            stamp = queues[0][0][0]; group=[queues[0][0]]; ready=True
            for queue in queues[1:]:
                while queue and queue[0][0]<stamp-.02:queue.popleft()
                if not queue:
                    if final: ready=False; break
                    return
                if abs(queue[0][0]-stamp)>.02: ready=False;break
                group.append(queue[0])
            queues[0].popleft()
            if ready:
                for q in queues[1:]: q.popleft()
                process(group)
            else:skipped+=1
    topics=[f'/point_cloud_hasco{i}' for i in range(1,5)]
    for file in metadata['files']:
        begin=file['starting_time']['nanoseconds_since_epoch']*1e-9
        end=begin+file['duration']['nanoseconds']*1e-9
        if end < window_start-1.0 or begin > window_end+0.2:continue
        if end<resume_receipt-.1:continue
        reader = reader_for(file,topics)
        if begin<resume_receipt-.1:reader.seek(int((resume_receipt-.1)*1e9))
        while reader.has_next():
            topic,buf,_=reader.read_next();lid=int(topic[-1])-1
            queues[lid].append((header(buf),deserialize_message(buf,PointCloud2)))
            drain()
    drain(final=True)
    if not frames or np.any(np.diff([f['stamp_s'] for f in frames])<=0):raise ValueError('No valid ordered scans')
    initial=Poses(args.map_run/'map/transformations.pcd').at(frames[0]['stamp_s'])
    manifest={'frames':frames,'initial_map_from_reference':initial.tolist(),'map_file':str(args.map_run/'map/SurfMap.pcd'),'map_run':str(args.map_run),'bag':str(args.bag),'bag_start_s':t0,'time_window_s':[window_start,window_end],'start_offset_s':args.start_offset,'duration_s':window_end-window_start,'calibration_dir':str(args.bag.parent/'calibration'),'initialization':'First saved SLAM pose in the selected window only; no later SLAM poses used by localization.','deskew':'Bias-corrected gyro rotation only.','limitations':['Same-data map replay is not independent accuracy.','No translation deskew yet; motion within a scan can contribute residual error.'],'skipped_incomplete':skipped}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.output/'status.json').write_text(json.dumps({'phase':'complete','scans':len(frames),'images':len(images),'imu':len(imu),'skipped':skipped}))
    print('FULL_PREP_READY',len(frames),flush=True)


if __name__=='__main__':main()
