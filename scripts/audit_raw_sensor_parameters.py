"""Read-only bag timestamp, calibration and raw GNSS audit; does not publish ROS data."""
import argparse,json,pathlib,struct
import cv2,numpy as np,yaml
from rgb_core import load_rig,decode_mosaic,split_cameras


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--session',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--cached',action='store_true');args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import NavSatFix,Image,PointCloud2
    from std_msgs.msg import String
    root=args.session;metadata=yaml.safe_load((root/'bag/metadata.yaml').read_text())['rosbag2_bagfile_information'];topics=['/Data_Cam_Global',*[f'/point_cloud_hasco{i}' for i in range(1,5)],'/imu/data','/gnss/fix_fixed','/gnss/nmea_gga'];times={t:[] for t in topics};gnss=[];gga=[];cloud_samples=[];image_checked=False;t0=metadata['starting_time']['nanoseconds_since_epoch']*1e-9
    cameras,lidars,quadrants,status=load_rig(root/'calibration')
    records=metadata['files']
    if args.cached:
        cached=np.load(args.output/'raw_timestamps.npz');times={k:cached[k].tolist() for k in cached.files};gnss=np.loadtxt(args.output/'raw_gnss.csv',delimiter=',',skiprows=1,ndmin=2).tolist();gga=json.loads((args.output/'raw_gga.json').read_text());records=[r for r in records if r['starting_time']['nanoseconds_since_epoch']*1e-9<=t0+276 and r['starting_time']['nanoseconds_since_epoch']*1e-9+r['duration']['nanoseconds']*1e-9>=t0+275]
    for record in records:
        reader=rosbag2_py.SequentialReader();reader.open(rosbag2_py.StorageOptions(uri=str(root/'bag'/record['path']),storage_id='mcap'),rosbag2_py.ConverterOptions('',''));reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
        while reader.has_next():
            topic,buf,receipt=reader.read_next()
            if args.cached and topic in ['/gnss/nmea_gga','/gnss/fix_fixed','/imu/data']:continue
            if topic=='/gnss/nmea_gga':
                text=deserialize_message(buf,String).data;gga.append({'receipt_s':receipt*1e-9,'text':text});continue
            endian='<' if buf[1]==1 else '>';sec,ns=struct.unpack_from(endian+'iI',buf,4);stamp=sec+ns*1e-9
            if not args.cached:times[topic].append((stamp,receipt*1e-9))
            if topic=='/gnss/fix_fixed':
                m=deserialize_message(buf,NavSatFix);gnss.append([stamp,receipt*1e-9,m.latitude,m.longitude,m.altitude,m.status.status,m.position_covariance_type,*m.position_covariance])
            if topic=='/Data_Cam_Global' and not image_checked and stamp-t0>275:
                m=deserialize_message(buf,Image);image=decode_mosaic(m);tiles=split_cameras(image,quadrants,cameras);cv2.imwrite(str(args.output/'raw_mosaic.jpg'),image)
                for cid,tile in tiles.items():cv2.imwrite(str(args.output/f'raw_camera{cid}.jpg'),tile)
                image_info={'stamp_s':stamp,'encoding':m.encoding,'storage_width':m.width,'storage_height':m.height,'step':m.step,'bytes':len(m.data),'decoded_shape':list(image.shape),'frame_id':m.header.frame_id};image_checked=True
            if topic.startswith('/point_cloud_hasco') and len(cloud_samples)<4 and stamp-t0>275 and topic not in [v['topic'] for v in cloud_samples]:
                m=deserialize_message(buf,PointCloud2);fields={f.name:f for f in m.fields};f=fields['timestamp'];formats={6:'u4',7:'f4',8:'f8'};a=np.ndarray((m.height,m.width),dtype=endian+formats[f.datatype],buffer=m.data,offset=f.offset,strides=(m.row_step,m.point_step)).ravel();cloud_samples.append({'topic':topic,'header_stamp_s':stamp,'frame_id':m.header.frame_id,'point_time_field':{'datatype':f.datatype,'offset':f.offset},'raw_point_time_min_max':[float(a.min()),float(a.max())],'point_time_assumed_unit':'microseconds'})
            if args.cached and image_checked and len(cloud_samples)==4:break
        print('AUDITED',record['path'],flush=True)
    np.savez_compressed(args.output/'raw_timestamps.npz',**{k:np.array(v) for k,v in times.items()});np.savetxt(args.output/'raw_gnss.csv',gnss,delimiter=',',header='stamp_s,receipt_s,latitude,longitude,altitude,status,covariance_type,c00,c01,c02,c10,c11,c12,c20,c21,c22',comments='');(args.output/'raw_gga.json').write_text(json.dumps(gga))
    stats={}
    for topic,values in times.items():
        if not values:continue
        a=np.array(values);dt=np.diff(a[:,0]);stats[topic]={'count':len(a),'header_start_end_s':a[[0,-1],0].tolist(),'nonpositive_intervals':int((dt<=0).sum()),'interval_p50_p95_max_s':np.quantile(dt,[.5,.95,1]).tolist(),'receipt_minus_header_p50_p95_s':np.quantile(a[:,1]-a[:,0],[.5,.95]).tolist()}
    camera=np.array(times['/Data_Cam_Global'])[:,0];sync={}
    for topic in topics[1:6]:
        a=np.array(times[topic])[:,0];indices=np.searchsorted(a,camera);left=np.clip(indices-1,0,len(a)-1);right=np.clip(indices,0,len(a)-1);selected=np.where(abs(a[left]-camera)<abs(a[right]-camera),left,right);offset=(a[selected]-camera)*1000;sync[topic]={'signed_offset_p0_p50_p95_p100_ms':np.quantile(offset,[0,.5,.95,1]).tolist(),'outside_20ms':int((abs(offset)>20).sum())}
    intrinsics={};rig=json.loads((root/'calibration/results/rig_cam0_with_lidars_paired_1203.json').read_text());session=json.loads((root/'calibration/config/aprilgrid_session.json').read_text())
    for cid,c in cameras.items():
        R=c['reference_from_camera'][:3,:3];intrinsics[str(cid)]={'image_size':list(c['size']),'K':c['K'].tolist(),'D':c['D'].tolist(),'matches_session_K_D':bool(np.allclose(c['K'],session['cameras'][str(cid)]['K']) and np.allclose(c['D'],session['cameras'][str(cid)]['D'])),'rotation_orthogonality_error':float(abs(R.T@R-np.eye(3)).max()),'rotation_determinant':float(np.linalg.det(R))}
    report={'bag_start_s':t0,'calibration_status':status,'quadrants':quadrants,'image_sample':image_info,'cloud_samples':cloud_samples,'topic_timestamps':stats,'nearest_camera_offsets':sync,'intrinsics_and_transform_checks':intrinsics,'limitations':['Header agreement does not prove exposure timing/hardware trigger or zero sensor latency.','Candidate calibration matrix consistency does not independently verify physical extrinsic accuracy.','NavSatFix status is not an RTK fixed-quality code; inspect raw GGA quality.']};(args.output/'raw_audit.json').write_text(json.dumps(report,indent=2));print('READY',json.dumps(report),flush=True)


if __name__=='__main__':main()
