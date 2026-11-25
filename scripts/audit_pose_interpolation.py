"""Compare sparse optimized-keyframe interpolation with dense odometry plus interpolated graph corrections."""
import argparse,json,pathlib
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from rgb_core import Poses,load_rig


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=pathlib.Path,required=True);ap.add_argument('--scene',type=pathlib.Path,required=True);ap.add_argument('--calibration',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args()
    poses=Poses(args.run/'map/transformations.pcd');rows=np.genfromtxt(args.run/'trajectory.csv',delimiter=',',names=True);times=rows['stamp_s'];keep=np.r_[True,np.diff(times)>0];times=times[keep];rows=rows[keep];xyz=np.column_stack([rows[k] for k in ['x','y','z']]);rot=Rotation.from_quat(np.column_stack([rows[k] for k in ['qx','qy','qz','qw']]));slerp=Slerp(times-times[0],rot)
    def dense(t):
        t=np.atleast_1d(t);m=np.tile(np.eye(4),(len(t),1,1));m[:,:3,:3]=slerp(t-times[0]).as_matrix()
        for j in range(3):m[:,j,3]=np.interp(t,times,xyz[:,j])
        return m
    valid=(poses.times>=times[0])&(poses.times<=times[-1]);keytimes=poses.times[valid];correction=poses.at(keytimes)@np.linalg.inv(dense(keytimes));graph_slerp=Slerp(keytimes-keytimes[0],Rotation.from_matrix(correction[:,:3,:3]));views=json.loads((args.scene/'cameras.json').read_text());cameras,_,_,_=load_rig(args.calibration);results=[]
    for v in views:
        t=v['stamp_s'];C=np.eye(4);C[:3,:3]=graph_slerp(t-keytimes[0]).as_matrix()
        for j in range(3):C[j,3]=np.interp(t,keytimes,correction[:,j,3])
        A=poses.at(t)@cameras[v['camera_id']]['reference_from_camera'];B=C@dense([t])[0]@cameras[v['camera_id']]['reference_from_camera'];angle=Rotation.from_matrix(A[:3,:3].T@B[:3,:3]).magnitude();results.append({'image':v['image_path'],'stamp_s':t,'position_difference_m':float(np.linalg.norm(A[:3,3]-B[:3,3])),'rotation_difference_deg':float(np.degrees(angle))})
    report={'dense_odometry_interval_p50_p95_max_s':np.quantile(np.diff(times),[.5,.95,1]).tolist(),'position_difference_p50_p95_max_m':np.quantile([v['position_difference_m'] for v in results],[.5,.95,1]).tolist(),'rotation_difference_p50_p95_max_deg':np.quantile([v['rotation_difference_deg'] for v in results],[.5,.95,1]).tolist(),'per_image':results,'notes':['Difference between two pose interpolation methods, not independent pose ground truth.','Dense online odometry is transported into final map with interpolated keyframe graph corrections.','Keyframe correction interpolation may still be insufficient near discontinuous graph updates.']};(args.output/'pose_interpolation_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='per_image'}),flush=True)


if __name__=='__main__':main()
