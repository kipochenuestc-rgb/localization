"""Audit NDT/EKF consistency and export timestamped camera poses, retaining missing/weak flags."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from rgb_core import Poses, load_rig
from gnss_core import LocalENU


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--image-times',type=Path)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.input/'manifest.json').read_text());t0=manifest['bag_start_s']
    e=np.genfromtxt(args.run/'ekf_trajectory.csv',delimiter=',',names=True)
    # At shutdown the EKF can publish twice at a stamp. Keep the last corrected value.
    _,rev=np.unique(e['stamp_s'][::-1],return_index=True);e=e[np.sort(len(e)-1-rev)]
    et=e['stamp_s'];ep=np.column_stack([e[k] for k in ['x','y','z']]);eq=Rotation.from_quat(np.column_stack([e[k] for k in ['qx','qy','qz','qw']]))
    if not np.isfinite(ep).all() or np.any(np.diff(et)<=0):raise ValueError('Invalid EKF trajectory')
    def sample(t):
        m=np.tile(np.eye(4),(len(t),1,1));m[:,:3,:3]=Slerp(et-et[0],eq)(t-et[0]).as_matrix()
        for k in range(3):m[:,k,3]=np.interp(t,et,ep[:,k])
        return m
    quality=[json.loads(l) for l in (args.run/'ndt_quality.jsonl').read_text().splitlines()]
    good=[s for s in quality if s['accepted'] and 'pose' in s]
    if not good:raise RuntimeError('No accepted NDT poses; camera export cannot be validated')
    nt=np.array([s['stamp_s'] for s in good]);np3=np.array([s['pose'][:3] for s in good])
    nvalid=(nt>=et[0])&(nt<=et[-1]);pred=sample(nt[nvalid]);ndt_difference=np.linalg.norm(pred[:,:3,3]-np3[nvalid],axis=1)
    cameras,_,_,cal_status=load_rig(manifest['calibration_dir'])
    stamps=np.load(args.image_times)['/Data_Cam_Global'][:,0] if args.image_times else np.array(json.loads((args.input/'images.json').read_text()))
    covered=(stamps>=et[0])&(stamps<=et[-1]);base=np.full((len(stamps),4,4),np.nan);base[covered]=sample(stamps[covered])
    cam=np.full((len(stamps),4,4,4),np.nan)
    for cid in range(4):cam[covered,cid]=base[covered]@cameras[cid]['reference_from_camera']
    nearest=np.searchsorted(nt,stamps).clip(0,len(nt)-1);left=np.maximum(nearest-1,0)
    nearest=np.where(abs(nt[left]-stamps)<abs(nt[nearest]-stamps),left,nearest)
    age=abs(nt[nearest]-stamps);review=(~covered)|(age>.3)
    speed=np.r_[0,np.linalg.norm(np.diff(np3,axis=0),axis=1)/np.diff(nt)]
    nq=Rotation.from_quat(np.array([s['pose'][3:] for s in good]))
    angular_rate=np.r_[0,np.degrees((nq[:-1].inv()*nq[1:]).magnitude())/np.diff(nt)]
    motion_outlier=(speed>5.)|(angular_rate>90.)
    image_motion_outlier=motion_outlier[nearest]&(age<.3)
    review |= image_motion_outlier
    ekf_bracket=np.full(len(stamps),np.nan)
    ki=np.searchsorted(et,stamps[covered],side='right').clip(1,len(et)-1)
    ekf_bracket[covered]=et[ki]-et[ki-1]
    review |= ekf_bracket>.1
    var=np.full(len(stamps),np.nan)
    var[covered]=sum(np.interp(stamps[covered],et,e[k]) for k in ['var_x','var_y','var_z'])
    review |= var>.5
    with (args.output/'all_camera_poses.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['group_index','camera_id','stamp_s','bag_time_s','x','y','z','qx','qy','qz','qw','pose_available','source','nearest_accepted_ndt_s','review_flags'])
        for i,t in enumerate(stamps):
            flags=[]
            if not covered[i]:flags.append('outside_ekf_time_support')
            if age[i]>.3:flags.append('no_accepted_ndt_within_0.3s')
            if var[i]>.5:flags.append('high_ekf_position_covariance')
            if image_motion_outlier[i]:flags.append('near_ndt_motion_outlier')
            if ekf_bracket[i]>.1:flags.append('ekf_bracket_over_0.1s')
            for cid in range(4):
                values=[*cam[i,cid,:3,3],*Rotation.from_matrix(cam[i,cid,:3,:3]).as_quat()] if covered[i] else ['']*7
                writer.writerow([i,cid,f'{t:.9f}',t-t0,*values,int(covered[i]),'ndt_gnss_ekf_interpolated' if covered[i] else 'missing',age[i],'|'.join(flags)])
    alignment=json.loads((Path(manifest['map_run'])/'gnss_alignment.json').read_text());local=LocalENU(*alignment['reference_wgs84']);R=np.array(alignment['enu_to_map_rotation']);tr=np.array(alignment['enu_to_map_translation'])
    fixes=np.array(json.loads((args.input/'gnss.json').read_text()));gxyz=np.array([R@local.convert(*g[1:4])+tr for g in fixes]);gt=fixes[:,0];gvalid=(gt>=et[0])&(gt<=et[-1]);gm=sample(gt[gvalid]);mount=json.loads((Path(manifest['calibration_dir'])/'config/antenna_mount.json').read_text());T=cameras[0]['reference_from_camera'];lever=T[:3,:3]@np.array(mount['antenna_origin_in_camera0_m'])+T[:3,3];residual=gm[:,:3,3]+np.einsum('nij,j->ni',gm[:,:3,:3],lever)-gxyz[gvalid]
    np.savez_compressed(args.output/'all_camera_poses.npz',stamps=stamps,bag_start_s=t0,base=base,cameras=cam,covered=covered,review=review,gnss_stamps=gt,gnss_xyz=gxyz)
    stats=lambda a:np.quantile(a,[.5,.95,1]).tolist() if len(a) else None
    reference=Poses(Path(manifest['map_run'])/'map/transformations.pcd');overlap=(nt>=reference.start)&(nt<=reference.end)
    graph_difference=np.linalg.norm(np3[overlap]-reference.at(nt[overlap])[:,:3,3],axis=1)
    report={'pose_source':'ndt_gnss_ekf','image_groups':len(stamps),'camera_images':len(stamps)*4,'groups_with_pose':int(covered.sum()),'groups_missing_pose':int((~covered).sum()),'groups_with_review_flags':int(review.sum()),'ndt_scans':len(quality),'ndt_accepted':len(good),'ndt_rejected':len(quality)-len(good),'ekf_samples':len(e),'ekf_support_bag_seconds':[float(et[0]-t0),float(et[-1]-t0)],'ndt_fitness_p50_p95_max_m2':stats([s['fitness_m2'] for s in quality]),'ndt_inlier_min_p50':np.quantile([s['inlier_fraction'] for s in quality],[0,.5]).tolist(),'ndt_speed_p50_p95_max_m_s':stats(speed),'ndt_angular_rate_p50_p95_max_deg_s':stats(angular_rate),'ndt_motion_outlier_count':int(motion_outlier.sum()),'ekf_ndt_difference_p50_p95_max_m':stats(ndt_difference),'map_slam_ndt_difference_p50_p95_max_m':stats(graph_difference),'gnss_same_source_horizontal_rmse_m':float(np.sqrt(np.mean(np.sum(residual[:,:2]**2,axis=1)))),'calibration_status':cal_status,'limitations':manifest['limitations']+['Map and GNSS are not independent ground truth.','Per-image poses are interpolated EKF estimates, not per-image visual localization.','The first SLAM pose initializes localization; no later SLAM poses used as observations.']}
    output_gaps=np.flatnonzero(np.diff(et)>.1)
    report.update({'ekf_output_interval_p50_p95_max_s':stats(np.diff(et)),
                   'image_groups_ekf_bracket_over_0.1s':int((ekf_bracket>.1).sum()),
                   'ekf_output_gaps_over_0.1s':[{'start_bag_s':float(et[k]-t0),'end_bag_s':float(et[k+1]-t0)} for k in output_gaps]})
    (args.output/'report.json').write_text(json.dumps(report,indent=2))
    np.savetxt(args.output/'gnss_residuals.csv',np.column_stack([gt[gvalid],residual]),delimiter=',',header='stamp_s,dx,dy,dz',comments='')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,3,figsize=(15,4),layout='constrained');ax[0].plot(ep[:,0],ep[:,1],label='EKF');ax[0].plot(gxyz[:,0],gxyz[:,1],'.',ms=2,label='GNSS antenna');ax[0].axis('equal');ax[0].legend();ax[1].plot(nt-t0,[s['fitness_m2'] for s in good]);ax[1].set(xlabel='Bag time (s)',ylabel='NDT nearest-point fitness (m2)');ax[2].plot(nt[nvalid]-t0,ndt_difference);ax[2].set(xlabel='Bag time (s)',ylabel='EKF to NDT position difference (m)');fig.savefig(args.output/'localization_checks.png',dpi=150)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
