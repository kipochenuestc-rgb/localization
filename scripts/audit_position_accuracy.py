"""Compare final SLAM reference/antenna poses to raw GNSS with disjoint alignment/evaluation times."""
import argparse,collections,json,pathlib
import numpy as np
from rgb_core import Poses,load_rig
from gnss_core import LocalENU


def fit_xy(source,target):
    a,b=source[:,:2].mean(0),target[:,:2].mean(0);u,_,vt=np.linalg.svd((source[:,:2]-a).T@(target[:,:2]-b));r=u@vt
    if np.linalg.det(r)<0:u[:,-1]*=-1;r=u@vt
    R=np.eye(3);R[:2,:2]=r;t=np.r_[b-a@r,np.median(target[:,2]-source[:,2])];return R,t


def metrics(error):
    if not len(error):return {'count':0}
    horizontal=np.linalg.norm(error[:,:2],axis=1);return {'count':len(error),'horizontal_rmse_m':float(np.sqrt(np.mean(horizontal**2))),'horizontal_median_m':float(np.median(horizontal)),'horizontal_p95_m':float(np.quantile(horizontal,.95)),'horizontal_max_m':float(horizontal.max()),'vertical_rmse_m':float(np.sqrt(np.mean(error[:,2]**2))),'vertical_bias_m':float(error[:,2].mean())}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--audit',type=pathlib.Path,required=True);ap.add_argument('--session',type=pathlib.Path,required=True);ap.add_argument('--gnss-run',type=pathlib.Path,required=True);ap.add_argument('--no-gnss-run',type=pathlib.Path,required=True);args=ap.parse_args();out=args.audit
    raw=np.loadtxt(out/'raw_gnss.csv',delimiter=',',skiprows=1,ndmin=2);raw=raw[np.isfinite(raw[:,[0,2,3,4]]).all(1)&(raw[:,5]>=0)];local=LocalENU(raw[0,3],raw[0,2],raw[0,4]);enu=np.array([local.convert(r[3],r[2],r[4]) for r in raw]);t0=json.loads((out/'raw_audit.json').read_text())['bag_start_s']
    cameras,_,_,_=load_rig(args.session/'calibration');antenna=json.loads((args.session/'calibration/config/antenna_mount.json').read_text());T=cameras[0]['reference_from_camera'];lever=T[:3,:3]@np.array(antenna['antenna_origin_in_camera0_m'])+T[:3,3]
    quality=collections.Counter();gga_times=[]
    for row in json.loads((out/'raw_gga.json').read_text()):
        text=row['text'].strip();fields=text.split(',')
        if len(fields)>6:
            quality[fields[6]]+=1
            if fields[6].isdigit():gga_times.append([row['receipt_s'],int(fields[6])])
    spacing=np.diff(raw[:,0]);report={'reference_wgs84':local.reference,'raw_valid_gnss_count':len(raw),'gnss_gap_p50_p95_max_s':np.quantile(spacing,[.5,.95,1]).tolist(),'gnss_gaps_gt2s':int((spacing>2).sum()),'gga_quality_counts':dict(quality),'antenna_lever_in_slam_reference_m':lever.tolist(),'antenna_lever_length_m':float(np.linalg.norm(lever)),'runs':{}}
    traces={};gnss_quality=np.full(len(raw),-1)
    if gga_times:
        g=np.array(gga_times);k=np.searchsorted(g[:,0],raw[:,1]);a=np.clip(k-1,0,len(g)-1);b=np.clip(k,0,len(g)-1);k=np.where(abs(g[a,0]-raw[:,1])<abs(g[b,0]-raw[:,1]),a,b);good=abs(g[k,0]-raw[:,1])<.5;gnss_quality[good]=g[k[good],1]
    for label,path in [('gnss_constrained',args.gnss_run),('no_gnss',args.no_gnss_run)]:
        poses=Poses(path/'map/transformations.pcd');valid=(raw[:,0]>=poses.start)&(raw[:,0]<=poses.end);times=raw[valid,0];truth=enu[valid];M=poses.at(times);reference=M[:,:3,3];predicted=reference+np.einsum('nij,j->ni',M[:,:3,:3],lever);relative=times-t0;alignment=relative<=120;test=relative>120
        if alignment.sum()<12:raise ValueError('Insufficient early alignment GNSS')
        R,translation=fit_xy(predicted[alignment],truth[alignment]);aligned=predicted@R+translation;error=aligned-truth
        R0,tr0=fit_xy(reference[alignment],truth[alignment]);error0=reference@R0+tr0-truth
        # Full-trajectory fit is diagnostic only: it leaks evaluation observations.
        fullR,fullT=fit_xy(predicted,truth);full_error=predicted@fullR+fullT-truth
        result={'alignment_interval_bag_seconds':[float(relative[alignment].min()),float(relative[alignment].max())],'alignment_count':int(alignment.sum()),'alignment_horizontal_span_m':float(np.linalg.norm(np.ptp(truth[alignment,:2],axis=0))),'alignment_residual':metrics(error[alignment]),'evaluation_after120s':metrics(error[test]),'evaluation_without_lever_correction':metrics(error0[test]),'full_trajectory_best_fit_not_independent':metrics(full_error),'segments':{name:metrics(error[(relative>=a)&(relative<b)]) for name,a,b in [('segment_275_295s',275,295),('uturn_379_399s',379,399),('after_uturn_400_600s',400,600),('tail_600s_plus',600,1e6)]},'rtk_fixed_quality4_evaluation':metrics(error[test&(gnss_quality[valid]==4)])}
        if label=='gnss_constrained':
            config=json.loads((path/'gnss_alignment.json').read_text());onlineR=np.array(config['enu_to_map_rotation']);onlineT=np.array(config['enu_to_map_translation']);online_truth=np.array([LocalENU(*config['reference_wgs84']).convert(r[3],r[2],r[4]) for r in raw[valid]])@onlineR.T+onlineT
            result['frozen_online_alignment_reference_residual_not_independent']=metrics(reference-online_truth);result['frozen_online_alignment_antenna_residual_not_independent']=metrics(predicted-online_truth)
        report['runs'][label]=result;traces[label]=(relative,aligned,error,truth)
        np.savetxt(out/f'{label}_accuracy.csv',np.column_stack([relative,aligned,truth,error,gnss_quality[valid]]),delimiter=',',header='bag_time_s,predE,predN,predU,gnssE,gnssN,gnssU,errorE,errorN,errorU,gga_quality',comments='')
    report['limitations']=['GNSS-constrained SLAM uses the same GNSS source; its residuals are not independent accuracy.','No-GNSS run excludes GNSS factors, but the first 120 seconds are used for yaw/translation registration; later comparisons measure agreement with GNSS, not survey truth.','GNSS receiver errors, geoid/altitude convention and installation uncertainty remain; no surveyed truth supplied.','Lever-arm audit does not retroactively undo missing lever compensation in existing GNSS graph factors.','Image feature reprojection error cannot be converted directly to absolute position accuracy.'];(out/'position_accuracy_audit.json').write_text(json.dumps(report,indent=2))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for label,(relative,p,error,truth) in traces.items():
        axes[0].plot(p[:,0],p[:,1],label=label);axes[1].plot(relative,np.linalg.norm(error[:,:2],axis=1),label=label);axes[2].plot(relative,error[:,2],label=label)
    axes[0].plot(truth[:,0],truth[:,1],'k--',label='raw GNSS');axes[0].axis('equal');axes[0].set(xlabel='East (m)',ylabel='North (m)',title='First 120 s registration, no scale fitting');axes[1].set(xlabel='Bag time (s)',ylabel='Horizontal difference (m)');axes[2].set(xlabel='Bag time (s)',ylabel='Height difference (m)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'position_accuracy.png',dpi=150);print('READY',json.dumps(report),flush=True)


if __name__=='__main__':main()
