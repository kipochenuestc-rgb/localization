"""Check observed static-image epipolar geometry against camera/SLAM transform hypotheses."""
import argparse,itertools,json,pathlib
import cv2,numpy as np
from rgb_core import load_rig,Poses


def residual(a,b,K,T):
    x,y,z=T[:3,3];cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]]);F=np.linalg.inv(K).T@cross@T[:3,:3]@np.linalg.inv(K);p=np.column_stack([a,np.ones(len(a))]);q=np.column_stack([b,np.ones(len(b))]);l=p@F.T;r=q@F;return abs((q*l).sum(1))/np.sqrt(np.maximum((l[:,:2]**2).sum(1)+(r[:,:2]**2).sum(1),1e-15))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--scene',type=pathlib.Path,required=True);ap.add_argument('--calibration',type=pathlib.Path,required=True);ap.add_argument('--run',type=pathlib.Path,required=True);ap.add_argument('--masks',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    views=json.loads((args.scene/'cameras.json').read_text());cameras,_,_,_=load_rig(args.calibration);poses=Poses(args.run/'map/transformations.pcd');sift=cv2.SIFT_create(nfeatures=4000);features={};bycam={cid:[] for cid in cameras};byset={}
    for i,v in enumerate(views):bycam[v['camera_id']].append(i);byset.setdefault(int(pathlib.Path(v['image_path']).stem.split('_')[0]),{})[v['camera_id']]=i
    pairs=[]
    for indices in bycam.values():
        for k in range(0,len(indices)-3,3):pairs.append((indices[k],indices[k+2],'temporal'))
    for group in list(byset.values())[::4]:
        for a,b in [(0,1),(1,2),(2,3),(3,0)]:pairs.append((group[a],group[b],'cross_camera'))
    def feature(i):
        if i not in features:
            image=cv2.imread(str(args.scene/views[i]['image_path']),0);mask=cv2.imread(str(args.masks/f'{i:04d}.png'),0);features[i]=sift.detectAndCompute(image,mask)
        return features[i]
    records=[];bf=cv2.BFMatcher()
    for i,j,kind in pairs:
        ka,da=feature(i);kb,db=feature(j)
        if da is None or db is None:continue
        match=bf.knnMatch(da,db,k=2);good=[m for m,n in match if m.distance<.7*n.distance]
        if len(good)<15:continue
        a=np.float64([ka[m.queryIdx].pt for m in good]);b=np.float64([kb[m.trainIdx].pt for m in good]);F,inlier=cv2.findFundamentalMat(a,b,cv2.USAC_MAGSAC,1.,.999)
        if F is None or inlier.sum()<12:continue
        a,b=a[inlier.ravel()>0],b[inlier.ravel()>0];records.append((i,j,kind,a,b))
    report={'matching_pairs':len(records),'temporal_pairs':sum(r[2]=='temporal' for r in records),'cross_camera_pairs':sum(r[2]=='cross_camera' for r in records),'hypotheses':[]}
    K=np.array(views[0]['K']);best=[]
    def score(mapping,shift=0,inverse=False,scale=1):
        errors={cid:[] for cid in cameras};cross=[];intr=K.copy();intr[:2,:]*=scale
        for i,j,kind,a,b in records:
            va,vb=views[i],views[j];A=cameras[mapping[va['camera_id']]]['reference_from_camera'];B=cameras[mapping[vb['camera_id']]]['reference_from_camera']
            if inverse:A=np.linalg.inv(A);B=np.linalg.inv(B)
            Ta=poses.at(va['stamp_s']+shift)@A;Tb=poses.at(vb['stamp_s']+shift)@B;e=residual(a,b,intr,np.linalg.inv(Tb)@Ta)
            if kind=='temporal':errors[va['camera_id']].extend(e)
            else:cross.extend(e)
        all_errors=np.concatenate([np.array(v) for v in errors.values()]);return {'temporal_median_sampson_px':float(np.median(all_errors)),'temporal_p90_px':float(np.quantile(all_errors,.9)),'per_camera_median_px':{str(k):float(np.median(v)) for k,v in errors.items()},'cross_camera_median_px':float(np.median(cross)) if cross else None,'temporal_matches':len(all_errors),'cross_camera_matches':len(cross)}
    for permutation in itertools.permutations(range(4)):
        result={'camera_assignment':list(permutation),**score(permutation)};best.append(result)
    report['assignment_ranking']=sorted(best,key=lambda x:x['temporal_median_sampson_px']);report['current']=score(range(4));report['inverse_extrinsics_hypothesis']=score(range(4),inverse=True);report['wrong_intrinsics_scale_2_4_hypothesis']=score(range(4),scale=2.4)
    report['time_shift_profile']=[{'camera_pose_time_shift_s':float(dt),**score(range(4),shift=float(dt))} for dt in np.linspace(-.2,.2,17)]
    t=np.array([v['stamp_s'] for v in views])[::4];gaps=[]
    for value in t:
        k=np.searchsorted(poses.times,value);gaps.append(float(poses.times[k]-poses.times[k-1]))
    report['pose_interpolation_bracket_s_p50_max']=np.quantile(gaps,[.5,1]).tolist();report['notes']=['Sampson errors on image-estimated fundamental-matrix inliers; static masks; errors are consistency diagnostics, not independent calibration truth.','Time-shift minimum can conflate keyframe interpolation, image exposure, rolling shutter and calibration errors.','All tests use the local 275-295 s scene; not a full-route calibration.']
    (args.output/'camera_geometry_audit.json').write_text(json.dumps(report,indent=2));print('READY',json.dumps({k:v for k,v in report.items() if k not in ['assignment_ranking','time_shift_profile']}),flush=True)


if __name__=='__main__':main()
