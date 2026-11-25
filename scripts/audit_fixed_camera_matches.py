"""Audit original camera epipolar consistency using unfiltered static SIFT matches.

An image-estimated fundamental matrix only rejects correspondence outliers. It
never changes the supplied camera transforms, calibration, geometry or training.
This is image consistency, not independent absolute positioning accuracy.
"""
import argparse,hashlib,json
from functools import lru_cache
from pathlib import Path
import cv2,numpy as np


def sampson(a,b,F):
    x=np.c_[a,np.ones(len(a))];y=np.c_[b,np.ones(len(b))]
    left=x@F.T;right=y@F
    return abs((left*y).sum(1))/np.sqrt((left[:,:2]**2).sum(1)+(right[:,:2]**2).sum(1)).clip(1e-12)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['scene','assets','output']:ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();views=json.loads((args.scene/'cameras.json').read_text());cv2.setNumThreads(2);cv2.setRNGSeed(42)
    args.output.mkdir(parents=True,exist_ok=False)
    transforms=np.asarray([v['camera_to_world'] for v in views]);centers=transforms[:,:3,3];directions=transforms[:,:3,2]
    times=np.asarray([v['stamp_s'] for v in views]);front=[i for i,v in enumerate(views) if v['camera_id']==1]
    core=min(front,key=lambda i:np.linalg.norm(centers[i,:2]));groups={views[core]['group_index']}
    for side in [-1,1]:
        candidates=[i for i in front if 0<side*(times[i]-times[core])<15]
        if candidates:groups.add(views[min(candidates,key=lambda i:abs(np.linalg.norm(centers[i]-centers[core])-5))]['group_index'])
    pairs={}
    for i,v in enumerate(views):
        if v['group_index'] not in groups:continue
        distance=np.linalg.norm(centers-centers[i],axis=1)
        compatible=[j for j,w in enumerate(views) if w['camera_id']==v['camera_id'] and .2<distance[j]<4
                    and directions[j]@directions[i]>.98]
        for sign in [-1,1]:
            options=[j for j in compatible if 0<sign*(times[j]-times[i])<10]
            for baseline in [.5,1.5,3.]:
                if options:
                    j=min(options,key=lambda j:abs(distance[j]-baseline));pairs[tuple(sorted([i,j]))]='same_pass'
        revisit=[j for j in compatible if abs(times[j]-times[i])>30]
        if revisit:
            j=min(revisit,key=lambda j:abs(distance[j]-1.5));pairs[tuple(sorted([i,j]))]='revisit'
    @lru_cache(maxsize=48)
    def features(i):
        path=args.assets/'features'/(Path(views[i]['image_path']).stem+'.npz')
        with np.load(path) as data:xy=data['xy'];desc=data['descriptors'].astype('f4')
        static=cv2.imread(str(args.scene/'valid_masks'/f'{i:04d}.png'),0)
        if static is None:raise ValueError('Missing static mask')
        sky=cv2.imread(str(args.scene/'sky_masks'/f'{i:04d}.png'),0)
        if sky is not None:static=static*(sky==0)
        uv=np.rint(xy).astype(int);inside=(uv[:,0]>=0)&(uv[:,0]<static.shape[1])&(uv[:,1]>=0)&(uv[:,1]<static.shape[0])
        ids=np.flatnonzero(inside);ids=ids[static[uv[ids,1],uv[ids,0]]>0]
        return xy[ids],desc[ids]
    matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=64));rows=[]
    for (i,j),kind in sorted(pairs.items()):
        a,da=features(i);b,db=features(j)
        if min(len(da),len(db))<20:continue
        forward=matcher.knnMatch(da,db,k=2);reverse=matcher.knnMatch(db,da,k=2)
        reverse_good={pair[0].queryIdx:pair[0].trainIdx for pair in reverse if len(pair)==2 and pair[0].distance<.7*pair[1].distance}
        matches=[pair[0] for pair in forward if len(pair)==2 and pair[0].distance<.7*pair[1].distance
                 and reverse_good.get(pair[0].trainIdx)==pair[0].queryIdx]
        if len(matches)<20:continue
        pa=np.asarray([a[m.queryIdx] for m in matches],dtype='f8');pb=np.asarray([b[m.trainIdx] for m in matches],dtype='f8')
        estimated,inlier=cv2.findFundamentalMat(pa,pb,cv2.USAC_MAGSAC,1.,.999)
        if estimated is None or estimated.shape!=(3,3) or int(inlier.sum())<20:continue
        pa=pa[inlier[:,0]>0];pb=pb[inlier[:,0]>0]
        relative=np.linalg.inv(transforms[j])@transforms[i];x,y,z=relative[:3,3]
        cross=np.asarray([[0,-z,y],[z,0,-x],[-y,x,0]])
        fixed=np.linalg.inv(views[j]['K']).T@cross@relative[:3,:3]@np.linalg.inv(views[i]['K'])
        error=sampson(pa,pb,fixed);control=sampson(pa,pb,estimated)
        row=dict(source=i,target=j,camera=views[i]['camera_id'],kind=kind,
                 source_group=views[i]['group_index'],target_group=views[j]['group_index'],
                 baseline_m=float(np.linalg.norm(centers[j]-centers[i])),time_difference_s=float(times[j]-times[i]),
                 mutual_ratio_matches=len(matches),image_ransac_inliers=len(pa),
                 fixed_sqrt_sampson_p50_p90_p95_px=np.percentile(error,[50,90,95]).tolist(),
                 image_estimated_sqrt_sampson_p50_p90_p95_px=np.percentile(control,[50,90,95]).tolist(),
                 fixed_fraction_above1px=float((error>1).mean()),fixed_fraction_above2px=float((error>2).mean()))
        rows.append(row)
        np.savez_compressed(args.output/f'{i:04d}_{j:04d}.npz',source=pa,target=pb,fixed_error=error,image_estimated_control_error=control)
        print(json.dumps(row),flush=True)
    summary={}
    for kind in ['same_pass','revisit']:
        for camera in range(4):
            sample=[r for r in rows if r['kind']==kind and r['camera']==camera]
            if sample:summary[f'{kind}_camera{camera}']=dict(pairs=len(sample),
                median_pair_median_fixed_px=float(np.median([r['fixed_sqrt_sampson_p50_p90_p95_px'][0] for r in sample])),
                median_pair_p90_fixed_px=float(np.median([r['fixed_sqrt_sampson_p50_p90_p95_px'][1] for r in sample])),
                median_pair_median_image_control_px=float(np.median([r['image_estimated_sqrt_sampson_p50_p90_p95_px'][0] for r in sample])))
    result=dict(scene=str(args.scene),cameras_sha256=hashlib.sha256((args.scene/'cameras.json').read_bytes()).hexdigest(),
        camera_optimization=False,lidar_used=False,summary=summary,pairs=rows,
        notes=['Mutual0.7 SIFT ratio matches; image-only MAGSAC1px inliers selected without fixed-camera epipolar filtering.',
               'Image-estimated F is a diagnostic control, not a replacement or optimized camera.',
               'Residuals include timing, calibration, localization, rolling shutter, matching and scene motion errors.',
               'Fitting/inlier selection makes the image-estimated control optimistic. Not independent ground truth.',
               'Fixed sqrt Sampson residual differs from one-direction point-to-epipolar-line distance.'])
    (args.output/'report.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
