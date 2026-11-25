"""Finish prepared full-route localization, validate/export camera poses and display results."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from localization_paths import SLAM_ROOT

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--run',default='ndt_ekf_full_v2')
    ap.add_argument('--preparation-pid',type=int)
    args=ap.parse_args()
    if Path(args.run).name!=args.run:raise ValueError('run must be a folder name')
    result=ROOT/'runs'/args.run;export=ROOT/'runs'/f'{args.run}_cameras'
    if result.exists() or export.exists():raise FileExistsError('Choose a new run name')
    previous=None
    while not (args.input/'manifest.json').exists():
        status=(args.input/'status.json').read_text() if (args.input/'status.json').exists() else 'preparing'
        if status!=previous:print('PREPARATION',status,flush=True);previous=status
        if args.preparation_pid:
            try:os.kill(args.preparation_pid,0)
            except ProcessLookupError:raise RuntimeError('Preparation stopped before producing a manifest')
        time.sleep(2)
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:command=(process/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if command and Path(os.fsdecode(command[0])).name.startswith('python') and any(v.endswith(b'show_camera_poses.py') for v in command) and any(b'all_image_pose_audit' in v for v in command):
            try:os.kill(int(process.name),signal.SIGINT)
            except ProcessLookupError:pass
    rviz_log=(ROOT/'runs'/f'{args.run}_rviz.log').open('w')
    rviz=subprocess.Popen(['python3','-u',str(ROOT/'scripts/launch_rviz.py'),'--repair','--config',str(ROOT/'config/ndt_camera_localization.rviz')],stdout=rviz_log,stderr=subprocess.STDOUT)
    with (ROOT/'runs'/f'{args.run}_controller.log').open('w') as log:
        subprocess.run(['python3','-u',str(ROOT/'scripts/run_ndt_localization.py'),'--input',str(args.input),'--run',args.run,'--no-rviz','--images','--exit-after-replay','--pace','0'],stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['python3',str(ROOT/'scripts/export_localized_camera_poses.py'),'--run',str(result),'--input',str(args.input),'--output',str(export),'--image-times',str(SLAM_ROOT/'runs/parameter_accuracy_audit/raw_timestamps.npz')],check=True)
    manifest=json.loads((args.input/'manifest.json').read_text())
    report=json.loads((export/'report.json').read_text())
    print('FULL_LOCALIZATION_EXPORTED',json.dumps(report),flush=True)
    # Preserve original timestamps in the saved playback rather than extrapolating missing frames.
    subprocess.Popen(['python3','-u',str(ROOT/'scripts/launch_rviz.py'),'--config',str(ROOT/'config/camera_pose_audit.rviz')],stdout=rviz_log,stderr=subprocess.STDOUT)
    subprocess.run(['python3','-u',str(ROOT/'scripts/show_camera_poses.py'),'--audit',str(export),'--map',manifest['map_file'],'--bag',manifest['bag'],'--rate','2'],check=True)


if __name__=='__main__':main()
