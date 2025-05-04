from argparse import ArgumentParser
import os.path as osp
import numpy as np
from tqdm import tqdm

from util.camera import Camera
from Tracker.PoseTracker import Detection_Sample, PoseTracker,TrackState

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'track')))


def main():
    parser = ArgumentParser()
    parser.add_argument("--result_root", type=str, default="")
    parser.add_argument("--calibrate_root", type=str, default="")
    parser.add_argument("--save_root", type=str, default="")
    args = parser.parse_args()

    result_dir = args.result_root
    cal_dir = args.calibrate_root
    save_path = osp.join(args.save_root, "track_results.txt")
    det_data = []
    det_data.append(np.loadtxt(osp.join(result_dir,'cam1_dets.txt'), delimiter=","))
    det_data.append(np.loadtxt(osp.join(result_dir,'cam2_dets.txt'), delimiter=","))

    pose_data=[]
    pose_data.append(np.loadtxt(osp.join(result_dir,'cam1_poses.txt')))
    pose_data.append(np.loadtxt(osp.join(result_dir,'cam2_poses.txt')))


    reid_data = []
    reid_data_scene = np.load(osp.join(result_dir, 'cam1_reid.npy'), mmap_mode='r')
    if len(reid_data_scene):
        reid_data_scene = reid_data_scene / np.linalg.norm(reid_data_scene, axis=1, keepdims=True)
    reid_data.append(reid_data_scene)

    reid_data_scene = np.load(osp.join(result_dir, 'cam2_reid.npy'), mmap_mode='r')
    if len(reid_data_scene):
        reid_data_scene = reid_data_scene / np.linalg.norm(reid_data_scene, axis=1, keepdims=True)
    reid_data.append(reid_data_scene)

    cals = []
    cals.append(Camera(osp.join(cal_dir, 'cam1-537', "calibration.json")))
    cals.append(Camera(osp.join(cal_dir, 'cam2-543', "calibration.json")))


    # co detection dai nhat
    max_frame = []
    for det_sv in det_data:
        if len(det_sv):
            max_frame.append(np.max(det_sv[:,0]))
    max_frame = int(np.max(max_frame))

    tracker = PoseTracker(cals)
    box_thred = 0.3
    results = []

    for frame_id in tqdm(range(max_frame + 1)):
        detection_sample_mv = []
        for v in range(tracker.num_cam):
            detection_sample_sv = []
            det_sv = det_data[v] #lay detection cua camera v
            if len(det_sv)==0: # neu ko detection thi them vao mv va tiep tuc
                detection_sample_mv.append(detection_sample_sv)
                continue

            idx = det_sv[:,0]==frame_id # lay ra index cua frame id hien tai
            cur_det = det_sv[idx]
            cur_pose = pose_data[v][idx]
            cur_reid = reid_data[v][idx]

            for det, pose, reid in zip(cur_det, cur_pose, cur_reid): #doi voi moi detection trong do
                if det[-1]<box_thred or len(det)==0:
                    continue #loai bo confident thap
                new_sample = Detection_Sample(bbox=det[2:],keypoints_2d=pose[6:].reshape(17,3), reid_feat=reid, cam_id = v, frame_id=frame_id)
                detection_sample_sv.append(new_sample)
            detection_sample_mv.append(detection_sample_sv)

        print("frame {}".format(frame_id),"det nums: ",[len(L) for L in detection_sample_mv])
        tracker.mv_update_wo_pred(detection_sample_mv, frame_id)

        frame_results = tracker.output(frame_id)
        results += frame_results

    results = np.concatenate(results,axis=0)
    sort_idx = np.lexsort((results[:,2],results[:,0]))
    results = np.ascontiguousarray(results[sort_idx])
    np.savetxt(save_path, results)




if __name__ == "__main__":
    main()