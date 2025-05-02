import torch
from torchvision.ops import roi_align,nms
import numpy as np
from argparse import ArgumentParser
import mmcv
from tqdm import tqdm
import os


class reid_inferencer():
    def __init__(self, reid):
        self.reid = reid
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.device = self.reid.device

    def mgn(self, crops):
        features = self.reid.backbone(crops)  # (bs, 2048, 16, 8)
        b1_feat = self.reid.b1(features)
        b2_feat = self.reid.b2(features)
        b21_feat, b22_feat = torch.chunk(b2_feat, 2, dim=2)
        b3_feat = self.reid.b3(features)
        b31_feat, b32_feat, b33_feat = torch.chunk(b3_feat, 3, dim=2)

        b1_pool_feat = self.reid.b1_head(b1_feat)
        b2_pool_feat = self.reid.b2_head(b2_feat)
        b21_pool_feat = self.reid.b21_head(b21_feat)
        b22_pool_feat = self.reid.b22_head(b22_feat)
        b3_pool_feat = self.reid.b3_head(b3_feat)
        b31_pool_feat = self.reid.b31_head(b31_feat)
        b32_pool_feat = self.reid.b32_head(b32_feat)
        b33_pool_feat = self.reid.b33_head(b33_feat)

        pred_feat = torch.cat([b1_pool_feat, b2_pool_feat, b3_pool_feat, b21_pool_feat,
                               b22_pool_feat, b31_pool_feat, b32_pool_feat, b33_pool_feat], dim=1)
        return pred_feat

    def process_frame_simplified(self, frame, bboxes):
        frame = torch.from_numpy(frame[:, :, ::-1].copy()).permute(2, 0, 1)

        frame = frame / 255.0
        # print(frame.shape)

        frame.sub_(self.mean).div_(self.std)
        frame = frame.unsqueeze(0)
        cbboxes = bboxes.copy()
        # cbboxes[:,[1,3]]+=540
        # cbboxes[:,[0,2]]+=960
        cbboxes = cbboxes.astype(np.float32)
        # print(cbboxes)
        # print(frame.dtype)
        # print(torch.cat([torch.zeros(len(cbboxes),1),torch.from_numpy(cbboxes)],1).dtype)
        newcrops = roi_align(frame, torch.cat([torch.zeros(len(cbboxes), 1), torch.from_numpy(cbboxes)], 1), (384, 128)).to(
            self.device)
        newfeats = (self.mgn(newcrops) + self.mgn(newcrops.flip(3))).detach().cpu().numpy() / 2

        return newfeats

def main():
    parser = ArgumentParser()
    parser.add_argument("--cam_id", type=int, default=0)
    parser.add_argument("--det_root", type=str, default="") # expecting .txt file
    parser.add_argument("--vid_root", type=str, default="") # expecting .mp4 file
    parser.add_argument("--save_root", type=str, default="")  # expecting .npy file
    parser.add_argument("--ckpt_path", type=str, default="")
    args = parser.parse_args()

    cam_id = args.cam_id
    det_path = os.path.join(args.det_root, f"cam{cam_id}_dets.txt")
    vid_path = os.path.join(args.vid_root, f"output{cam_id}.mp4")
    save_path = os.path.join(args.save_root, f"cam{cam_id}_reid.npy")
    ckpt_path = args.ckpt_path

    reid=torch.load(ckpt_path ,map_location='cpu').cuda().eval()
    reid_model = reid_inferencer(reid)


    det_annot = np.ascontiguousarray(np.loadtxt(det_path, delimiter=","))
    if len(det_annot) == 0:
        all_results = np.array([])
        np.save(save_path, all_results)
        return
    video = mmcv.VideoReader(vid_path)
    all_results = []
    for frame_id, frame in enumerate(tqdm(video)):
        dets = det_annot[det_annot[:, 0] == frame_id]
        bboxes_s = dets[:, 2:7]  # x1y1x2y2s

        screen_width = 1920
        screen_height = 1080


        x1 = bboxes_s[:, 0]
        y1 = bboxes_s[:, 1]
        x2 = bboxes_s[:, 2]
        y2 = bboxes_s[:, 3]

        x1 = np.maximum(0, x1)
        y1 = np.maximum(0, y1)
        x2 = np.minimum(screen_width, x2)
        y2 = np.minimum(screen_height, y2)

        bboxes_s[:, 0] = x1
        bboxes_s[:, 1] = y1
        bboxes_s[:, 2] = x2
        bboxes_s[:, 3] = y2

        if len(bboxes_s) == 0:
            return
        with torch.no_grad():
            feat_sim = reid_model.process_frame_simplified(frame, bboxes_s[:, :-1])

        all_results.append(feat_sim)

    all_results = np.concatenate(all_results)
    np.save(save_path, all_results)
    print(f"The shape of the result: {all_results.shape}")

if __name__ == "__main__":
    main()