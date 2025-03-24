# Enviroment Setup
import os, sys
WORKING_DIR = '/home/ruifanj2/workspace/Avatar/flame-head-tracker'
os.chdir(WORKING_DIR) # change the working directory to the project's absolute path
print("Current Working Directory: ", os.getcwd())

## Computing Device
device = 'cuda:0'
import torch
torch.cuda.set_device(device) # this will solve the problem that OpenGL not on the same device with torch tensors

sys.path.append(WORKING_DIR)
sys.path.append('./utils/flame_lib/')
sys.path.append('./utils/flame_fitting/')
sys.path.append('./utils/face_parsing/')
sys.path.append('./utils/decalib/')
sys.path.append('./utils/mesh_renderer')
sys.path.append('./utils/scene')

import matplotlib.pyplot as plt
import numpy as np

from tracker_base import Tracker
from utils.video_utils import video_to_images_original


def plot(ret_dict, filename):
    # plot some results
    plt.figure(figsize=(15,6))

    plt.subplot(1,6,1)
    plt.imshow(ret_dict['img']); plt.title('img')

    plt.subplot(1,6,2)
    plt.imshow(ret_dict['img_aligned']); plt.title('img_aligned')

    plt.subplot(1,6,3)
    plt.imshow(ret_dict['parsing']); plt.title('parsing')

    plt.subplot(1,6,4)
    plt.imshow(ret_dict['parsing_aligned']); plt.title('parsing_aligned')

    plt.subplot(1,6,5)
    plt.imshow(ret_dict['img_rendered']); plt.title('img_rendered')

    plt.subplot(1,6,6)
    plt.imshow(ret_dict['mesh_rendered']); plt.title('mesh_rendered')

    plt.savefig(filename)

tracker_cfg = {
    'mediapipe_face_landmarker_v2_path': './models/face_landmarker_v2_with_blendshapes.task',
    'flame_model_path': './models/FLAME2020/generic_model.pkl',
    'flame_lmk_embedding_path': './models/landmark_embedding.npy',
    'ear_landmarker_path': './models/ear_landmarker.pth', # this is optional, if you do not want to use ear landmarks during fitting, just remove this line
    'tex_space_path': './models/FLAME_albedo_from_BFM.npz',
    'face_parsing_model_path': './utils/face_parsing/79999_iter.pth',
    'template_mesh_file_path': './models/head_template.obj',
    'result_img_size': 512,
    'device': device,
}

tracker = Tracker(tracker_cfg)

# tracker.update_fov(fov=20)                 # optional setting
# tracker.set_landmark_detector('FAN')      # optional setting
tracker.set_landmark_detector('mediapipe') # optional setting
tracker.use_ear_landmarks = False

video_path = './assets/obama.mp4'
video_frames = video_to_images_original(video_path)
print("successfully read", len(video_frames), "video frames")

# if realign == True, the fitting is on the realigned image
# otherwise the fitting is on the original image
ret_dict = tracker.run_bare(video_frames[0], realign=True, photometric_fitting=False, prev_ret_dict=None)
for i in range len(video_frames):
    frame = video_frames[i]
    ret_dict = tracker.run_bare(frame, realign=True, photometric_fitting=False, prev_ret_dict=ret_dict)
    if i % 10 == 0:
        print("processed", i, "frames")

