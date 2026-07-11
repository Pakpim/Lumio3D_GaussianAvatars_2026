from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import datetime
import shutil
import math
import json
import cv2
import os

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source_dir', type = Path, default = Path('.'))
parser.add_argument('-o', '--output_dir', type = Path, default = Path('.'))
parser.add_argument('--dynamic_set', type = str, default = "1")
parser.add_argument('--skip_cam', nargs = '*', default = [5,21,37,53])
parser.add_argument('--train_cam', nargs = '*', default = [1,3,4,7,9,10,11,12,14,15,16,19,20,26,29,31,32,34,36,38,40,42,43,44,45,46,48,51,52,58,61,63])
parser.add_argument('--camera_num', default = 66)
parser.add_argument('--timestep_indices', nargs = '*', default = list(range(15)))

args = parser.parse_args()

input_path = args.source_dir
datetime = '-'.join(str(datetime.datetime.now()).split())
output_path = args.output_dir / Path(datetime)

selected_set = args.dynamic_set

flip = {}
frames = []
flip2 = {}

camera_num = args.camera_num
timestep_indices = args.timestep_indices
scale = 1

skip_cam = [int(i) for i in args.skip_cam]
selected_cam = [int(i) for i in args.train_cam]
if len(selected_cam) == 0:
    selected_cam = list(range(camera_num))

camera_indices = [i for i in selected_cam if i not in skip_cam]

print("source path:", input_path)
print("output path:", output_path)
print("dynamic set:", selected_set)
print("timestep indices", timestep_indices)
print("camera indices", camera_indices)

output_path.mkdir(parents=True, exist_ok=True)
(output_path / Path('fg_masks')).mkdir(exist_ok=True)
(output_path / Path('images')).mkdir(exist_ok=True)
(output_path / Path('flame_param')).mkdir(exist_ok=True)

print("Processing FLAME params ...")
for ts in tqdm(timestep_indices):
    timestep_id = '0'*(5-len(str(ts))) + str(ts)
    f = open(input_path / Path('dynamic/outfit'+selected_set+'/'+str(ts)+'/Flame_Params.txt'))
    s = f.read()
    s = [[float(i) for i in ar.split()] for ar in s.split('\n')]
    f_off = open(input_path / Path('dynamic/outfit'+selected_set+'/'+str(ts)+'/offsets.txt'))
    s_off = f_off.read()
    s_off = np.array([[[float(i) for i in ar.split()] for ar in s_off.split('\n')]])
    s_off = np.concatenate((s_off, np.zeros(shape=[1,120,3])), axis=1) # (1, 5032+120, 3)
    s_off = np.zeros(shape=[1, 5143, 3]) # still bug in train.py

    translation = np.array([s[2][:]])
    rotation = np.array([s[1][:3][:]]) * -1
    neck_pose = np.array([s[1][3:6][:]]) * -1
    jaw_pose =  np.array([s[1][6:9][:]]) * -1
    eyes_pose = np.array([s[1][9:][:]]) * -1
    shape = s[0][:300]
    expr = np.array([s[0][300:]])
    static_offset = s_off
    dynamic_offset = np.zeros(shape=[1, 5143, 3])
    np.savez(output_path / Path('flame_param/'+timestep_id+'.npz'), 
            translation = translation, 
            rotation = rotation, 
            neck_pose = neck_pose, 
            jaw_pose = jaw_pose,
            eyes_pose = eyes_pose,
            shape = shape,
            expr = expr,
            static_offset = static_offset,
            dynamic_offset = dynamic_offset)
    if ts == timestep_indices[0]:
        np.savez(output_path / Path("canonical_flame_param.npz"), 
                translation = translation, 
                rotation = rotation, 
                neck_pose = neck_pose, 
                jaw_pose = jaw_pose,
                eyes_pose = eyes_pose,
                shape = shape,
                expr = np.zeros_like(expr),
                static_offset = static_offset,
                dynamic_offset = dynamic_offset)

camera_params = {}
w2c = {}
intrinsics = {}
sum_fx = 0.0
sum_fy = 0.0

rot_90 = np.array([[ 0, -1,  0,  0],
                   [ 1,  0,  0,  0],
                   [ 0,  0,  1,  0],
                   [ 0,  0,  0,  1]])

rot_180 = np.array([[-1,  0,  0,  0],
                    [ 0, -1,  0,  0],
                    [ 0,  0,  1,  0],
                    [ 0,  0,  0,  1]])

mirror = np.array([[-1,  0,  0,  0],
                   [ 0,  1,  0,  0],
                   [ 0,  0,  1,  0],
                   [ 0,  0,  0,  1]])

to_gs = np.array([[ 1,  0,  0,  0],
                  [ 0, -1,  0,  0],
                  [ 0,  0, -1,  0],
                  [ 0,  0,  0,  1]])

transforms = {}
transforms["timestep_indices"] = timestep_indices
transforms["camera_indices"] = camera_indices

print("processing transform.json ...")
for ts in tqdm(timestep_indices):
    for i in camera_indices:
        timestep_id = '0'*(5-len(str(ts))) + str(ts)
        
        cam =str(i)
        cam = '0'*(2-len(cam)) + cam
        f = open(input_path / Path('static/camera/camera'+cam+'.txt'))
        s = f.read()
        s = s.split('\n')
        cam = '0'*(5-len(cam)) + cam
        
        now_frame = {
            "timestep_index": int(timestep_id),
            "timestep_index_original": int(timestep_id),
            "timestep_id": "frame_" + timestep_id,
            "camera_index": i,
            "camera_id": "camera"+cam,
            "file_path": "images/"+timestep_id+"_"+cam+".png",
            "fg_mask_path": "fg_masks/"+timestep_id+"_"+cam+".png",
            "flame_param_path": "flame_param/"+timestep_id+".npz"
        }
        
        fl_y, fl_x = [math.ceil(float (i) * scale) for i in s[1].split()]
        cy, cx     = [math.ceil(float (i) * scale) for i in s[3].split()]
        m          = np.array([[float(j) for j in i.split('\t')] for i in s[10:14]])
        h = cy*2
        w = cx*2
        # sh = sz_y/h
        # sw = sz_x/w
        sh = 1.0
        sw = 1.0

        ### scale images
        w    *= sw
        fl_x *= sw
        cx   *= sw
        h    *= sh
        fl_y *= sh
        cy   *= sh

        intr = np.array([[      fl_x,          0.0,          cx],
                        [       0.0,         fl_y,          cy],
                        [       0.0,          0.0,         1.0],])

        r = np.identity(4)
        t = np.identity(4)
        t_off = np.identity(4)
        t_off2 = t_off.copy()
        t_off2[3:, :3] *= -1.0
        r[:3, :3] = m[:3, :3]
        t[:3,  3] = m[:3,  3] * -1.0 * 0.01

        r = r @ rot_90 
        r = r @ rot_180
        if i in flip:
            r = r @ rot_180
        if i in flip2:
            r = r @ rot_180
        r = r @ mirror
        r = r.T
        r2 = r.copy()
        r2 = r2.T
        r2 = r2 @ to_gs
        t2 = t.copy()
        t2[:3, 3] *= -1.0
        m2 = t2 @ r2 #c2w
        angle_x = math.atan(w / (fl_x * 2)) * 2
        angle_y = math.atan(h / (fl_y * 2)) * 2

        now_frame["transform_matrix"] = m2.tolist()
        now_frame["cx"] = cx
        now_frame["cy"] = cy
        now_frame["fl_x"] = fl_x
        now_frame["fl_y"] = fl_y
        now_frame["h"] = h
        now_frame["w"] = w
        now_frame["camera_angle_x"] = angle_x
        now_frame["camera_angle_y"] = angle_y

        frames.append(now_frame)

        if i == 0:
            camera_params["width"] = w
            camera_params["height"] = h
        sum_fx += fl_x
        sum_fy += fl_y

transforms["frames"] = frames
transforms["cx"] = transforms["frames"][0]["cx"]
transforms["cy"] = transforms["frames"][0]["cy"]
transforms["fl_x"] = transforms["frames"][0]["fl_x"]
transforms["fl_y"] = transforms["frames"][0]["fl_y"]
transforms["h"] = transforms["frames"][0]["h"]
transforms["w"] = transforms["frames"][0]["w"]
transforms["camera_angle_x"] = transforms["frames"][0]["camera_angle_x"]
transforms["camera_angle_y"] = transforms["frames"][0]["camera_angle_y"]

json_object = json.dumps(transforms, indent=4)

with open(output_path / Path("transforms.json"), "x") as outfile:
    outfile.write(json_object)
with open(output_path / Path("transforms_backup_flame.json"), "x") as outfile:
    outfile.write(json_object)
with open(output_path / Path("transforms_backup.json"), "x") as outfile:
    outfile.write(json_object)
with open(output_path / Path("transforms_val.json"), "x") as outfile:
    outfile.write(json_object)
with open(output_path / Path("transforms_train.json"), "x") as outfile:
    outfile.write(json_object)
with open(output_path / Path("transforms_test.json"), "x") as outfile:
    outfile.write(json_object) 

print("processing images ...")
for ts in tqdm(timestep_indices):
    for i in camera_indices:
        n = str(i)
        nn = '0'*(5-len(n)) + n
        timestep = str(ts)
        timestep = '0'*(5-len(timestep)) + timestep
        mask_img_path = input_path / Path('dynamic/alphas'+selected_set+'/'+n+'_'+str(ts)+'_mask.png')
        alpha_tmp_path = output_path / Path('fg_masks/tmp_'+nn+'.jpg')
        cmd = "convert " + str(mask_img_path) + " " + str(alpha_tmp_path)
        os.system(cmd)
        alpha_tmp_img = cv2.imread(alpha_tmp_path)
        h, w, channels = alpha_tmp_img.shape
        if i in flip: 
            alpha_tmp_img = cv2.rotate(alpha_tmp_img, cv2.ROTATE_180)

        alpha_png = cv2.rotate(alpha_tmp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        alpha_png = cv2.resize(alpha_png, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        alpha_png = cv2.GaussianBlur(alpha_png, (3, 3), 0) ## bluring edges
        cv2.imwrite(output_path / Path('fg_masks/'+timestep+'_'+nn+'.png'), alpha_png)
        os.remove(alpha_tmp_path)

        exr_path = input_path / Path('dynamic/outraw'+selected_set+'/'+n+'_'+str(ts)+'.png')
        jpg_tmp_path = output_path / Path('images/tmp_'+nn+'.jpg')
        cmd = "convert " + str(exr_path) + " " + str(jpg_tmp_path)
        os.system(cmd)
        jpg_tmp_img = cv2.imread(jpg_tmp_path)
        if i in flip: 
            jpg_tmp_img = cv2.rotate(jpg_tmp_img, cv2.ROTATE_180)
            
        img_png = cv2.rotate(jpg_tmp_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_png = cv2.resize(img_png, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path / Path('images/'+timestep+'_'+nn+'.png'), img_png)
        os.remove(jpg_tmp_path)