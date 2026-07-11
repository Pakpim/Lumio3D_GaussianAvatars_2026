## 1. Data prep

### Static
```
python dataprep/static.py
```
#### arguments 

- `-s` source_dir
- `-o` output_dir
- `--skip_cam` default: [5,21,37,53]
- `--train_cam` default: 32cams indices
- `--camera_num` default: 66

### Dynamic
```
python dataprep/dynamic.py
```
#### arguments 

- `-s` source_dir
- `-o` output_dir
- `--skip_cam` default: [5,21,37,53]
- `--train_cam` default: 32cams indices
- `--camera_num` default: 66
- `--dynamic_set` default: set 1
- `--timestep_indices` default: 0 to 14

## 2. Training
```
python train.py -s /rpool/data/pim/GA_res/bird_new_10_32cams -m /rpool/data/pim/GA_out/bird_new_v10_test02 --eval --bind_to_mesh --port 9999
```
#### arguments
- `-s` source_dir (output from 1.)
- `-m` model_dir
- more in `-h`

## 3. View
```
python local_viewer.py --point_path /rpool/data/pim/GA_out/bird_new_v10_test01/point_cloud/iteration_2000/point_cloud_bary.ply --sh_degree 3
```
#### arguments
- `--point_path` point_cloud_bary.ply path
- `--sh_degree` sh degree of model
- more in `-h`

## 4. render
```
python render.py -m /rpool/data/pim/GA_out/bird_new_v10_test01
```
#### arguments
- `-m` model_dir
- more in `-h`