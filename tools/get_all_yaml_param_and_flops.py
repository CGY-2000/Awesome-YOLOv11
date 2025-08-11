import warnings
warnings.filterwarnings('ignore')
import torch, glob, tqdm
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# 使用教程视频：https://pan.baidu.com/s/1ZDzglU7EIzzfaUDhAhagBA?pwd=kg8k

if __name__ == '__main__':
    flops_dict = {}
    model_size, model_type = 'n', 'yolo11'

    if model_type == 'yolo11':
        yaml_base_path = 'ultralytics/cfg/models/11'
    elif model_type == 'yolo12':
        yaml_base_path = 'ultralytics/cfg/models/12'

    for yaml_path in tqdm.tqdm(glob.glob(f'{yaml_base_path}/*.yaml')):
        yaml_path = yaml_path.replace(f'{yaml_base_path}/{model_type}', f'{yaml_base_path}/{model_type}{model_size}')

        if 'DCN' in yaml_path:
            continue

        try:
            model = YOLO(yaml_path)
            model.fuse()
            n_l, n_p, n_g, flops = model_info(model.model)
            flops_dict[yaml_path] = [flops, n_p]
        except:
            continue
    
    sorted_items = sorted(flops_dict.items(), key=lambda x: x[1][0])
    for key, value in sorted_items:
        print(f"{key}: {value[0]:.2f} GFLOPs {value[1]:,} Params")