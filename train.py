# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。

# 训练时候输出的AMP Check使用的YOLO11n的权重不是代表载入了预训练权重的意思，只是用于测试AMP，正常的不需要理会。

# 使用项目前必看<项目视频百度云链接.txt>的第一行有一个必看的视频!!


# 在20250502更新中，修改保存权重的逻辑，训练结束(注意是正常训练结束后，手动停止的没有)后统一会保存4个模型，
# 分别是best.pt、last.pt、best_fp32.pt、last_fp32.pt，其中不带fp32后缀的是fp16格式保存的，
# 但由于有些模块对fp16非常敏感，会出现后续使用val.py的时候精度为0的情况，这种情况下可以用后缀带fp32去测试。

# 想找到哪些yaml是做轻量化的话可以用get_all_yaml_param_and_flops.py脚本，这个脚本里面有对应的教程视频。

# YOLO11配置文件路径：ultralytics/cfg/models/11
# YOLO12配置文件路径：ultralytics/cfg/models/12 预训练权重在这里下:https://github.com/sunsmarterjie/yolov12 Turbo版本
# YOLO13配置文件路径：ultralytics/cfg/models/13 预训练权重在这里下:https://github.com/iMoonLab/yolov13
# tensorboard --logdir = "绝对路径"
# tensorboard --logdir="/home/ubuntu/code/CGY/ultralytics-yolo11/runs"


import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from ultralytics import settings
settings["tensorboard"]=True

if __name__ == '__main__':
    DATASET_NAME = "visdrone2019"

    SCALE = "11s"

    CFG_NAME = "CAMSAM"

    CFG_FILE_NAME = (f"yolo{SCALE}-{CFG_NAME}" if CFG_NAME else f"yolo{SCALE}")
    print(f"yaml配置文件:{CFG_FILE_NAME}")
    print(f"数据集配置文件:/home/ubuntu/code/CGY/CGY_Datasets/{DATASET_NAME}/data.yaml")

    cfg_path = f"/home/ubuntu/code/CGY/ultralytics-yolo11/ultralytics/cfg/model_cfg/{CFG_FILE_NAME}.yaml"

    model = YOLO(cfg_path)# YOLO11

    # model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml') # YOLO11
    # model = YOLO('/home/ubuntu/code/CGY/ultralytics-yolo11/runs/train/visdrone2019/11s-tiny-EUCB/weights/best.pt') #断点训练
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=f'/home/ubuntu/code/CGY/CGY_Datasets/{DATASET_NAME}/data.yaml',
                project=f'/home/ubuntu/code/CGY/ultralytics-yolo11/runs/train/{DATASET_NAME}',
                name= SCALE + f"-{CFG_NAME}" if CFG_NAME else "baseline" ,  # 保存路径
                cache=False,
                imgsz=640,
                epochs=200,
                batch=8,
                close_mosaic=0, # 最后多少个epoch关闭mosaic数据增强，设置0代表全程开启mosaic训练
                workers=4, # Windows下出现莫名其妙卡主的情况可以尝试把workers设置为0
                # device='0,1', # 指定显卡和多卡训练参考<YOLOV11配置文件.md>下方常见错误和解决方案
                optimizer='SGD', # using SGD
                # patience=0, # set 0 to close earlystop.
                # resume=True, # 断点续训,YOLO初始化时选择last.pt,不懂就在百度云.txt找断点续训的视频
                # amp=False, # close amp | loss出现nan可以关闭amp
                # fraction=0.2,
                )