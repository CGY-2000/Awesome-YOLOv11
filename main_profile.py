import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('/home/ubuntu/code/CGY/ultralytics-yolo11/ultralytics/cfg/models/11/yolo11-SPDConv.yaml')
    model.info(detailed=False)
    try:
        model.profile(imgsz=[640, 640])
    except Exception as e:
        print(e)
        pass
    model.fuse()