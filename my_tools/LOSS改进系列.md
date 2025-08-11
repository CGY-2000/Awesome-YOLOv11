# Loss系列 

1. SlideLoss,EMASlideLoss.(可动态调节正负样本的系数,让模型更加注重难分类,错误分类的样本上)[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)

    在ultralytics/utils/loss.py中的class v8DetectionLoss进行设定.(支持v11-detect、v11-seg、v11-pose)
    EMASlideLoss具体思想可以参考：https://www.bilibili.com/video/BV1W14y1i79U/?vd_source=c8452371e7ca510979593165c8d7ac27

2. FocalLoss,VarifocalLoss,QualityfocalLoss(支持v11-detect、v11-seg、v11-pose、v11-obb)

    项目视频百度云链接-20240111版本更新说明.

3. Normalized Gaussian Wasserstein Distance(支持v11-detect、v11-seg、v11-pose)[论文链接](https://arxiv.org/abs/2110.13389)

    在Loss中使用:
        在ultralytics/utils/loss.py中的BboxLoss class中的__init__函数里面设置self.nwd_loss为True.  
        比例系数调整self.iou_ratio, self.iou_ratio代表iou的占比,(1-self.iou_ratio)为代表nwd的占比.  
    在TAL标签分配中使用:
        在ultralytics/utils/tal.py中的def iou_calculation函数中进行更换即可.
    以上这两可以配合使用,也可以单独使用.

4. 定位损失系列(支持v11-detect、v11-seg、v11-pose)

    1. IoU,GIoU,DIoU,CIoU,EIoU,SIoU,MPDIoU,ShapeIoU.
    2. Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU,Inner-ShapeIoU,Inner-MPDIoU.
    3. Focaler-IoU系列(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,WIoU,MPDIoU,ShapeIoU)
    4. Powerful-IoU,Powerful-IoUV2,Inner-Powerful-IoU,Inner-Powerful-IoUV2,Focaler-Powerful-IoU,Focaler-Powerful-IoUV2[论文链接](https://www.sciencedirect.com/science/article/abs/pii/S0893608023006640)
    项目视频百度云链接-定位损失系列更换说明

    1. Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU,Powerful-IoU,Powerful-IoUV2).
    2. Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU,Powerful-IoU,Powerful-IoUV2).
    项目视频百度云链接-20240111版本更新说明