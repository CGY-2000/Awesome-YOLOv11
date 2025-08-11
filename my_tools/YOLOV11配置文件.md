# YOLOV11改进方案(持续更新) 创作人：BiliBili魔傀面具

<a id="b"></a>

#### 目前支持的一些block (部分block可能会与主结构有冲突,具体以是否能运行为主)

##### C3k2系列
C3k2, C3k2_Faster, C3k2_ODConv, C3k2_Faster_EMA, C3k2_DBB, C3k2_CloAtt, C3k2_SCConv, C3k2_ScConv, C3k2_EMSC, C3k2_EMSCP, C3k2_KW, C3k2_DCNv2, C3k2_DCNv2, C3k2_OREPA, C3k2_REPVGGOREPA, C3k2_DCNv2_Dynamic, C3k2_MSBlock, C3k2_ContextGuided, C3k2_DLKA, C3k2_EMBC, C3k2_Parc, C3k2_DWR, C3k2_RFAConv, C3k2_RFCBAMConv, C3k2_RFCAConv, C3k2_MLCA, C3k2_AKConv, C3k2_UniRepLKNetBlock, C3k2_DRB, C3k2_DWR_DRB, C3k2_AggregatedAtt....(C3k2系列的改进都合适、太多了就不一一标注了)
##### 其他系列
VoVGSCSP, VoVGSCSPC, RCSOSA, CSP_EDLAN

<a id="c"></a>

#### 目前整合的一些注意力机制 还需要别的注意力机制可从[github](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)拉取对应的代码到ultralytics/nn/extra_modules/attention.py即可. 视频教程可看项目视频中的(如何在yaml配置文件中添加注意力层)
EMA, SimAM, SpatialGroupEnhance, BiLevelRoutingAttention, BiLevelRoutingAttention_nchw, TripletAttention, CoordAtt, CBAM, BAMBlock, EfficientAttention(CloFormer中的注意力), LSKBlock, SEAttention, CPCA, deformable_LKA, EffectiveSEModule, LSKA, SegNext_Attention, DAttention(Vision Transformer with Deformable Attention CVPR2022), FocusedLinearAttention(ICCV2023), MLCA, TransNeXt_AggregatedAttention, LocalWindowAttention(CVPR2023-EfficientViT), ELA, CAA, CAFM, AFGCAttention(Neural Networks ECCV2024)

### YOLOV11
1. ultralytics/cfg/models/11/yolo11-efficientViT.yaml

    (CVPR2023)efficientViT替换yolo11主干.

2. ultralytics/cfg/models/11/yolo11-fasternet.yaml

    (CVPR2023)fasternet替换yolo11主干.

3. ultralytics/cfg/models/11/yolo11-timm.yaml

    使用timm支持的主干网络替换yolo11主干.timm的内容可看[这期视频](https://www.bilibili.com/video/BV1Mx4y1A7jy/)

4. ultralytics/cfg/models/11/yolo11-convnextv2.yaml

    使用convnextv2网络替换yolo11主干.

5. ultralytics/cfg/models/11/yolo11-dyhead.yaml

    添加基于注意力机制的目标检测头到yolo11中.

6. ultralytics/cfg/models/11/yolo11-bifpn.yaml

    添加BIFPN到yolo11中.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持五种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        其中支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.
    项目视频百度云链接:关于BIFPN的说明

7. ultralytics/cfg/models/11/yolo11-C3k2-Faster.yaml

    使用C3k2-Faster替换C3k2.(使用FasterNet中的FasterBlock替换C3k2中的Bottleneck)
    项目视频百度云链接:20240729版本更新说明

8. ultralytics/cfg/models/11/yolo11-C3k2-ODConv.yaml

    使用C3k2-ODConv替换C3k2.(使用ODConv替换C3k2中的Bottleneck中的Conv)
    项目视频百度云链接:20240729版本更新说明

9. ultralytics/cfg/models/11/yolo11-EfficientFormerV2.yaml

    使用EfficientFormerV2网络替换yolo11主干.(需要看[常见错误和解决方案的第五点](#a))  
10. ultralytics/cfg/models/11/yolo11-C3k2-Faster-EMA.yaml

    使用C3k2-Faster-EMA替换C3k2.(C3k2-Faster-EMA推荐可以放在主干上,Neck和head部分可以选择C3k2-Faster)
    项目视频百度云链接:20240729版本更新说明

11. ultralytics/cfg/models/11/yolo11-C3k2-DBB.yaml

    使用C3k2-DBB替换C3k2.(使用DiverseBranchBlock替换C3k2中的Bottleneck中的Conv)
    项目视频百度云链接:20240729版本更新说明

12. 增加Adaptive Training Sample Selection匹配策略.

    在ultralytics/utils/loss.py中的class v8DetectionLoss中自行选择对应的self.assigner即可.  
    此ATSS匹配策略目前占用显存比较大,因此使用的时候需要设置更小的batch,后续会进行优化这一功能.

13. ultralytics/cfg/models/11/yolo11-slimneck.yaml

    使用[VoVGSCSP\VoVGSCSPC和GSConv](https://github.com/AlanLi1997/slim-neck-by-gsconv)替换yolo11 neck中的C3k2和Conv.

14. ultralytics/cfg/models/11/yolo11-attention.yaml

    可以看项目视频-如何在yaml配置文件中添加注意力层  
    多种注意力机制在yolo11中的使用. [多种注意力机制github地址](https://github.com/z1069614715/objectdetection_script/tree/master/cv-attention)  
    目前内部整合的注意力可看[链接](#c)

15. Asymptotic Feature Pyramid Network[reference](https://github.com/gyyang23/AFPN/tree/master)

    a. ultralytics/cfg/models/11/yolo11-AFPN-P345.yaml  
    b. ultralytics/cfg/models/11/yolo11-AFPN-P345-Custom.yaml  
    c. ultralytics/cfg/models/11/yolo11-AFPN-P2345.yaml  
    d. ultralytics/cfg/models/11/yolo11-AFPN-P2345-Custom.yaml  
    其中Custom中的block支持这些[结构](#b) [B站介绍说明](https://www.bilibili.com/video/BV1bh411A7yj/)

16. ultralytics/cfg/models/11/yolo11-vanillanet.yaml

    vanillanet替换yolo11主干.

17. ultralytics/cfg/models/11/yolo11-C3k2-CloAtt.yaml

    使用C3k2-CloAtt替换C3k2.(使用CloFormer中的具有全局和局部特征的注意力机制添加到C3k2中的Bottleneck中)(需要看[常见错误和解决方案的第五点](#a))  
    项目视频百度云链接:20240729版本更新说明

18. ultralytics/cfg/models/11/yolo11-RevCol.yaml

    使用[(ICLR2023)Reversible Column Networks](https://github.com/megvii-research/RevCol)对yolo11主干进行重设计.
    视频:https://www.bilibili.com/video/BV1Mh4y1y76u/

19. ultralytics/cfg/models/11/yolo11-LSKNet.yaml

    LSKNet(2023旋转目标检测SOTA的主干)替换yolo11主干.

20. ultralytics/cfg/models/11/yolo11-C3k2-SCConv.yaml

    SCConv(CVPR2020 http://mftp.mmcheng.net/Papers/20cvprSCNet.pdf)与C3k2融合.
    项目视频百度云链接:SCConv和ScConv的使用教程
    针对v11的架构，C3k2-SCConv在n大小下只适用于替换通道数为1024的c3k2，如果不是n大小可以尝试替换更多，不报错就证明没问题。

21. ultralytics/cfg/models/11/yolo11-C3k2-SCcConv.yaml

    ScConv(CVPR2023 https://openaccess.thecvf.com/content/CVPR2023/papers/Li_SCConv_Spatial_and_Channel_Reconstruction_Convolution_for_Feature_Redundancy_CVPR_2023_paper.pdf)与C3k2融合.  
    (取名为SCcConv的原因是在windows下命名是不区分大小写的)
    项目视频百度云链接:SCConv和ScConv的使用教程
    针对v11的架构，C3k2-SCcConv在n大小下只适用于替换通道数为1024的c3k2，如果不是n大小可以尝试替换更多，不报错就证明没问题。

22. MPDiou.[论文链接](https://arxiv.org/pdf/2307.07662v1.pdf)

    请看LOSS改进系列.md

23. ultralytics/cfg/models/11/yolo11-LAWDS.yaml

    Light Adaptive-weight downsampling.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:LAWDS讲解

24. ultralytics/cfg/models/11/yolo11-C3k2-EMSC.yaml

    Efficient Multi-Scale Conv.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:EMSC,EMSCP讲解

25. ultralytics/cfg/models/11/yolo11-C3k2-EMSCP.yaml

    Efficient Multi-Scale Conv Plus.自研模块,具体讲解请看百度云链接中的视频.
    项目视频百度云链接:EMSC,EMSCP讲解

26. ultralytics/cfg/models/11/yolo11-RCSOSA.yaml

    使用[RCS-YOLO](https://github.com/mkang315/RCS-YOLO/tree/main)中的RCSOSA替换C3k2.
    项目视频百度云链接:20240908版本更新说明

27. ultralytics/cfg/models/11/yolo11-KernelWarehouse.yaml
    
    使用[Towards Parameter-Efficient Dynamic Convolution](https://github.com/OSVAI/KernelWarehouse)添加到yolo11中.  
    使用此模块需要注意,在epoch0-20的时候精度会非常低,过了20epoch会正常.
    项目视频百度云链接:20240803版本更新说明

28. Normalized Gaussian Wasserstein Distance.[论文链接](https://arxiv.org/abs/2110.13389)

    在Loss中使用:
        在ultralytics/utils/loss.py中的BboxLoss class中的__init__函数里面设置self.nwd_loss为True.  
        比例系数调整self.iou_ratio, self.iou_ratio代表iou的占比,(1-self.iou_ratio)为代表nwd的占比.  
    在TAL标签分配中使用:
        在ultralytics/utils/tal.py中的def iou_calculation函数中进行更换即可.
    以上这两可以配合使用,也可以单独使用.

29. SlideLoss and EMASlideLoss.[Yolo-Face V2](https://github.com/Krasjet-Yu/YOLO-FaceV2/blob/master/utils/loss.py)

    在ultralytics/utils/loss.py中的class v8DetectionLoss进行设定.

30. ultralytics/cfg/models/11/yolo11-C3k2-DySnakeConv.yaml

    [DySnakeConv](https://github.com/YaoleiQi/DSCNet)与C3k2融合.
    项目视频百度云链接:20240803版本更新说明

31. ultralytics/cfg/models/11/yolo11-EfficientHead.yaml

    对检测头进行重设计,支持多种轻量化检测头.
    项目视频百度云链接:Detect_Efficient、Segment_Efficient改进说明

32. ultralytics/cfg/models/11/yolo11-aux.yaml

    参考YOLOV7-Aux对YOLO11添加额外辅助训练头,在训练阶段参与训练,在最终推理阶段去掉.  
    其中辅助训练头的损失权重系数可在ultralytics/utils/loss.py中的class v8DetectionLoss中的__init__函数中的self.aux_loss_ratio设定,默认值参考yolov7为0.25.

33. ultralytics/cfg/models/11/yolo11-C3k2-DCNV2.yaml

    使用C3k2-DCNV2替换C3k2.(DCNV2为可变形卷积V2)
    项目视频百度云链接:DCNV2,DCNV3,DyHeadWithDCNV3相关讲解

34. ultralytics/cfg/models/11/yolo11-C3k2-DCNV3.yaml

    使用C3k2-DCNV3替换C3k2.([DCNV3](https://github.com/OpenGVLab/InternImage)为可变形卷积V3(CVPR2023,众多排行榜的SOTA))  
    官方中包含了一些指定版本的DCNV3 whl包,下载后直接pip install xxx即可.具体和安装DCNV3可看百度云链接中的视频.
    项目视频百度云链接:DCNV2,DCNV3,DyHeadWithDCNV3相关讲解

35. ultralytics/cfg/models/11/yolo11-dyhead-DCNV3.yaml

    使用[DCNV3](https://github.com/OpenGVLab/InternImage)替换DyHead中的DCNV2.安装DCNV3可看百度云链接中的视频.
    项目视频百度云链接:DCNV2,DCNV3,DyHeadWithDCNV3相关讲解

36. ultralytics/cfg/models/11/yolo11-FocalModulation.yaml

    使用[Focal Modulation](https://github.com/microsoft/FocalNet)替换SPPF.

37. ultralytics/cfg/models/11/yolo11-C3k2-OREPA.yaml

    使用C3k2-OREPA替换C3k2.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)
    项目视频百度云链接:关于OREPA,REPVGGOREPA,EMSC-OREPA,EMSCP-OREPA的说明

38. ultralytics/cfg/models/11/yolo11-C3k2-REPVGGOREPA.yaml

    使用C3k2-REPVGGOREPA替换C3k2.[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)
    项目视频百度云链接:关于OREPA,REPVGGOREPA,EMSC-OREPA,EMSCP-OREPA的说明

39. ultralytics/cfg/models/11/yolo11-swintransformer.yaml

    SwinTransformer-Tiny替换yolo11主干.

40. ultralytics/cfg/models/11/yolo11-repvit.yaml

    [CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)替换yolo11主干.

41. ultralytics/cfg/models/11/yolo11-fasternet-bifpn.yaml

    fasternet与bifpn的结合.  
    其中BIFPN中有三个可选参数：
    1. Fusion  
        其中BIFPN中的Fusion模块支持四种: weight, adaptive, concat, bifpn(default), SDI  
        其中weight, adaptive, concat出自[paper链接-Figure 3](https://openreview.net/pdf?id=q2ZaVU6bEsT), SDI出自[U-NetV2](https://github.com/yaoppeng/U-Net_v2)
    2. node_mode  
        其中目前(后续会更新喔)支持这些[结构](#b)
    3. head_channel  
        BIFPN中的通道数,默认设置为256.

42. ultralytics/cfg/models/11/yolo11-C3k2-DCNV2-Dynamic.yaml

    利用自研注意力机制MPCA强化DCNV2中的offset和mask.
    项目视频百度云链接:MPCA与DCNV2_Dynamic的说明

43. ultralytics/cfg/models/11/yolo11-goldyolo.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute进行改进特征融合模块
    介绍可看这期视频：https://www.bilibili.com/video/BV1Bp4y1w7MB/

44. ultralytics/cfg/models/11/yolo11-C3k2-ContextGuided.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided改进C3k2.

45. ultralytics/cfg/models/11/yolo11-ContextGuidedDown.yaml

    使用[CGNet](https://github.com/wutianyiRosun/CGNet/tree/master)中的Light-weight Context Guided DownSample进行下采样.

46. ultralytics/cfg/models/11/yolo11-C3k2-MSBlock.yaml

    使用[YOLO-MS](https://github.com/FishAndWasabi/YOLO-MS/tree/main)中的MSBlock改进C3k2.
    项目视频百度云链接:20231010版本更新说明

47. ultralytics/cfg/models/11/yolo11-C3k2-DLKA.yaml

    使用[deformableLKA](https://github.com/xmindflow/deformableLKA)改进C3k2.
    项目视频百度云链接:20231010版本更新说明

48. ultralytics/cfg/models/11/yolo11-GFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN改进Neck.
    项目视频百度云链接:20231010版本更新说明

49. ultralytics/cfg/models/11/yolo11-SPDConv.yaml

    使用[SPDConv](https://github.com/LabSAINT/SPD-Conv/tree/main)进行下采样.
    项目视频百度云链接:20231010版本更新说明

50. ultralytics/cfg/models/11/yolo11-EfficientRepBiPAN.yaml

    使用[YOLOV6](https://github.com/meituan/YOLOv6/tree/main)中的EfficientRepBiPAN改进Neck.
    项目视频百度云链接:20231010版本更新说明

51. ultralytics/cfg/models/11/yolo11-C3k2-EMBC.yaml

    使用[Efficientnet](https://blog.csdn.net/weixin_43334693/article/details/131114618?spm=1001.2014.3001.5501)中的MBConv与EffectiveSE改进C3k2.
    项目视频百度云链接:20240920版本更新说明

52. ultralytics/cfg/models/11/yolo11-SPPF-LSKA.yaml

    使用[LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention)注意力机制改进SPPF,增强多尺度特征提取能力.

53. ultralytics/cfg/models/11/yolo11-C3k2-DAttention.yaml

    使用[Vision Transformer with Deformable Attention(CVPR2022)](https://github.com/LeapLabTHU/DAT)改进C3k2.(需要看[常见错误和解决方案的第五点](#a))  
    使用注意点请看百度云视频.(DAttention(Vision Transformer with Deformable Attention CVPR2022)使用注意说明.)

54. ultralytics/cfg/models/11/yolo11-CSwinTransformer.yaml

    使用[CSWin-Transformer(CVPR2022)](https://github.com/microsoft/CSWin-Transformer/tree/main)替换yolo11主干.(需要看[常见错误和解决方案的第五点](#a))

55. ultralytics/cfg/models/11/yolo11-AIFI.yaml

    使用[RT-DETR](https://arxiv.org/pdf/2304.08069.pdf)中的Attention-based Intrascale Feature Interaction(AIFI)改进yolo11.
    项目视频百度云链接:20231107版本更新说明

56. ultralytics/cfg/models/11/yolo11-C3k2-Parc.yaml

    使用[ParC-Net](https://github.com/hkzhang-git/ParC-Net/tree/main)中的ParC_Operator改进C3k2.(需要看[常见错误和解决方案的第五点](#a))  
    使用注意点请看百度云视频.(20231031更新说明)    

57. ultralytics/cfg/models/11/yolo11-C3k2-DWR.yaml

    使用[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)模块,加强从网络高层的可扩展感受野中提取特征.
    项目视频百度云链接:20231107版本更新说明

58. ultralytics/cfg/models/11/yolo11-C3k2-RFAConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFAConv改进yolo11.

59. ultralytics/cfg/models/11/yolo11-C3k2-RFCBAMConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFCBAMConv改进yolo11.

60. ultralytics/cfg/models/11/yolo11-C3k2-RFCAConv.yaml

    使用[RFAConv](https://github.com/Liuchen1997/RFAConv/tree/main)中的RFCAConv改进yolo11.

61. ultralytics/cfg/models/11/yolo11-HGNetV2.yaml

    使用HGNetV2作为YOLO11的backbone.

62. ultralytics/cfg/models/11/yolo11-GhostHGNetV2.yaml

    使用Ghost_HGNetV2作为YOLO11的backbone.
    项目视频百度云链接:GhostHGNetV2、RepHGNetV2改进说明

63. ultralytics/cfg/models/11/yolo11-RepHGNetV2.yaml

    使用Rep_HGNetV2作为YOLO11的backbone.
    项目视频百度云链接:GhostHGNetV2、RepHGNetV2改进说明

64. ultralytics/cfg/models/11/yolo11-seg-EfficientHead.yaml(实例分割)

    对检测头进行重设计,支持多种轻量化检测头.
    项目视频百度云链接:Detect_Efficient、Segment_Efficient改进说明

65. ultralytics/cfg/models/11/yolo11-C3k2-FocusedLinearAttention.yaml

    使用[FLatten Transformer(ICCV2023)](https://github.com/LeapLabTHU/FLatten-Transformer)中的FocusedLinearAttention改进C3k2.(需要看[常见错误和解决方案的第五点](#a))    
    使用注意点请看百度云视频.(20231114版本更新说明.)

66. IoU,GIoU,DIoU,CIoU,EIoU,SIoU更换方法.

    请看LOSS改进系列.md

67. Inner-IoU,Inner-GIoU,Inner-DIoU,Inner-CIoU,Inner-EIoU,Inner-SIoU更换方法.

    请看LOSS改进系列.md

68. Inner-MPDIoU更换方法.

    请看LOSS改进系列.md

69. ultralytics/cfg/models/11/yolo11-C3k2-MLCA.yaml

    使用[Mixed Local Channel Attention 2023](https://github.com/wandahangFY/MLCA/tree/master)改进C3k2.(用法请看百度云视频-20231129版本更新说明)

70. ultralytics/cfg/models/11/yolo11-C3k2-AKConv.yaml

    使用[AKConv 2023](https://github.com/CV-ZhangXin/AKConv)改进C3k2.(用法请看百度云视频-20231129版本更新说明)

71. ultralytics/cfg/models/11/yolo11-unireplknet.yaml

    使用[CVPR2024 UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)替换yolo11主干.
    项目视频百度云链接:20231207版本更新说明

72. ultralytics/cfg/models/11/yolo11-C3k2-UniRepLKNetBlock.yaml

    使用[CVPR2024 UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的UniRepLKNetBlock改进C3k2.
    项目视频百度云链接:20231207版本更新说明

73. ultralytics/cfg/models/11/yolo11-C3k2-DRB.yaml

    使用[CVPR2024 UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock改进C3k2.
    项目视频百度云链接:20231207版本更新说明

74. ultralytics/cfg/models/11/yolo11-C3k2-DWR-DRB.yaml

    使用[CVPR2024 UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[DWRSeg](https://arxiv.org/abs/2212.01173)中的Dilation-wise Residual(DWR)的模块进行二次创新后改进C3k2.
    项目视频百度云链接:20231207版本更新说明

75. ultralytics/cfg/models/11/yolo11-ASF.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion改进yolo11.
    项目视频百度云链接:20231217版本更新说明

76. ultralytics/cfg/models/11/yolo11-ASF-P2.yaml

    在ultralytics/cfg/models/11/yolo11-ASF.yaml的基础上进行二次创新，引入P2检测层并对网络结构进行优化.
    项目视频百度云链接:20231217版本更新说明

77. ultralytics/cfg/models/11/yolo11-CSP-EDLAN.yaml

    使用[DualConv](https://github.com/ChipsGuardian/DualConv)打造CSP Efficient Dual Layer Aggregation Networks改进yolo11.
    项目视频百度云链接:20231217版本更新说明

78. ultralytics/cfg/models/11/yolo11-TransNeXt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)改进yolo11的backbone.(需要看[常见错误和解决方案的第五点](#a))   

79. ultralytics/cfg/models/11/yolo11-AggregatedAttention.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进yolo11的backbone.(需要看[常见错误和解决方案的第五点](#a))   
    项目视频百度云链接:20231227版本更新说明

80. ultralytics/cfg/models/11/yolo11-C3k2-AggregatedAtt.yaml

    使用[TransNeXt](https://github.com/DaiShiResearch/TransNeXt)中的聚合感知注意力改进C3k2.(需要看[常见错误和解决方案的第五点](#a))   
    项目视频百度云链接:20231227版本更新说明

81. ultralytics/cfg/models/11/yolo11-bifpn-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对BIFPN进行二次创新.
    项目视频百度云链接:20231227版本更新说明

82. ultralytics/cfg/models/11/yolo11-SDI.yaml

    使用[U-NetV2](https://github.com/yaoppeng/U-Net_v2)中的 Semantics and Detail Infusion Module对yolo11中的feature fusion部分进行重设计.
    项目视频百度云链接:20231227版本更新说明

83. Shape-IoU,Inner-Shape-IoU更换方法.

    请看LOSS改进系列.md

84. FocalLoss,VarifocalLoss,QualityfocalLoss更换方法.

    请看LOSS改进系列.md

85. Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)更换方法.

    请看LOSS改进系列.md

86. Inner-Wise-IoU(v1,v2,v3)系列(IoU,WIoU,EIoU,GIoU,DIoU,CIoU,SIoU,MPDIoU,ShapeIoU)更换方法.

    请看LOSS改进系列.md

87. ultralytics/cfg/models/11/yolo11-goldyolo-asf.yaml

    利用华为2023最新GOLD-YOLO中的Gatherand-Distribute与[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion进行二次创新改进yolo11的neck.
    项目视频百度云链接:20240116版本更新说明

88. ultralytics/cfg/models/11/yolo11-C3k2-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)改进C3k2.(请关闭AMP进行训练,编译使用教程请看20240116版本更新说明)

89. ultralytics/cfg/models/11/yolo11-dyhead-DCNV4.yaml

    使用[DCNV4](https://github.com/OpenGVLab/DCNv4)对DyHead进行二次创新.(请关闭AMP进行训练,编译使用教程请看20240116版本更新说明)

90. ultralytics/cfg/models/11/yolo11-HSFPN.yaml

    使用[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN改进yolo11的neck.
    项目视频百度云链接:20240122版本更新说明

91. ultralytics/cfg/models/11/yolo11-HSPAN.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN改进yolo11的neck.
    项目视频百度云链接:20240122版本更新说明

92. soft-nms(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,ShapeIoU)

    soft-nms替换nms.(建议:仅在val.py时候使用,具体替换请看20240122版本更新说明)

93. ultralytics/cfg/models/11/yolo11-dysample.yaml

    使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进yolo11-neck中的上采样.
    项目视频百度云链接:20240122版本更新说明

94. ultralytics/cfg/models/11/yolo11-CARAFE.yaml

    使用[ICCV2019 CARAFE](https://arxiv.org/abs/1905.02188)改进yolo11-neck中的上采样.
    项目视频百度云链接:20240122版本更新说明

95. ultralytics/cfg/models/11/yolo11-HWD.yaml

    使用[Haar wavelet downsampling](https://www.sciencedirect.com/science/article/abs/pii/S0031320323005174)改进yolo11的下采样.(请关闭AMP情况下使用)
    项目视频百度云链接:20240122版本更新说明

96. Focaler-IoU系列(IoU,GIoU,DIoU,CIoU,EIoU,SIoU,WIoU,MPDIoU,ShapeIoU)

    请看LOSS改进系列.md

97. ultralytics/cfg/models/11/yolo11-GDFPN.yaml

    使用[DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO)中的RepGFPN与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)进行二次创新改进Neck.
    项目视频百度云链接:20240203版本更新说明

98. ultralytics/cfg/models/11/yolo11-HSPAN-DySample.yaml

    对[MFDS-DETR](https://github.com/JustlfC03/MFDS-DETR)中的HS-FPN进行二次创新后得到HSPAN再进行创新,使用[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)改进其上采样模块.
    项目视频百度云链接:20240203版本更新说明

99. ultralytics/cfg/models/11/yolo11-ASF-DySample.yaml

    使用[ASF-YOLO](https://github.com/mkang315/ASF-YOLO)中的Attentional Scale Sequence Fusion与[ICCV2023 DySample](https://arxiv.org/abs/2308.15085)组合得到Dynamic Sample Attentional Scale Sequence Fusion.
    项目视频百度云链接:20240203版本更新说明

100. ultralytics/cfg/models/11/yolo11-SEAMHead.yaml

    使用[YOLO-Face V2](https://arxiv.org/pdf/2208.02019v2.pdf)中的遮挡感知注意力改进Head,使其有效地处理遮挡场景.
    项目视频百度云链接:20240920版本更新说明

101. ultralytics/cfg/models/11/yolo11-MultiSEAMHead.yaml

    使用[YOLO-Face V2](https://arxiv.org/pdf/2208.02019v2.pdf)中的遮挡感知注意力改进Head,使其有效地处理遮挡场景.
    项目视频百度云链接:20240920版本更新说明

102. ultralytics/cfg/models/11/yolo11-C3k2-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)改进yolo11中的C3k2.(详细请看20240216更新说明)
    项目视频百度云链接:20240216版本更新说明

103. ultralytics/cfg/models/11/yolo11-C3k2-iRMB.yaml

    使用[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB改进C3k2.(详细请看20240216更新说明)

104. ultralytics/cfg/models/11/yolo11-C3k2-iRMB-Cascaded.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C3k2.(详细请看20240216更新说明)

105. ultralytics/cfg/models/11/yolo11-C3k2-iRMB-DRB.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C3k2.(详细请看20240216更新说明)

106. ultralytics/cfg/models/11/yolo11-C3k2-iRMB-SWC.yaml

    使用[shift-wise conv](https://arxiv.org/abs/2401.12736)对[EMO ICCV2023](https://github.com/zhangzjn/EMO)中的iRMB进行二次创新来改进C3k2.(详细请看20240216更新说明)

107. ultralytics/cfg/models/11/yolo11-C3k2-VSS.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)对C3k2中的BottleNeck进行改进,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.<编译教程请看百度云视频-20240219版本更新说明>
    项目视频百度云链接:20240219版本更新说明

108. ultralytics/cfg/models/11/yolo11-C3k2-LVMB.yaml

    使用最新的Mamba架构[Mamba-UNet中的VSS](https://github.com/ziyangwang007/Mamba-UNet)与Cross Stage Partial进行结合,使其能更有效地捕获图像中的复杂细节和更广泛的语义上下文.<编译教程请看百度云视频-20240219版本更新说明>
    项目视频百度云链接:20240219版本更新说明

109. Powerful-IoU系列.

    请看LOSS改进系列.md

110. ultralytics/cfg/models/11/yolo11-RepNCSPELAN.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行改进yolo11.
    项目视频百度云链接:20240222版本更新说明

111. ultralytics/cfg/models/11/yolo11-DBBNCSPELAN.yaml

    使用[Diverse Branch Block CVPR2021](https://arxiv.org/abs/2103.13425)对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolo11.
    项目视频百度云链接:20240222版本更新说明

112. ultralytics/cfg/models/11/yolo11-OREPANCSPELAN.yaml

    使用[Online Convolutional Re-parameterization (CVPR2022)](https://github.com/JUGGHM/OREPA_CVPR2022/tree/main)对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolo11.
    项目视频百度云链接:20240222版本更新说明

113. ultralytics/cfg/models/11/yolo11-DRBNCSPELAN.yaml

    使用[UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock对[YOLOV9](https://github.com/WongKinYiu/yolov9)中的RepNCSPELAN进行二次创新后改进yolo11.
    项目视频百度云链接:20240222版本更新说明

114. ultralytics/cfg/models/11/yolo11-v7DS.yaml

    使用[YOLOV7 CVPR2023](https://arxiv.org/abs/2207.02696)的下采样结构改进YOLO11中的下采样.
    项目视频百度云链接:20240229版本更新说明

115. ultralytics/cfg/models/11/yolo11-ADown.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的下采样结构改进YOLO11中的下采样.
    项目视频百度云链接:20240229版本更新说明

116. ultralytics/cfg/models/11/yolo11-PGI.yaml

    使用[YOLOV9](https://github.com/WongKinYiu/yolov9)的programmable gradient information改进YOLO11.(PGI模块可在训练结束后去掉)
    项目视频百度云链接:20240229版本更新说明

117. ultralytics/cfg/models/11/yolo11-C3k2-DynamicConv.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的DynamicConv改进C3k2.
    项目视频百度云链接:20240303版本更新说明

118. ultralytics/cfg/models/11/yolo11-C3k2-GhostDynamicConv.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的GhostModule改进C3k2.
    项目视频百度云链接:20240303版本更新说明

119. ultralytics/cfg/models/11/yolo11-DynamicHGNetV2.yaml

    使用[CVPR2024 parameternet](https://arxiv.org/pdf/2306.14525v2.pdf)中的DynamicConv对[CVPR2024 RTDETR](https://arxiv.org/abs/2304.08069)中的HGBlokc进行二次创新.
    项目视频百度云链接:20240303版本更新说明

120. ultralytics/cfg/models/11/yolo11-C3k2-RVB.yaml

    使用[CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)中的RepViTBlock改进C3k2.
    项目视频百度云链接:20240309版本更新说明

121. ultralytics/cfg/models/11/yolo11-C3k2-RVB-EMA.yaml

    使用[CVPR2024 RepViT](https://github.com/THU-MIG/RepViT/tree/main)中的RepViTBlock和EMA注意力机制改进C3k2.
    项目视频百度云链接:20240309版本更新说明

122. Lightweight Shared Convolutional Detection Head

    自研轻量化检测头.
    detect:ultralytics/cfg/models/11/yolo11-LSCD.yaml
    seg:ultralytics/cfg/models/11/yolo11-seg-LSCD.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-LSCD.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-LSCD.yaml
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.
    3. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    综合以上，我们可以让检测头做到参数量更少、计算量更少的情况下，尽可能减少精度的损失.
    项目视频百度云链接:20240309版本更新说明

123. ultralytics/cfg/models/11/yolo11-DGCST.yaml

    使用[Lightweight Object Detection](https://arxiv.org/abs/2403.01736)中的Dynamic Group Convolution Shuffle Transformer改进yolo11.
    项目视频百度云链接:20240309版本更新说明

124. Task Align Dynamic Detection Head

    自研任务对齐动态检测头
    detect:ultralytics/cfg/models/11/yolo11-TADDH.yaml
    seg:ultralytics/cfg/models/11/yolo11-seg-TADDH.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-TADDH.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-TADDH.yaml
    1. GroupNorm在FCOS论文中已经证实可以提升检测头定位和分类的性能.
    2. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.并且在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    3. 参照TOOD的思想,除了标签分配策略上的任务对齐,我们也在检测头上进行定制任务对齐的结构,现有的目标检测器头部通常使用独立的分类和定位分支,这会导致两个任务之间缺乏交互,TADDH通过特征提取器从多个卷积层中学习任务交互特征,得到联合特征,定位分支使用DCNV2和交互特征生成DCNV2的offset和mask,分类分支使用交互特征进行动态特征选择.
    项目视频百度云链接:20240314版本更新说明

125. ultralytics/cfg/models/11/yolo11-rmt.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)改进yolo11的主干.(需要看[常见错误和解决方案的第五点](#a)) 
    项目视频百度云链接:20240323版本更新说明

126. ultralytics/cfg/models/11/yolo11-C3k2-RetBlock.yaml

    使用[CVPR2024 RMT](https://arxiv.org/abs/2309.11523)中的RetBlock改进C3k2.
    项目视频百度云链接:20240323版本更新说明

127. ultralytics/cfg/models/11/yolo11-ELA-HSFPN.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN.
    项目视频百度云链接:20240323版本更新说明

128. ultralytics/cfg/models/11/yolo11-CA-HSFPN.yaml

    使用[Coordinate Attention CVPR2021](https://github.com/houqb/CoordAttention)改进HSFPN.
    项目视频百度云链接:20240323版本更新说明

129. ultralytics/cfg/models/11/yolo11-ELA-HSFPN-TADDH.yaml

    使用[Efficient Local Attention](https://arxiv.org/abs/2403.01123)改进HSFPN,使用自研动态动态对齐检测头改进Head.

130. ultralytics/cfg/models/11/yolo11-pkinet.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)改进backbone.(需要安装mmcv和mmengine)
    项目视频百度云链接:20240330版本更新说明

131. ultralytics/cfg/models/11/yolo11-C3k2-PKI.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的PKIModule和CAA模块改进C3k2.
    项目视频百度云链接:20240330版本更新说明

132. ultralytics/cfg/models/11/yolo11-RepNCSPELAN_CAA.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块改进RepNCSPELAN.
    项目视频百度云链接:20240330版本更新说明

133. ultralytics/cfg/models/11/yolo11-CAA-HSFPN.yaml

    使用[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA模块HSFPN.
    项目视频百度云链接:20240330版本更新说明

134. ultralytics/cfg/models/11/yolo11-C3k2-fadc.yaml

    使用[CVPR2024 Frequency-Adaptive Dilated Convolution](https://github.com/Linwei-Chen/FADC)改进C3k2.
    项目视频百度云链接:20240406版本更新说明

135. ultralytics/cfg/models/11/yolo11-FDPN.yaml

    自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    1. 通过定制的特征聚焦模块与特征扩散机制，能让每个尺度的特征都具有详细的上下文信息，更有利于后续目标的检测与分类。
    2. 定制的特征聚焦模块可以接受三个尺度的输入，其内部包含一个Inception-Style的模块，其利用一组并行深度卷积来捕获丰富的跨多个尺度的信息。
    3. 通过扩散机制使具有丰富的上下文信息的特征进行扩散到各个检测尺度.
    项目视频百度云链接:20240406版本更新说明

136. ultralytics/cfg/models/11/yolo11-FDPN-TADDH.yaml

    自研结构的融合.
    1. 自研特征聚焦扩散金字塔网络(Focusing Diffusion Pyramid Network)
    2. 自研任务对齐动态检测头(Task Align Dynamic Detection Head)
    项目视频百度云链接:20240408版本更新说明

137. ultralytics/cfg/models/11/yolo11-C3k2-PPA.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Parallelized Patch-Aware Attention Module改进C3k2.
    项目视频百度云链接:20240408版本更新说明

138. ultralytics/cfg/models/11/yolo11-FDPN-DASI.yaml

    使用[HCFNet](https://github.com/zhengshuchen/HCFNet)中的Dimension-Aware Selective Integration Module对自研的Focusing Diffusion Pyramid Network再次创新.
    项目视频百度云链接:20240408版本更新说明

139. ultralytics/cfg/models/11/yolo11-CSMHSA.yaml

    对Mutil-Head Self-Attention进行创新得到Cross-Scale Mutil-Head Self-Attention.
    1. 由于高维通常包含更高级别的语义信息，而低维包含更多细节信息，因此高维信息作为query，而低维信息作为key和Value，将两者结合起来可以利用高维的特征帮助低维的特征进行精细过滤，可以实现更全面和丰富的特征表达。
    2. 通过使用高维的上采样信息进行Query操作，可以更好地捕捉到目标的全局信息，从而有助于增强模型对目标的识别和定位能力。
    项目视频百度云链接:20240414版本更新说明

140. ultralytics/cfg/models/11/yolo11-SRFD.yaml

    使用[A Robust Feature Downsampling Module for Remote Sensing Visual Tasks](https://ieeexplore.ieee.org/document/10142024)改进yolo11的下采样.
    项目视频百度云链接:20240420版本更新说明

141. ultralytics/cfg/models/11/yolo11-CSFCN.yaml

    使用[Context and Spatial Feature Calibration for Real-Time Semantic Segmentation](https://github.com/kaigelee/CSFCN/tree/main)中的Context and Spatial Feature Calibration模块改进yolo11.
    项目视频百度云链接:20240420版本更新说明

142. ultralytics/cfg/models/11/yolo11-mobilenetv4.yaml

    使用[MobileNetV4](https://github.com/jaiwei98/MobileNetV4-pytorch/tree/main)改进yolo11-backbone.
    项目视频百度云链接:20240428版本更新说明

143. ultralytics/cfg/models/11/yolo11-CGAFusion.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的content-guided attention fusion改进yolo11-neck.
    项目视频百度云链接:20240428版本更新说明

144. ultralytics/cfg/models/11/yolo11-CAFMFusion.yaml

    利用具有[HCANet](https://github.com/summitgao/HCANet)中的CAFM，其具有获取全局和局部信息的注意力机制进行二次改进content-guided attention fusion.
    项目视频百度云链接:20240428版本更新说明

145. ultralytics/cfg/models/11/yolo11-RGCSPELAN.yaml

    自研RepGhostCSPELAN.
    1. 参考GhostNet中的思想(主流CNN计算的中间特征映射存在广泛的冗余)，采用廉价的操作生成一部分冗余特征图，以此来降低计算量和参数量。
    2. 舍弃yolov5与yolo11中常用的BottleNeck，为了弥补舍弃残差块所带来的性能损失，在梯度流通分支上使用RepConv，以此来增强特征提取和梯度流通的能力，并且RepConv可以在推理的时候进行融合，一举两得。
    3. 可以通过缩放因子控制RGCSPELAN的大小，使其可以兼顾小模型和大模型。
    项目视频百度云链接:20240501版本更新说明

146. Lightweight Asymmetric Detection Head

    detect:ultralytics/cfg/models/11/yolo11-LADH.yaml
    segment:ultralytics/cfg/models/11/yolo11-seg-LADH.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-LADH.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-LADH.yaml
    使用[Faster and Lightweight: An Improved YOLOv5 Object Detector for Remote Sensing Images](https://www.mdpi.com/2072-4292/15/20/4974)中的Lightweight Asymmetric Detection Head改进yolo11-head.
    项目视频百度云链接:20240505版本更新说明

147. ultralytics/cfg/models/11/yolo11-C3k2-Faster-CGLU.yaml

    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU对CVPR2023中的FasterNet进行二次创新.
    项目视频百度云链接:20240505版本更新说明

148. ultralytics/cfg/models/11/yolo11-SDFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的superficial detail fusion module改进yolo11-neck.
    项目视频百度云链接:20240512版本更新说明

149. ultralytics/cfg/models/11/yolo11-PSFM.yaml

    使用[PSFusion](https://github.com/Linfeng-Tang/PSFusion)中的profound semantic fusion module改进yolo11-neck.
    项目视频百度云链接:20240512版本更新说明

150. Lightweight Shared Convolutional Separamter BN Detection Head

    基于自研轻量化检测头(LSCD)上，参考NASFPN的设计思路把GN换成BN，并且BN层参数不共享.
    detect:ultralytics/cfg/models/11/yolo11-LSCSBD.yaml
    seg:ultralytics/cfg/models/11/yolo11-seg-LSCSBD.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-LSCSBD.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-LSCSBD.yaml
    1. 由于不同层级之间特征的统计量仍存在差异，Normalization layer依然是必须的，由于直接在共享参数的检测头中引入BN会导致其滑动平均值产生误差，而引入 GN 又会增加推理时的开销，因此我们参考NASFPN的做法，让检测头共享卷积层，而BN则分别独立计算。
    项目视频百度云链接:20240512版本更新说明

151. ultralytics/cfg/models/11/yolo11-starnet.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)改进yolo11-backbone.
    项目视频百度云链接:20240513版本更新说明

152. ultralytics/cfg/models/11/yolo11-C3k2-Star.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock改进C3k2.
    项目视频百度云链接:20240513版本更新说明

153. ultralytics/cfg/models/11/yolo11-C3k2-Star-CAA.yaml

    使用[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock和[CVPR2024 PKINet](https://github.com/PKINet/PKINet)中的CAA改进C3k2.
    项目视频百度云链接:20240513版本更新说明

154. ultralytics/cfg/models/11/yolo11-starnet-C3k2-Star-LSCD.yaml

    轻量化模型组合.
    1. CVPR2024-StarNet Backbone.
    2. C3k2-Star.
    3. Lightweight Shared Convolutional Detection Head.
    项目视频百度云链接:20240513版本更新说明

155. ultralytics/cfg/models/11/yolo11-C3k2-KAN.yaml

    KAN In! Mamba Out! Kolmogorov-Arnold Networks.[Pytorch-Conv-KAN](https://github.com/IvanDrokin/torch-conv-kan)
    目前支持:
    1. FastKANConv2DLayer
    2. KANConv2DLayer
    3. KALNConv2DLayer
    4. KACNConv2DLayer
    5. KAGNConv2DLayer
    项目视频百度云链接:20240523版本更新说明

156. ultralytics/cfg/models/11/yolo11-nmsfree.yaml

    仿照yolov10的思想采用双重标签分配和一致匹配度量进行训练,后处理不需要NMS!
    项目视频百度云链接:20240526版本更新说明

157. ultralytics/cfg/models/11/yolo11-EIEStem.yaml

    提出了一种新的EIEStem模块，旨在作为图像识别任务中的高效前端模块。该模块结合了提取边缘信息的SobelConv分支和提取空间信息的卷积分支，能够学习到更加丰富的图像特征表示。
    1. 通过SobelConv分支，可以提取图像的边缘信息。由于Sobel滤波器可以检测图像中强度的突然变化，因此可以很好地捕捉图像的边缘特征。这些边缘特征在许多计算机视觉任务中都非常重要，例如图像分割和物体检测。
    2. EIEStem模块还结合空间信息，除了边缘信息，EIEStem还通过池化分支提取空间信息，保留重要的空间信息。结合边缘信息和空间信息，可以帮助模型更好地理解图像内容。
    3. 通过3D组卷积高效实现Sobel算子。
    项目视频百度云链接:20240526版本更新说明

158. ultralytics/cfg/models/11/yolo11-C3k2-EIEM.yaml

    提出了一种新的C3k2-EIEM模块，旨在作为图像识别任务中的高效前端模块。该模块结合了提取边缘信息的SobelConv分支和提取空间信息的卷积分支，能够学习到更加丰富的图像特征表示。
    1. 边缘信息学习: 卷积神经网络 (CNN)通常擅长学习空间信息，但是对于提取图像中的边缘信息可能稍显不足。EIEStem 模块通过SobelConv分支，显式地提取图像的边缘特征。Sobel滤波器是一种经典的边缘检测滤波器，可以有效地捕捉图像中强度的突然变化，从而获得重要的边缘信息。
    2. 空间信息保留: 除了边缘信息，图像中的空间信息也同样重要。EIEStem模块通过一个额外的卷积分支 (conv_branch) 来提取空间信息。与SobelCon 分支不同，conv_branch提取的是原始图像的特征，可以保留丰富的空间细节。
    3. 特征融合: EIEStem模块将来自SobelConv分支和conv_branch提取的特征进行融合 (concatenate)。 这种融合操作使得学习到的特征表示既包含了丰富的边缘信息，又包含了空间信息，能够更加全面地刻画图像内容。
    项目视频百度云链接:20240526版本更新说明

159. ultralytics/cfg/models/11/yolo11-ContextGuideFPN.yaml

    Context Guide Fusion Module（CGFM）是一个创新的特征融合模块，旨在改进YOLO11中的特征金字塔网络（FPN）。该模块的设计考虑了多尺度特征融合过程中上下文信息的引导和自适应调整。
    1. 上下文信息的有效融合：通过SE注意力机制，模块能够在特征融合过程中捕捉并利用重要的上下文信息，从而增强特征表示的有效性，并有效引导模型学习检测目标的信息，从而提高模型的检测精度。
    2. 特征增强：通过权重化的特征重组操作，模块能够增强重要特征，同时抑制不重要特征，提升特征图的判别能力。
    3. 简单高效：模块结构相对简单，不会引入过多的计算开销，适合在实时目标检测任务中应用。
    这期视频讲解在B站:https://www.bilibili.com/video/BV1Vx4y1n7hZ/
    项目视频百度云链接:20240601版本更新说明

160. ultralytics/cfg/models/11/yolo11-C3k2-DEConv.yaml

    使用[DEA-Net](https://github.com/cecret3350/DEA-Net)中的detail-enhanced convolution改进C3k2.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.
    由于V11的C3k2的特性，在n大小模型时候前两个C3k2没法用C3k2_DEConv，如果用比n大的模型可以尝试替换前两个C3k2.
    项目视频百度云链接:20240601版本更新说明

161. ultralytics/cfg/models/11/yolo11-LSDECD.yaml

    基于自研轻量化检测头上(LSCD)，使用detail-enhanced convolution进一步改进，提高检测头的细节捕获能力，进一步改善检测精度.
    关于DEConv在运行的时候重参数化后比重参数化前的计算量还要大的问题:是因为重参数化前thop库其计算不准的问题,看重参数化后的参数即可.
    detect:ultralytics/cfg/models/11/yolo11-LSDECD.yaml
    segment:ultralytics/cfg/models/11/yolo11-seg-LSDECD.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-LSDECD.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-LSDECD.yaml
    1. DEA-Net中设计了一个细节增强卷积（DEConv），具体来说DEConv将先验信息整合到普通卷积层，以增强表征和泛化能力。然后，通过使用重参数化技术，DEConv等效地转换为普通卷积，不需要额外的参数和计算成本。
    项目视频百度云链接:20240601版本更新说明

162. ultralytics/cfg/models/11/yolo11-C3k2-SMPCGLU.yaml

    Self-moving Point Convolutional GLU模型改进C3k2.<需要编译请看百度云视频-20240601版本更新说明>
    SMP来源于[CVPR2023-SMPConv](https://github.com/sangnekim/SMPConv),Convolutional GLU来源于[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt).
    1. 普通的卷积在面对数据中的多样性和复杂性时，可能无法捕捉到有效的特征，因此我们采用了SMPConv，其具备最新的自适应点移动机制，从而更好地捕捉局部特征，提高特征提取的灵活性和准确性。
    2. 在SMPConv后添加CGLU，Convolutional GLU 结合了卷积和门控机制，能够选择性地通过信息通道，提高了特征提取的有效性和灵活性。
    3. 这种类似Transformer的encoder结构，利用DEConv进行特征提取，CGLU可以进一步的非线性变换和加强特征提取，提升模型的表达能力。
    项目视频百度云链接:20240601版本更新说明

163. ultralytics/cfg/models/11/yolo11-C3k2-Heat.yaml

    使用[CVPR2025 vHeat](https://github.com/MzeroMiko/vHeat/tree/main)中的HeatBlock改进C3k2.(需要看[常见错误和解决方案的第五点](#a)) 
    项目视频百度云链接:20240609版本更新说明

164. Re-CalibrationFPN

    为了加强浅层和深层特征的相互交互能力，推出重校准特征金字塔网络(Re-CalibrationFPN).
    P2345：ultralytics/cfg/models/11/yolo11-ReCalibrationFPN-P2345.yaml(带有小目标检测头的ReCalibrationFPN)
    P345：ultralytics/cfg/models/11/yolo11-ReCalibrationFPN-P345.yaml
    P3456：ultralytics/cfg/models/11/yolo11-ReCalibrationFPN-P3456.yaml(带有大目标检测头的ReCalibrationFPN)
    1. 浅层语义较少，但细节丰富，有更明显的边界和减少失真。此外，深层蕴藏着丰富的物质语义信息。因此，直接融合低级具有高级特性的特性可能导致冗余和不一致。为了解决这个问题，我们提出了[SBA](https://github.com/Barrett-python/DuAT)模块，它有选择地聚合边界信息和语义信息来描绘更细粒度的物体轮廓和重新校准物体的位置。
    2. 相比传统的FPN结构，[SBA](https://github.com/Barrett-python/DuAT)模块引入了高分辨率和低分辨率特征之间的双向融合机制，使得特征之间的信息传递更加充分，进一步提升了多尺度特征融合的效果。
    3. [SBA](https://github.com/Barrett-python/DuAT)模块通过自适应的注意力机制，根据特征图的不同分辨率和内容，自适应地调整特征的权重，从而更好地捕捉目标的多尺度特征。
    项目视频百度云链接:20240609版本更新说明

165. ultralytics/cfg/models/11/yolo11-WaveletPool.yaml

    使用[Wavelet Pooling](https://openreview.net/forum?id=rkhlb8lCZ)改进YOLO11的上采样和下采样。
    项目视频百度云链接:20240613版本更新说明

166. ultralytics/cfg/models/11/yolo11-CSP-PTB.yaml

    Cross Stage Partial - Partially Transformer Block
    在计算机视觉任务中，Transformer结构因其强大的全局特征提取能力而受到广泛关注。然而，由于Transformer结构的计算复杂度较高，直接将其应用于所有通道会导致显著的计算开销。为了在保证高效特征提取的同时降低计算成本，我们设计了一种混合结构，将输入特征图分为两部分，分别由CNN和Transformer处理，结合了卷积神经网络(CNN)和Transformer机制的模块，旨在增强特征提取的能力。
    我们提出了一种名为CSP_PTB(Cross Stage Partial - Partially Transformer Block)的模块，旨在结合CNN和Transformer的优势，通过对输入通道进行部分分配来优化计算效率和特征提取能力。
    1. 融合局部和全局特征：多项研究表明，CNN的感受野大小较少，导致其只能提取局部特征，但Transformer的MHSA能够提取全局特征，能够同时利用两者的优势。
    2. 保证高效特征提取的同时降低计算成本：为了能引入Transformer结构来提取全局特征又不想大幅度增加计算复杂度，因此提出Partially Transformer Block，只对部分通道使用TransformerBlock。
    3. MHSA_CGLU包含Mutil-Head-Self-Attention和[ConvolutionalGLU(TransNext CVPR2024)](https://github.com/DaiShiResearch/TransNeXt)，其中Mutil-Head-Self-Attention负责提取全局特征，ConvolutionalGLU用于增强非线性特征表达能力，ConvolutionalGLU相比于传统的FFN，具有更强的性能。
    4. 可以根据不同的模型大小和具体的运行情况调节用于Transformer的通道数。
    项目视频百度云链接:20240613版本更新说明

167. [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)

    集成Mamba-YOLO.(需要编译请看百度云视频-20240619版本更新说明)
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-T.yaml
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-B.yaml
    ultralytics/cfg/models/mamba-yolo/Mamba-YOLO-L.yaml
    ultralytics/cfg/models/mamba-yolo/yolo-mamba-seg.yaml
    项目视频百度云链接:20240619版本更新说明

168. ultralytics/cfg/models/11/yolo11-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块改进yolo11的neck.
    项目视频百度云链接:20240619版本更新说明

169. ultralytics/cfg/models/11/yolo11-bifpn-GLSA.yaml

    使用[GLSA](https://github.com/Barrett-python/DuAT)模块对bifpn进行二次创新.
    项目视频百度云链接:20240619版本更新说明

170. ultralytics/cfg/models/11/yolo11-SOEP.yaml  
    
    小目标在正常的P3、P4、P5检测层上略显吃力，比较传统的做法是加上P2检测层来提升小目标的检测能力，但是同时也会带来一系列的问题，例如加上P2检测层后计算量过大、后处理更加耗时等问题，日益激发需要开发新的针对小目标有效的特征金字塔，我们基于原本的PAFPN上进行改进，提出SmallObjectEnhancePyramid，相对于传统的添加P2检测层，我们使用P2特征层经过SPDConv得到富含小目标信息的特征给到P3进行融合，然后使用CSP思想和基于[AAAI2024的OmniKernel](https://ojs.aaai.org/index.php/AAAI/article/view/27907)进行改进得到CSP-OmniKernel进行特征整合，OmniKernel模块由三个分支组成，即三个分支，即全局分支、大分支和局部分支、以有效地学习从全局到局部的特征表征，最终从而提高小目标的检测性能。
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    出现这个报错的:RuntimeError: cuFFT error: CUFFT_INTERNAL_ERROR,如果你是40系显卡,需要更新torch大于2.0，并且cuda大于12.0.
    项目视频百度云链接:20240627版本更新说明

171. ultralytics/cfg/models/11/yolo11-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进yolo11-neck.(需要看[常见错误和解决方案的第五点](#a))  
    项目视频百度云链接:20240627版本更新说明

172. ultralytics/cfg/models/11/yolo11-p6-CTrans.yaml

    使用[[AAAI2022] UCTransNet](https://github.com/McGregorWwww/UCTransNet/tree/main)中的ChannelTransformer改进yolo11-neck.(带有p6版本)(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20240627版本更新说明  

173. ultralytics/cfg/models/11/yolo11-C3k2-WTConv.yaml

    使用[ECCV2024 Wavelet Convolutions for Large Receptive Fields](https://github.com/BGU-CS-VIL/WTConv)中的WTConv改进C3k2-BottleNeck.
    项目视频百度云链接:20240720版本更新说明

174. ultralytics/cfg/models/11/yolo11-CGRFPN.yaml

    Context-Guided Spatial Feature Reconstruction Feature Pyramid Network.
    1. 借鉴[ECCV2024-CGRSeg](https://github.com/nizhenliang/CGRSeg)中的Rectangular Self-Calibration Module经过精心设计,用于空间特征重建和金字塔上下文提取,它在水平和垂直方向上捕获全局上下文，并获得轴向全局上下文来显式地建模矩形关键区域.
    2. PyramidContextExtraction Module使用金字塔上下文提取模块（PyramidContextExtraction），有效整合不同层级的特征信息，提升模型的上下文感知能力。
    3. FuseBlockMulti 和 DynamicInterpolationFusion 这些模块用于多尺度特征的融合，通过动态插值和多特征融合，进一步提高了模型的多尺度特征表示能力和提升模型对复杂背景下目标的识别能力。
    项目视频百度云链接:20240720版本更新说明

175. ultralytics/cfg/models/11/yolo11-FeaturePyramidSharedConv.yaml

    1. 多尺度特征提取
        通过使用不同膨胀率的卷积层，模块能够提取不同尺度的特征。这对捕捉图像中不同大小和不同上下文的信息非常有利。
        低膨胀率捕捉局部细节，高膨胀率捕捉全局上下文。
    2. 参数共享
        使用共享的卷积层 self.share_conv，大大减少了需要训练的参数数量。相比于每个膨胀率使用独立的卷积层，共享卷积层能够减少冗余，提升模型效率。
        减少了模型的存储和计算开销，提升了计算效率。
    3. 高效的通道变换
        通过1x1卷积层 self.cv1 和 self.cv2，模块能够高效地调整通道数，并进行特征融合。1x1卷积层在减少参数量的同时还能保留重要的特征信息。
    4. 更细粒度的特征提取
        FeaturePyramidSharedConv 使用卷积操作进行特征提取，能够捕捉更加细粒度的特征。相比之下，SPPF 的池化操作可能会丢失一些细节信息。
        卷积操作在特征提取时具有更高的灵活性和表达能力，可以更好地捕捉图像中的细节和复杂模式。
    项目视频百度云链接:20240729版本更新说明

176. ultralytics/cfg/models/11/yolo11-C3k2-FMB.yaml

    使用[ECCV2024 SMFANet](https://github.com/Zheng-MJ/SMFANet/tree/main)的Feature Modulation block改进C3k2.
    项目视频百度云链接:20240729版本更新说明

177. ultralytics/cfg/models/11/yolo11-LDConv.yaml

    使用[LDConv](https://github.com/CV-ZhangXin/LDConv/tree/main)改进下采样.
    项目视频百度云链接:20240803版本更新说明

178. ultralytics/cfg/models/11/yolo11-C3k2-gConv.yaml

    使用[Rethinking Performance Gains in Image Dehazing Networks](https://arxiv.org/abs/2209.11448)的gConvblock改进C3k2.
    项目视频百度云链接:20240803版本更新说明

179. ultralytics/cfg/models/11/yolo11-MAFPN.yaml

    使用[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN改进Neck.
    项目视频百度云链接:20240803版本更新说明

180. ultralytics/cfg/models/11/yolo11-BIMAFPN.yaml

    利用BIFPN的思想对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次改进得到BIMAFPN.
    项目视频百度云链接:20240803版本更新说明

181. APT(Adaptive Power Transformation)-TAL.

    为了使不同gt预测对的匹配质量和损失权重更具鉴别性，我们通过自定义的PowerTransformer显著增强高质量预测框的权重，抑制低质量预测框的影响，并使模型在学习的过程可以更关注质量高的预测框。
    修改方式：ultralytics/utils/tal.py中的get_box_metrics函数中的power参数设置为True.
    项目视频百度云链接:20240813版本更新说明

182. ultralytics/cfg/models/11/yolo11-C3k2-WDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的WDBB改进C3k2.
    项目视频百度云链接:20240813版本更新说明

183. ultralytics/cfg/models/11/yolo11-C3k2-DeepDBB.yaml

    使用[YOLO-MIF](https://github.com/wandahangFY/YOLO-MIF)中的DeepDBB改进C3k2.
    项目视频百度云链接:20240813版本更新说明

184. ultralytics/cfg/models/11/yolo11-AIFIRepBN.yaml

    使用[ICML-2024 SLAB](https://github.com/xinghaochen/SLAB)中的RepBN改进AIFI.
    项目视频百度云链接:20240813版本更新说明

185. ultralytics/cfg/models/11/yolo11-C3k2-AdditiveBlock.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock改进C3k2.
    项目视频百度云链接:20240822版本更新说明

186. ultralytics/cfg/models/11/yolo11-C3k2-AdditiveBlock-CGLU.yaml

    使用[CAS-ViT](https://github.com/Tianfang-Zhang/CAS-ViT)中的AdditiveBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C3k2.
    项目视频百度云链接:20240822版本更新说明

187. ultralytics/cfg/models/11/yolo11-EMBSFPN.yaml

    基于BIFPN、[MAF-YOLO](https://arxiv.org/pdf/2407.04381)、[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)提出全新的Efficient Multi-Branch&Scale FPN.
    Efficient Multi-Branch&Scale FPN拥有<轻量化>、<多尺度特征加权融合>、<多尺度高效卷积模块>、<高效上采样模块>、<全局异构核选择机制>。
    1. 具有多尺度高效卷积模块和全局异构核选择机制，Trident网络的研究表明，具有较大感受野的网络更适合检测较大的物体，反之，较小尺度的目标则从较小的感受野中受益，因此我们在FPN阶段，对于不同尺度的特征层选择不同的多尺度卷积核以适应并逐步获得多尺度感知场信息。
    2. 借鉴BIFPN中的多尺度特征加权融合，能把Concat换成Add来减少参数量和计算量的情况下，还能通过不同尺度特征的重要性进行自适用选择加权融合。
    3. 高效上采样模块来源于CVPR2024-EMCAD中的EUCB，能够在保证一定效果的同时保持高效性。
    项目视频百度云链接:20240822版本更新说明

188. ultralytics/cfg/models/11/yolo11-C3k2-MSMHSA-CGLU.yaml

    使用[CMTFNet](https://github.com/DrWuHonglin/CMTFNet/tree/main)中的M2SA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C3k2.
    项目视频百度云链接:20240831版本更新说明

189. ultralytics/cfg/models/11/yolo11-CSP-PMSFA.yaml

    自研模块:CSP-Partial Multi-Scale Feature Aggregation.
    1. 部分多尺度特征提取：参考CVPR2020-GhostNet、CVPR2024-FasterNet的思想，采用高效的PartialConv，该模块能够从输入中提取多种尺度的特征信息，但它并不是在所有通道上进行这种操作，而是部分（Partial）地进行，从而提高了计算效率。
    2. 增强的特征融合: 最后的 1x1 卷积层通过将不同尺度的特征融合在一起，同时使用残差连接将输入特征与处理后的特征相加，有效保留了原始信息并引入了新的多尺度信息，从而提高模型的表达能力。
    项目视频百度云链接:20240831版本更新说明

191. ultralytics/cfg/models/11/yolo11-C3k2-MogaBlock.yaml

    使用[MogaNet ICLR2024](https://github.com/Westlake-AI/MogaNet)中的MogaBlock改进C3k2.
    项目视频百度云链接:20240908版本更新说明

192. ultralytics/cfg/models/11/yolo11-C3k2-SHSA.yaml

    使用[SHViT CVPR2024](https://github.com/ysj9909/SHViT)中的SHSABlock改进C3k2.
    项目视频百度云链接:20240920版本更新说明

193. ultralytics/cfg/models/11/yolo11-C3k2-SHSA-CGLU.yaml

    使用[SHViT CVPR2024](https://github.com/ysj9909/SHViT)中的SHSABlock与[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进C3k2.
    项目视频百度云链接:20240920版本更新说明

194. ultralytics/cfg/models/11/yolo11-C3k2-SMAFB.yaml

    使用[SMAFormer BIBM2024](https://github.com/CXH-Research/SMAFormer)中的SMAFormerBlock改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20240920版本更新说明

195. ultralytics/cfg/models/11/yolo11-C3k2-SMAFB-CGLU.yaml

    使用[SMAFormer BIBM2024](https://github.com/CXH-Research/SMAFormer)中的SMAFormerBlock与[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20240920版本更新说明

196. ultralytics/cfg/models/11/yolo11-FreqFFPN.yaml

    使用[FreqFusion TPAMI2024](https://github.com/Linwei-Chen/FreqFusion)中的FreqFusion改进Neck.(这个需要python3.10,不然最后保存模型会出错.)
    项目视频百度云链接:20240920版本更新说明

197. ultralytics/cfg/models/11/yolo11-MutilBackbone-DAF.yaml

    自研MutilBackbone-DynamicAlignFusion.
    1. 为了避免在浅层特征图上消耗过多计算资源，设计的MutilBackbone共享一个stem的信息，这个设计有利于避免计算量过大，推理时间过大的问题。
    2. 为了避免不同Backbone信息融合出现不同来源特征之间的空间差异，我们为此设计了DynamicAlignFusion，其先通过融合来自两个不同模块学习到的特征，然后生成一个名为DynamicAlignWeight去调整各自的特征，最后使用一个可学习的通道权重，其可以根据输入特征动态调整两条路径的权重，从而增强模型对不同特征的适应能力。
    项目视频百度云链接:20241007版本更新说明

198. ultralytics/cfg/models/11/yolo11-C3k2-IdentityFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的IdentityFormer改进C3k2.
    项目视频百度云链接:20241007版本更新说明

199. ultralytics/cfg/models/11/yolo11-C3k2-RandomMixing.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的RandomMixingFormer改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20241007版本更新说明

200. ultralytics/cfg/models/11/yolo11-C3k2-PoolingFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的PoolingFormer改进C3k2.
    项目视频百度云链接:20241007版本更新说明

201. ultralytics/cfg/models/11/yolo11-C3k2-ConvFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的ConvFormer改进C3k2.
    项目视频百度云链接:20241007版本更新说明

202. ultralytics/cfg/models/11/yolo11-C3k2-CaFormer.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的CaFormer改进C3k2.
    项目视频百度云链接:20241007版本更新说明

203. ultralytics/cfg/models/11/yolo11-C3k2-IdentityFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的IdentityFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进C3k2.
    项目视频百度云链接:20241007版本更新说明

204. ultralytics/cfg/models/11/yolo11-C3k2-RandomMixing-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的RandomMixing和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20241007版本更新说明

205. ultralytics/cfg/models/11/yolo11-C3k2-PoolingFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的PoolingFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进C3k2.
    项目视频百度云链接:20241007版本更新说明

206. ultralytics/cfg/models/11/yolo11-C3k2-ConvFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的ConvFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进C3k2.
    项目视频百度云链接:20241007版本更新说明

207. ultralytics/cfg/models/11/yolo11-C3k2-CaFormer-CGLU.yaml

    使用[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)中的CaFormer和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的CGLU改进C3k2.
    项目视频百度云链接:20241007版本更新说明

208. ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationEnhance.yaml

    自研CSP-MutilScaleEdgeInformationEnhance.
    MutilScaleEdgeInformationEnhance模块结合了多尺度特征提取、边缘信息增强和卷积操作。它的主要目的是从不同尺度上提取特征，突出边缘信息，并将这些多尺度特征整合到一起，最后通过卷积层输出增强的特征。这个模块在特征提取和边缘增强的基础上有很好的表征能力.
    1. 多尺度特征提取：通过 nn.AdaptiveAvgPool2d 进行多尺度的池化，提取不同大小的局部信息，有助于捕捉图像的多层次特征。
    2. 边缘增强：EdgeEnhancer 模块专门用于提取边缘信息，使得网络对边缘的敏感度增强，这对许多视觉任务（如目标检测、语义分割等）有重要作用。
    3. 特征融合：将不同尺度下提取的特征通过插值操作对齐到同一尺度，然后将它们拼接在一起，最后经过卷积层融合成统一的特征表示，能够提高模型对多尺度特征的感知。
    项目视频百度云链接:20241027版本更新说明

209. ultralytics/cfg/models/11/yolo11-C3k2-FFCM.yaml

    使用[Efficient Frequency-Domain Image Deraining with Contrastive Regularization ECCV2024](https://github.com/deng-ai-lab/FADformer)中的Fused_Fourier_Conv_Mixer改C3k2.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20241027版本更新说明

210. Rep Shared Convolutional Detection Head

    自研重参数轻量化检测头.
    detect:ultralytics/cfg/models/11/yolo11-RSCD.yaml
    seg:ultralytics/cfg/models/11/yolo11-seg-RSCD.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-RSCD.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-RSCD.yaml
    1. 通过使用共享卷积，可以大幅减少参数数量，这使得模型更轻便，特别是在资源受限的设备上.但由于共享参数可能限制模型的表达能力，因为不同特征可能需要不同的卷积核来捕捉复杂的模式。共享参数可能无法充分捕捉这些差异。为了尽量弥补实现轻量化所采取的共享卷积带来的负面影响，我们使用可重参数化卷积，通过引入更多的可学习参数，网络可以更有效地从数据中提取特征，进而弥补轻量化模型后可能带来的精度丢失问题，并且重参数化卷积可以大大提升参数利用率，并且在推理阶段与普通卷积无差，为模型带来无损的优化方案。
    2. 在使用共享卷积的同时，为了应对每个检测头所检测的目标尺度不一致的问题，使用Scale层对特征进行缩放.
    项目视频百度云链接:20241103版本更新说明

211. ultralytics/cfg/models/11/yolo11-C3k2-SFHF.yaml

    使用[SFHformer ECCV2024](https://github.com/deng-ai-lab/SFHformer)中的block改进C3k2.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20241112版本更新说明

212. ultralytics/cfg/models/11/yolo11-CSP-FreqSpatial.yaml

    FreqSpatial 是一个融合时域和频域特征的卷积神经网络（CNN）模块。该模块通过在时域和频域中提取特征，旨在捕捉不同层次的空间和频率信息，以增强模型在处理图像数据时的鲁棒性和表示能力。模块的主要特点是将 Scharr 算子（用于边缘检测）与 时域卷积 和 频域卷积 结合，通过多种视角捕获图像的结构特征。
    1. 时域特征提取：从原始图像中提取出基于空间结构的特征，主要捕捉图像的细节、边缘信息等。
    2. 频域特征提取：从频率域中提取出频率相关的模式，捕捉到图像的低频和高频成分，能够帮助模型在全局和局部的尺度上提取信息。
    3. 特征融合：将时域和频域的特征进行加权相加，得到最终的输出特征图。这种加权融合允许模型同时考虑空间结构信息和频率信息，从而增强模型在多种场景下的表现能力。
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20241112版本更新说明

213. ultralytics/cfg/models/11/yolo11-C3k2-MSM.yaml

    使用[Revitalizing Convolutional Network for Image Restoration TPAMI2024](https://zhuanlan.zhihu.com/p/720777160)中的MSM改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20241112版本更新说明

214. ultralytics/cfg/models/11/yolo11-C3k2-MutilScaleEdgeInformationSelect.yaml

    基于自研CSP-MutilScaleEdgeInformationEnhance再次创新.
    我们提出了一个 多尺度边缘信息选择模块（MutilScaleEdgeInformationSelect），其目的是从多尺度边缘信息中高效选择与目标任务高度相关的关键特征。为了实现这一目标，我们引入了一个具有通过聚焦更重要的区域能力的注意力机制[ICCV2023 DualDomainSelectionMechanism, DSM](https://github.com/c-yn/FocalNet)。该机制通过聚焦图像中更重要的区域（如复杂边缘和高频信号区域），在多尺度特征中自适应地筛选具有更高任务相关性的特征，从而显著提升了特征选择的精准度和整体模型性能。
    项目视频百度云链接:20241124版本更新说明

215. ultralytics/cfg/models/11/yolo11-C3k2-HDRAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的RAB( residual attention block)改进C3k2.
    项目视频百度云链接:20241124版本更新说明

216. ultralytics/cfg/models/11/yolo11-C3k2-RAB.yaml

    使用[Pattern Recognition 2024|DRANet](https://github.com/WenCongWu/DRANet)中的HDRAB(hybrid dilated residual attention block)改进C3k2.
    项目视频百度云链接:20241124版本更新说明

217. ultralytics/cfg/models/11/yolo11-C3k2-LFE.yaml

    使用[Efficient Long-Range Attention Network for Image Super-resolution ECCV2022](https://github.com/xindongzhang/ELAN)中的Local feature extraction改进C3k2.
    项目视频百度云链接:20241124版本更新说明

218. ultralytics/cfg/models/11/yolo11-C2BRA.yaml

    使用[BIFormer CVPR2023](https://github.com/rayleizhu/BiFormer)中的Bi-Level Routing Attention改进C2PSA.
    项目视频百度云链接:20241124版本更新说明

219. ultralytics/cfg/models/11/yolo11-C2CGA.yaml

    使用[EfficientViT CVPR2023](https://github.com/microsoft/Cream/tree/main/EfficientViT)中的CascadedGroupAttention改进C2PSA.
    项目视频百度云链接:20241124版本更新说明

220. ultralytics/cfg/models/11/yolo11-C2DA.yaml

    使用[Vision Transformer with Deformable Attention(CVPR2022)](https://github.com/LeapLabTHU/DAT)中的DAttention改进C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20241124版本更新说明

221. ultralytics/cfg/models/11/yolo11-C2DPB.yaml

    使用[CrossFormer](https://arxiv.org/pdf/2108.00154)中的DynamicPosBias-Attention改进C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250319版本更新说明

222. ultralytics/cfg/models/11/yolo11-C3k2-SFA.yaml

    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)的Frequency-aware Cascade Attention-SFA改进C3k2.
    项目视频百度云链接:20241207版本更新说明

223. ultralytics/cfg/models/11/yolo11-C3k2-CTA.yaml

    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)的Frequency-aware Cascade Attention-CTA改进C3k2.
    项目视频百度云链接:20241207版本更新说明

224. GlobalEdgeInformationTransfer

    实现版本1：ultralytics/cfg/models/11/yolo11-GlobalEdgeInformationTransfer1.yaml
    实现版本2：ultralytics/cfg/models/11/yolo11-GlobalEdgeInformationTransfer2.yaml
    实现版本3：ultralytics/cfg/models/11/yolo11-GlobalEdgeInformationTransfer3.yaml
    总所周知，物体框的定位非常之依赖物体的边缘信息，但是对于常规的目标检测网络来说，没有任何组件能提高网络对物体边缘信息的关注度，我们需要开发一个能让边缘信息融合到各个尺度所提取的特征中，因此我们提出一个名为GlobalEdgeInformationTransfer(GEIT)的模块，其可以帮助我们把浅层特征中提取到的边缘信息传递到整个backbone上，并与不同尺度的特征进行融合。
    1. 由于原始图像中含有大量背景信息，因此从原始图像上直接提取边缘信息传递到整个backbone上会给网络的学习带来噪声，而且浅层的卷积层会帮助我们过滤不必要的背景信息，因此我们选择在网络的浅层开发一个名为MutilScaleEdgeInfoGenetator的模块，其会利用网络的浅层特征层去生成多个尺度的边缘信息特征图并投放到主干的各个尺度中进行融合。
    2. 对于下采样方面的选择，我们需要较为谨慎，我们的目标是保留并增强边缘信息，同时进行下采样，选择MaxPool 会更合适。它能够保留局部区域的最强特征，更好地体现边缘信息。因为 AvgPool 更适用于需要平滑或均匀化特征的场景，但在保留细节和边缘信息方面的表现不如 MaxPool。
    3. 对于融合部分，ConvEdgeFusion巧妙地结合边缘信息和普通卷积特征，提出了一种新的跨通道特征融合方式。首先，使用conv_channel_fusion进行边缘信息与普通卷积特征的跨通道融合，帮助模型更好地整合不同来源的特征。然后采用conv_3x3_feature_extract进一步提取融合后的特征，以增强模型对局部细节的捕捉能力。最后通过conv_1x1调整输出特征维度。
    项目视频百度云链接:20241207版本更新说明

225. ultralytics/cfg/models/11/yolo11-C3k2-IDWC.yaml

    使用[InceptionNeXt CVPR2024](https://github.com/sail-sg/inceptionnext)中的InceptionDWConv2d改进C3k2.
    项目视频百度云链接:20241207版本更新说明

226. ultralytics/cfg/models/11/yolo11-C3k2-IDWB.yaml

    使用[InceptionNeXt CVPR2024](https://github.com/sail-sg/inceptionnext)中的InceptionDWBlock改进C3k2.
    项目视频百度云链接:20241207版本更新说明

227. ultralytics/cfg/models/11/yolo11-inceptionnext.yaml

    使用[InceptionNeXt CVPR2024](https://github.com/sail-sg/inceptionnext)替换backbone.
    项目视频百度云链接:20241207版本更新说明

228. ultralytics/cfg/models/11/yolo11-C3k2-PConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv改进C3k2.
    项目视频百度云链接:20241221版本更新说明

229. ultralytics/cfg/models/11/yolo11-atthead.yaml

    B站注意力教程例子.链接:https://www.bilibili.com/video/BV1mXkVYAEGM/

230. ultralytics/cfg/models/11/yolo11-C3k2-EMA.yaml

    B站注意力教程例子.链接:https://www.bilibili.com/video/BV1mXkVYAEGM/

231. ultralytics/cfg/models/11/yolo11-C3k2-CAMixer.yaml

    使用[CAMixerSR CVPR2024](https://github.com/icandle/CAMixerSR)中的CAMixer改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20241221版本更新说明

232. ultralytics/cfg/models/11/yolo11-MAN.yaml

    使用[Hyper-YOLO TPAMI2025](https://www.arxiv.org/pdf/2408.04804)中的Mixed Aggregation Network改进yolov11.
    项目视频百度云链接:20241221版本更新说明

233. ultralytics/cfg/models/11/yolo11-hyper.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的Hypergraph Computation in Semantic Space改进yolov11.
    项目视频百度云链接:20241221版本更新说明

234. Hyper-YOLO(TPAMI2025)

    1. ultralytics/cfg/models/hyper-yolo/hyper-yolo.yaml
    2. ultralytics/cfg/models/hyper-yolo/hyper-yolot.yaml
    3. ultralytics/cfg/models/hyper-yolo/hyper-yolo-seg.yaml
    项目视频百度云链接:20241221版本更新说明

235. ultralytics/cfg/models/11/yolo11-MAN-Faster.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block进行二次创新改进yolo11.
    项目视频百度云链接:20241228版本更新说明

236. ultralytics/cfg/models/11/yolo11-MAN-FasterCGLU.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新改进yolo11.
    项目视频百度云链接:20241228版本更新说明

237. ultralytics/cfg/models/11/yolo11-MAN-Star.yaml

    使用[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[StarNet CVPR2024](https://github.com/ma-xu/Rewrite-the-Stars/tree/main)中的StarBlock进行二次创新改进yolo11.
    项目视频百度云链接:20241228版本更新说明

238. ultralytics/cfg/models/11/yolo11-msga.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate改进yolo11-neck.
    项目视频百度云链接:20241228版本更新说明

239. ultralytics/cfg/models/11/yolo11-MutilBackbone-MSGA.yaml

    使用[MSA^2 Net](https://github.com/xmindflow/MSA-2Net)中的Multi-Scale Adaptive Spatial Attention Gate对自研系列MutilBackbone再次创新.
    项目视频百度云链接:20241228版本更新说明

240. ultralytics/cfg/models/11/yolo11-C3k2-HFERB.yaml

    使用[ICCV2023 CRAFT-SR](https://github.com/AVC2-UESTC/CRAFT-SR)中的high-frequency enhancement residual block改进C3k2.
    项目视频百度云链接:20250112版本更新说明

241. ultralytics/cfg/models/11/yolo11-C3k2-DTAB.yaml

    使用[AAAI2025 TBSN](https://github.com/nagejacob/TBSN)中的DTAB改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250112版本更新说明

242. ultralytics/cfg/models/11/yolo11-DTAB.yaml

    使用[AAAI2025 TBSN](https://github.com/nagejacob/TBSN)中的DTAB替换C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250112版本更新说明

243. ultralytics/cfg/models/11/yolo11-C3k2-JDPM.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的joint domain perception module改进C3k2.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250112版本更新说明

244. ultralytics/cfg/models/11/yolo11-C3k2-ETB.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的entanglement transformer block改进C3k2.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250112版本更新说明

245. ultralytics/cfg/models/11/yolo11-ETB.yaml

    使用[ECCV2024 FSEL](https://github.com/CSYSI/FSEL)中的entanglement transformer block替换C2PSA.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250112版本更新说明

246. ultralytics/cfg/models/11/yolo11-C3k2-FDT.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Full-domain Transformer改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250112版本更新说明

247. ultralytics/cfg/models/11/yolo11-FDT.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Full-domain Transformer替换C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250112版本更新说明

248. ultralytics/cfg/models/11/yolo11-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade改进yolo11-neck.
    项目视频百度云链接:20250112版本更新说明

249. ultralytics/cfg/models/11/yolo11-slimneck-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade改进slimneck.
    项目视频百度云链接:20250112版本更新说明

250. ultralytics/cfg/models/11/yolo11-MAN-FasterCGLU-WFU.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的Wavelet Feature Upgrade和[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的 Mixed Aggregation Network和[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的Faster-Block和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU进行二次创新改进yolo11.
    项目视频百度云链接:20250112版本更新说明

251. ultralytics/cfg/models/11/yolo11-C3k2-AP.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Asymmetric Padding bottleneck改进yolo11.
    项目视频百度云链接:20250119版本更新说明

252. ultralytics/cfg/models/11/yolo11-PSConv.yaml

    使用[AAAI2025 Pinwheel-shaped Convolution and Scale-based Dynamic Loss for Infrared Small Target Detection](https://github.com/JN-Yang/PConv-SDloss-Data)中的Pinwheel-shaped Convolution改进yolo11.
    项目视频百度云链接:20250119版本更新说明

253. ultralytics/cfg/models/11/yolo11-CDFA.yaml

    使用[ACMMM2024 WFEN](https://github.com/PRIS-CV/WFEN)中的WaveletConv与[AAAI2025 ConDSeg](https://github.com/Mengqi-Lei/ConDSeg)的ContrastDrivenFeatureAggregation结合改进yolo11.
    视频链接:https://www.bilibili.com/video/BV1gXwzeZEqp/

254. ultralytics/cfg/models/11/yolo11-C3k2-ELGCA.yaml

    使用[ELGC-Net](https://github.com/techmn/elgcnet)中的ELGCA改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250205版本更新说明

255. ultralytics/cfg/models/11/yolo11-C3k2-ELGCACGLU.yaml

    使用[ELGC-Net](https://github.com/techmn/elgcnet)中的ELGCA和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250205版本更新说明

256. ultralytics/cfg/models/11/yolo11-C3k2-Strip.yaml

    使用[Strip R-CNN](https://arxiv.org/pdf/2501.03775)中的StripBlock改进C3k2.
    项目视频百度云链接:20250205版本更新说明

257. ultralytics/cfg/models/11/yolo11-C3k2-StripCGLU.yaml

    使用[Strip R-CNN](https://arxiv.org/pdf/2501.03775)中的StripBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C3k2.
    项目视频百度云链接:20250205版本更新说明

258. ultralytics/cfg/models/11/yolo11-C2Pola.yaml

    使用[ICLR2025 PolaFormer](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention改进C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250319版本更新说明

259. ultralytics/cfg/models/11/yolo11-fsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention改进yolo11.(需要看[常见错误和解决方案的第五点](#a))
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250205版本更新说明

260. ultralytics/cfg/models/11/yolo11-mpcafsa.yaml

    使用[BIBM2024 Spatial-Frequency Dual Domain Attention Network For Medical Image Segmentation](https://github.com/nkicsl/SF-UNet)的Frequency-Spatial Attention和Multi-scale Progressive Channel Attention改进yolo11-neck.(需要看[常见错误和解决方案的第五点](#a))
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250205版本更新说明

261. ultralytics/cfg/models/11/yolo11-C3k2-Kat.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAT改进C3k2.<需要编译请看百度云视频-20250205版本更新说明>
    项目视频百度云链接:20250205版本更新说明

262. ultralytics/cfg/models/11/yolo11-C3k2-Faster-KAN.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN对(CVPR2023)fasternet中的FastetBlock进行二次创新.<需要编译请看百度云视频-20250205版本更新说明>
    Windows Triton安装参考链接:
    1. https://blog.csdn.net/yyywxk/article/details/144868136
    2. https://github.com/woct0rdho/triton-windows/releases
    项目视频百度云链接:20250205版本更新说明

263. ultralytics/cfg/models/11/yolo11-C3k2-DIMB.yaml

    自研模块DynamicInceptionDWConv2d.
    动态混合卷积网络：自适应深度卷积与高效特征融合
    ----------------------------------------------------
    当前挑战：
    在传统卷积神经网络（CNN）中，卷积核通常是固定且静态的，这种设计在处理复杂的、具有多尺度特征和高变动性的输入时存在一定的局限性。尤其是在以下几个方面：
        1.	卷积核选择的单一性：大多数网络在卷积操作中依赖于固定大小的卷积核，这可能无法适应不同尺度、不同类型特征的需求。尤其在处理具有显著尺度差异的图像时，单一卷积核无法有效捕捉细节和全局信息的平衡。
        2.	信息融合的不足：在深度神经网络中，不同层次的特征图具有不同的语义信息，传统网络往往忽视了这些信息的高效融合，导致了上下层信息的丢失或不充分利用。这会影响模型在复杂任务中的表现，尤其是在目标检测、图像分割等任务中，特征之间的跨层关联至关重要。
        3.	网络灵活性不足：大部分网络的卷积操作缺乏灵活性，不能根据输入的特征自适应地调整卷积方式。这使得模型在面临各种输入变动时，无法快速调整策略，从而影响模型的鲁棒性和适应能力。
        4.	计算效率与参数浪费：虽然深度卷积操作能够提取丰富的特征，但固定的卷积核和冗余的计算开销会导致模型的训练和推理效率降低，特别是在实时处理和大规模数据集时，过多的计算和参数导致了内存浪费和速度瓶颈。
    ----------------------------------------------------
    本文提出的解决方案：
    我们提出的动态混合卷积网络（Dynamic Inception Mixer）模块，旨在解决上述问题，提供一个灵活且高效的卷积架构，具体通过以下方式应对当前挑战：
        1.	动态卷积核选择与适应性调整
    通过引入动态卷积核权重（DKW）机制，我们能够为每个卷积核分配一个自适应的权重，使得网络能够根据输入特征图的不同需求调整卷积核的使用。这种机制打破了传统固定卷积核的限制，使得卷积操作能够在每次前向传播时根据输入的不同特征进行自适应的调整，进而有效捕捉多尺度和多样化的特征信息。
        2.	跨尺度特征融合与信息共享
    DynamicInceptionMixer通过多尺度卷积核的融合，在不同尺度上提取信息并进行灵活组合。这种跨尺度的特征提取方式使得网络能够在多个尺度上捕捉细节信息，而不仅仅依赖于某一个固定大小的卷积核，从而大大提升了模型对复杂场景的适应能力。
        3.	增强层间信息融合
    通过引入DynamicIncMixerBlock模块，我们实现了跨层信息的有效融合。在每个模块中，卷积后的特征图会与输入特征图进行残差加和，保持了网络中低层和高层特征的有效传递。这不仅使得模型能够充分利用多层次的信息，还能减少由于网络层次过深导致的梯度消失或特征丢失问题。
        4.	高效计算与参数共享
    在DynamicInceptionDWConv2d模块中，我们使用了深度卷积（depthwise convolution），这种方法能够显著减少参数量和计算量，同时仍然保持对输入特征图的细致处理。通过对卷积核的动态调整，不仅提升了模型在多样性输入上的表现，还减少了不必要的计算冗余。
        5.	灵活的网络训练与正则化
    引入的Layer Scale机制，通过在每个模块中动态地调整权重缩放，有助于在训练过程中更好地控制模型的收敛性和稳定性。此外，结合DropPath策略，进一步提升了模型的正则化效果，有效防止了过拟合，尤其是在高维特征空间中的表现尤为突出。
    ----------------------------------------------------
    总结：
    通过动态卷积核选择、跨层信息融合、以及高效的计算设计，Dynamic Inception Mixer能够有效解决传统卷积神经网络在多尺度特征提取、信息融合及计算效率方面的不足。我们的设计不仅提升了网络的适应性和灵活性，还在实际任务中展示了更强的鲁棒性和高效的计算性能。因此，该模块为深度学习模型的构建和优化提供了一个全新的思路，特别适用于处理多变和复杂输入数据的任务，如目标检测、图像分割等。
    项目视频百度云链接:20250215版本更新说明

264. ultralytics/cfg/models/11/yolo11-C3k2-DIMB-KAN.yaml

    在263的基础上把mlp模块换成[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN.<KAN 需要编译请看百度云视频-20250205版本更新说明>
    项目视频百度云链接:20250215版本更新说明

265. ultralytics/cfg/models/11/yolo11-C3k2-GlobalFilter.yaml

    使用[T-PAMI Global Filter Networks for Image Classification](https://github.com/raoyongming/GFNet)中的GlobalFilterBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250215版本更新说明

266. ultralytics/cfg/models/11/yolo11-C3k2-DynamicFilter.yaml

    使用[AAAI2024 FFT-Based Dynamic Token Mixer for Vision](https://github.com/okojoalg/dfformer)中的DynamicFilter改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250215版本更新说明

267. ultralytics/cfg/models/11/yolo11-C3k2-TSSA.yaml
     
    使用[ICLR2025 Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention和[Metaformer TPAMI2024](https://github.com/sail-sg/metaformer)的metaformer改进C3k2.
    项目视频百度云链接:20250301版本更新说明

268. ultralytics/cfg/models/11/yolo11-C2TSSA.yaml
     
    使用[ICLR2025 Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention改进C2PSA.
    项目视频百度云链接:20250319版本更新说明

269. ultralytics/cfg/models/12/yolo12-A2C2f-CGLU.yaml
     
    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进A2C2f.
    项目视频百度云链接:20250301版本更新说明

270. ultralytics/cfg/models/12/yolo12-A2C2f-KAN.yaml

    使用[ICLR2025 Kolmogorov-Arnold Transformer](https://github.com/Adamdad/kat)中的KAN改进A2C2f.<KAN 需要编译请看百度云视频-20250205版本更新说明>
    项目视频百度云链接:20250301版本更新说明

271. ultralytics/cfg/models/12/yolo12-A2C2f-DFFN.yaml
    
    使用[FreqFormer](https://github.com/JPWang-CS/FreqFormer)中的DFFN改进A2C2f.
    项目视频百度云链接:20250301版本更新说明

272. ultralytics/cfg/models/11/yolo11-RepHMS.yaml

    使用[MHAF-YOLO](https://github.com/yang-0201/MHAF-YOLO)中的RepHMS改进yolo11.
    项目视频百度云链接:20250301版本更新说明

273. ultralytics/cfg/models/11/yolo11-HAFB-1.yaml
     
    在深度学习领域，尤其是计算机视觉任务中，多尺度特征融合和注意力机制已被广泛证明能够显著提升模型对复杂场景的感知能力。然而，现有方法在融合多源特征时，往往面临局部与全局信息难以有效协调的问题。为此，我们提出了层次注意力融合块（Hierarchical Attention Fusion Block, HAFB），旨在通过层次化的注意力机制实现局部与全局特征的自适应融合，从而提升特征表达的鲁棒性和任务适配性。该模块特别适用于需要高效处理多尺度输入的场景，例如目标检测、图像分割等。

    HAFB 的核心创新在于其结合了局部与全局注意力分支（通过 LocalGlobalAttention 子模块实现），并通过层次化的特征处理路径对输入特征进行精细化融合。这种设计不仅保留了输入特征的空间细节，还通过自适应加权机制增强了全局上下文的语义表达能力。
    1. 特征预处理
        输入特征分别先经过1x1卷积进行降维，这一步骤旨在统一输入特征的维度，并减少后续计算的复杂度。并且通过一个3x3卷积对双输入特征进行初步融合得到基线特征。
    2. 层次注意力分支（LocalGlobalAttention）此部分来源于：[HCFNet](https://github.com/zhengshuchen/HCFNet)
    3. 特征融合与优化
        最后将经过LocalGlobalAttention分支的特征与基线特征，分别经过一个Conv1x1,RepConv3x3,Conv1x1。其中第一个卷积主要是降低输入特征的维度，减少后续计算的复杂度，这里我们利用了可重参数化卷积来做特征重组，能够极大提高参数利用率，并且能够更好地重组来自多个分支的特征。
    作用：
    1. 层次化特征融合：HAFB模块通过结合局部和全局特征的注意力机制，有效提升了特征的多样性和表达能力。相比传统的单一尺度处理，HAFB能够更全面地捕捉图像中的细节和宏观信息。
	2. 高效的注意力机制：通过使用LocalGlobalAttention模块，HAFB能够灵活地处理不同尺度的特征，并自适应地选择关注的区域，提高了模型的计算效率和表现能力。
	3. 增强的跨通道信息传递：通过引入多个注意力机制和卷积操作，HAFB不仅优化了空间特征，还加强了通道之间的信息交互，为后续任务提供了更具判别力的特征表示。
    项目视频百度云链接:20250301版本更新说明

274. ultralytics/cfg/models/11/yolo11-HAFB-2.yaml
     
    HAFB的另外一种使用方式.
    项目视频百度云链接:20250301版本更新说明

275. ultralytics/cfg/models/11/yolo11-MutilBackbone-HAFB.yaml
    
    yolo11-MutilBackbone-DAF.yaml基础上用上HAFB.
    项目视频百度云链接:20250301版本更新说明

276. ultralytics/cfg/models/11/yolo11-C2ASSA.yaml
     
    使用[CVPR2024 Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)中的Adaptive Sparse Self-Attention改进C2PSA.
    (需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250319版本更新说明

277. ultralytics/cfg/models/12/yolo12-A2C2f-FRFN.yaml
     
    使用[CVPR2024 Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf)中的feature refinement feed-forward改进A2C2f.
    项目视频百度云链接:20250312版本更新说明

278. ultralytics/cfg/models/11/yolo11-ASSR.yaml

    使用[CVPR2025 MambaIR](https://github.com/csguoh/MambaIR)中的Attentive State Space Group改进yolo11.(需要看[常见错误和解决方案的第五点](#a))(编译教程请看:20240219版本更新说明)
    项目视频百度云链接:20250312版本更新说明

279. ultralytics/cfg/models/11/yolo11-C3k2-SAVSS.yaml

    使用[CVPR2025 SCSegamba](https://github.com/Karl1109/SCSegamba)中的Structure-Aware Scanning Strategy改进C3k2.(编译教程请看:20240219版本更新说明)
    项目视频百度云链接:20250312版本更新说明

280. ultralytics/cfg/models/11/yolo11-C3k2-MobileMamba.yaml
     
    使用[CVPR2025 MobileMamba](https://github.com/lewandofskee/MobileMamba)中的MobileMambaBlock改进C3k2.(编译教程请看20240619版本更新说明)
    项目视频百度云链接:20250312版本更新说明

281. ultralytics/cfg/models/11/yolo11-MobileMamba.yaml
     
    使用[CVPR2025 MobileMamba](https://github.com/lewandofskee/MobileMamba)中的MobileMamba改进Backbone.(编译教程请看20240619版本更新说明)
    项目视频百度云链接:20250312版本更新说明

282. ultralytics/cfg/models/11/yolo11-mambaout.yaml
     
    使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOut替换BackBone.
    项目视频百度云链接:20250312版本更新说明

283. ultralytics/cfg/models/11/yolo11-C3k2-MambaOut.yaml
     
    使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock改进C3k2.
    项目视频百度云链接:20250312版本更新说明

284. ultralytics/cfg/models/11/yolo11-C2PSA-DYT.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTanh改进C2PSA.
    项目视频百度云链接:20250319版本更新说明

285. ultralytics/cfg/models/11/yolo11-C2TSSA-DYT.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan和[ICLR2025 Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention改进C2PSA.
    项目视频百度云链接:20250319版本更新说明

286. ultralytics/cfg/models/11/yolo11-C2Pola-DYT.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan和[ICLR2025 PolaFormer](https://github.com/ZacharyMeng/PolaFormer)中的PolaAttention改进C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250319版本更新说明

287. ultralytics/cfg/models/12/yolo12-A2C2f-DYT.yaml
     
    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTanh改进A2C2f.
    项目视频百度云链接:20250319版本更新说明

288. ultralytics/cfg/models/12/yolo12-A2C2f-CGLU-DYT.yaml
     
    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTanh和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进A2C2f.
    项目视频百度云链接:20250319版本更新说明

289. ultralytics/cfg/models/12/yolo12-A2C2f-DFFN-DYT.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTanh和[FreqFormer](https://github.com/JPWang-CS/FreqFormer)中的DFFN改进A2C2f.
    项目视频百度云链接:20250319版本更新说明

290. ultralytics/cfg/models/11/yolo11-C3k2-MambaOut-UniRepLK.yaml

    使用[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock和[CVPR2024 UniRepLKNet](https://github.com/AILab-CVC/UniRepLKNet/tree/main)中的DilatedReparamBlock二次创新后改进C3k2.
    项目视频百度云链接:20250329版本更新说明

291. ultralytics/cfg/models/11/yolo11-C3k2-EfficientVIM.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock改进C3k2.
    项目视频百度云链接:20250329版本更新说明

292. ultralytics/cfg/models/11/yolo11-C3k2-EfficientVIM-CGLU.yaml

    使用[CVPR2025 EfficientViM](https://github.com/mlvlab/EfficientViM)中的EfficientViMBlock和[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C3k2.
    项目视频百度云链接:20250329版本更新说明

293. Localization Quality Estimation Head

    此模块出自[GFocalV2](https://arxiv.org/abs/2011.12885).
    detect:ultralytics/cfg/models/11/yolo11-LQEHead.yaml
    segmet:ultralytics/cfg/models/11/yolo11-seg-LQE.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-LQE.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-LQE.yaml
    视频https://www.bilibili.com/video/BV1BZZVY9EK3

294. Localization Quality Estimation - Lightweight Shared Convolutional Detection Head

    Localization Quality Estimation模块出自[GFocalV2](https://arxiv.org/abs/2011.12885).
    detect:ultralytics/cfg/models/11/yolo11-LSCD-LQE.yaml
    seg:ultralytics/cfg/models/11/yolo11-seg-LSCD-LQE.yaml
    pose:ultralytics/cfg/models/11/yolo11-pose-LSCD-LQE.yaml
    obb:ultralytics/cfg/models/11/yolo11-obb-LSCD-LQE.yaml
    视频:https://www.bilibili.com/video/BV1BZZVY9EK3

295. ultralytics/cfg/models/11/yolo11-EUCB.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB改进yolo11的上采样.
    项目视频百度云链接:20250329版本更新说明

296. ultralytics/cfg/models/11/yolo11-EUCB-SC.yaml

    使用[CVPR2024 EMCAD](https://github.com/SLDGroup/EMCAD)中的EUCB和[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix改进yolo11的上采样.
    项目视频百度云链接:20250329版本更新说明

297. ultralytics/cfg/models/11/yolo11-EMBSFPN-SC.yaml

    在ultralytics/cfg/models/11/yolo11-EMBSFPN.yaml方案上引入[CVPR2025 BHViT](https://github.com/IMRL/BHViT)中的ShiftChannelMix.
    项目视频百度云链接:20250329版本更新说明

298. ultralytics/cfg/models/11/yolo11-C3k2-FAT.yaml

    使用[ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC)中的FATBlock改进C3k2.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250415版本更新说明

299. ultralytics/cfg/models/11/yolo11-C2PSA-FMFFN.yaml

    使用[ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC)中的FMFFN改进C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250415版本更新说明

300. ultralytics/cfg/models/11/yolo11-C2PSA-CGLU.yaml

    使用[TransNeXt CVPR2024](https://github.com/DaiShiResearch/TransNeXt)中的Convolutional GLU改进C2PSA.
    项目视频百度云链接:20250415版本更新说明

301. ultralytics/cfg/models/12/yolo12-A2C2f-FMFFN.yaml

    使用[ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC)中的FMFFN改进A2C2f.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250415版本更新说明

302. ultralytics/cfg/models/12/yolo12-A2C2f-FMFFN-DYT.yaml

    使用[ICLR2024-FTIC](https://github.com/qingshi9974/ICLR2024-FTIC)中的FMFFN和[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan对A2C2f二次创新.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250415版本更新说明

303. ultralytics/cfg/models/11/yolo11-C3k2-LEGM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的LEGM改进C3k2.
    项目视频百度云链接:20250415版本更新说明

304. ultralytics/cfg/models/11/yolo11-MFM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM改进neck.
    项目视频百度云链接:20250415版本更新说明

305. ultralytics/cfg/models/11/yolo11-MFMMAFPN.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对[MAF-YOLO](https://arxiv.org/pdf/2407.04381)的MAFPN进行二次创新.
    项目视频百度云链接:20250415版本更新说明

306. ultralytics/cfg/models/11/yolo11-MBSMFFPN.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对yolo11-EMBSFPN.yaml再次创新 Multi-Branch&Scale Modulation-Fusion FPN.
    项目视频百度云链接:20250415版本更新说明

307. ultralytics/cfg/models/11/yolo11-overlock.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的overlock-backbone替换backbone.<安装教程请看本期视频>
    项目视频百度云链接:20250415版本更新说明

308. ultralytics/cfg/models/11/yolo11-C3k2-RCB.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的RepConvBlock改进C3k2.
    项目视频百度云链接:20250415版本更新说明

309. ultralytics/cfg/models/11/yolo11-GDSAFusion.yaml

    使用[CVPR2025 OverLock](https://arxiv.org/pdf/2502.20087)中的GDSAFusion改进neck.<安装教程请看本期视频>
    项目视频百度云链接:20250415版本更新说明

310. ultralytics/cfg/models/11/yolo11-hyper-MFM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对[Hyper-YOLO](https://www.arxiv.org/pdf/2408.04804)中的Hypergraph Computation in Semantic Space进行二次创新.
    项目视频百度云链接:20250415版本更新说明

311. ultralytics/cfg/models/11/yolo11-C3k2-LFEM.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LFEModule改进C3k2.
    项目视频百度云链接:20250502版本更新说明

312. ultralytics/cfg/models/11/yolo11-LoGStem.yaml

    使用[LEGNet](https://github.com/lwCVer/LEGNet)中的LoGStem改进Stem(第一第二层卷积).
    项目视频百度云链接:20250502版本更新说明

313. ultralytics/cfg/models/11/yolo11-C3k2-SBSM.yaml

    使用[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Snake Bi-Directional Sequence Modelling (SBSM)改进C3k2.(编译教程请看20240619版本更新说明)
    项目视频百度云链接:20250502版本更新说明

314. ultralytics/cfg/models/11/yolo11-C2PSA-SEFN.yaml

    使用[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)改进C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250502版本更新说明

315. ultralytics/cfg/models/12/yolo12-A2C2f-SEFN.yaml

    使用[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)改进A2C2f.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250502版本更新说明

316. ultralytics/cfg/models/12/yolo12-A2C2f-Mona.yaml

    使用[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进A2C2f.
    项目视频百度云链接:20250502版本更新说明

317. ultralytics/cfg/models/11/yolo11-C2PSA-Mona.yaml

    使用[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进C2PSA.
    项目视频百度云链接:20250502版本更新说明

318. ultralytics/cfg/models/11/yolo11-lsnet.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSNet替换yolo11-backbone.(需要安装triton==3.2.0这个包，linux安装比较简单，windows可以参考使用说明.md环境部分的链接)(需要看[常见错误和解决方案的第五点](#a))  
    项目视频百度云链接:20250502版本更新说明

318. ultralytics/cfg/models/11/yolo11-C3k2-LSBlock.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSBlock改进C3k2.(需要安装triton==3.2.0这个包，linux安装比较简单，windows可以参考使用说明.md环境部分的链接)
    项目视频百度云链接:20250502版本更新说明

319. ultralytics/cfg/models/11/yolo11-C3k2-MambaOut-LSConv.yaml

    使用[CVPR2025 LSNet](https://github.com/THU-MIG/lsnet)的LSConv与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C3k2.(需要安装triton==3.2.0这个包，linux安装比较简单，windows可以参考使用说明.md环境部分的链接)
    项目视频百度云链接:20250502版本更新说明

320. ultralytics/cfg/models/12/yolo12-A2C2f-DFFN-DYT-Mona.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTanh和[FreqFormer](https://github.com/JPWang-CS/FreqFormer)中的DFFN和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进A2C2f.
    项目视频百度云链接:20250502版本更新说明

321. ultralytics/cfg/models/11/yolo11-C2TSSA-DYT-Mona.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan和[ICLR2025 Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona改进C2PSA.
    项目视频百度云链接:20250502版本更新说明

322. ultralytics/cfg/models/11/yolo11-C2TSSA-DYT-Mona-SEFN.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan和[ICLR2025 Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona和[WACV2025 SEM-Net](https://github.com/ChrisChen1023/SEM-Net)的Spatially-Enhanced Feedforward Network (SEFN)改进C2PSA.(需要看[常见错误和解决方案的第五点](#a))
    项目视频百度云链接:20250502版本更新说明

323. ultralytics/cfg/models/11/yolo11-C2PSA-SEFFN.yaml

    使用[TransMamba](https://github.com/sunshangquan/TransMamba)的SpectralEnhancedFFN改进C2PSA.
    项目视频百度云链接:20250518版本更新说明

324. ultralytics/cfg/models/11/yolo11-C2TSSA-DYT-Mona-SEFFN.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan和[ICLR2025 Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona和[TransMamba](https://github.com/sunshangquan/TransMamba)的SpectralEnhancedFFN改进C2PSA.
    项目视频百度云链接:20250518版本更新说明

325. ultralytics/cfg/models/11/yolo11-C3k2-TransMamba.yaml

    使用[TransMamba](https://github.com/sunshangquan/TransMamba)的TransMamba改进C3k2.(编译教程请看:20240219版本更新说明)
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250518版本更新说明

326. ultralytics/cfg/models/12/yolo12-A2C2f-SEFFN.yaml

    使用[TransMamba](https://github.com/sunshangquan/TransMamba)的SpectralEnhancedFFN改进A2C2f.
    项目视频百度云链接:20250518版本更新说明

327. ultralytics/cfg/models/11/yolo11-C3k2-EVS.yaml

    使用[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EVS改进C3k2.(编译教程请看:20240219版本更新说明)
    项目视频百度云链接:20250518版本更新说明

328. ultralytics/cfg/models/11/yolo11-C2PSA-EDFFN.yaml

    使用[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EDFFN改进C2PSA.
    项目视频百度云链接:20250518版本更新说明

329. ultralytics/cfg/models/11/yolo11-C2TSSA-DYT-Mona-EDFFN.yaml

    使用[CVPR2025 DyT](https://github.com/jiachenzhu/DyT)中的DynamicTan和[ICLR2025 Token Statistics Transformer](https://github.com/RobinWu218/ToST)中的Token Statistics Self-Attention和[CVPR2025 Mona](https://github.com/Leiyi-Hu/mona)的Mona和[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EDFFN改进C2PSA.
    项目视频百度云链接:20250518版本更新说明

330. ultralytics/cfg/models/12/yolo12-A2C2f-EDFFN.yaml

    使用[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EDFFN改进A2C2f.
    项目视频百度云链接:20250518版本更新说明

331. ultralytics/cfg/models/11/yolo11-C3k2-EBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的EBlock改进C3k2.
    项目视频百度云链接:20250518版本更新说明

332. ultralytics/cfg/models/11/yolo11-C3k2-DBlock.yaml

    使用[CVPR2025 DarkIR](https://github.com/cidautai/DarkIR)中的DBlock改进C3k2.
    项目视频百度云链接:20250518版本更新说明

333. ultralytics/cfg/models/11/yolo11-C3k2-FDConv.yaml

    使用[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv改进C3k2.
    项目视频百度云链接:20250601版本更新说明

334. ultralytics/cfg/models/11/yolo11-C3k2-MambaOut-FDConv.yaml

    使用[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C3k2.
    项目视频百度云链接:20250601版本更新说明

335. ultralytics/cfg/models/11/yolo11-C3k2-PFDConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv与[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv二次创新后改进C3k2.
    项目视频百度云链接:20250601版本更新说明

336. ultralytics/cfg/models/11/yolo11-C3k2-FasterFD.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的FasterBlock与[CVPR2025 Frequency Dynamic Convolution for Dense Image Prediction](https://github.com/Linwei-Chen/FDConv)的FDConv二次创新后改进C3k2.
    项目视频百度云链接:20250601版本更新说明

337. ultralytics/cfg/models/11/yolo11-C3k2-DSAN.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention Block改进C3k2.(编译教程请看:20250601版本更新说明)
    项目视频百度云链接:20250601版本更新说明

338. ultralytics/cfg/models/11/yolo11-C3k2-MambaOut-DSA.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention Block与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C3k2.(编译教程请看:20250601版本更新说明)
    项目视频百度云链接:20250601版本更新说明

339. ultralytics/cfg/models/11/yolo11-C3k2-DSA.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention改进C3k2.(编译教程请看:20250601版本更新说明)
    项目视频百度云链接:20250601版本更新说明

340. ultralytics/cfg/models/11/yolo11-C3k2-DSAN-EDFFN.yaml

    使用[DSA: Deformable Spatial Attention](https://www.techrxiv.org/users/628671/articles/775010-deformable-spatial-attention-networks-enhancing-lightweight-convolutional-models-for-vision-tasks)中的Deformable Spatial Attention Block和[CVPR2025 EVSSM](https://github.com/kkkls/EVSSM)中的EDFFN进行二次创新后改进C3k2.(编译教程请看:20250601版本更新说明)
    项目视频百度云链接:20250601版本更新说明

341. ultralytics/cfg/models/11/yolo11-C3k2-RMB.yaml

    使用[CVPR2025 MaIR](https://github.com/XLearning-SCU/2025-CVPR-MaIR)中的Residual Mamba Block改进C3k2.(需要看[常见错误和解决方案的第五点](#a))(编译教程请看:20240219版本更新说明)
    项目视频百度云链接:20250601版本更新说明

342. ultralytics/cfg/models/11/yolo11-RFPN.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE改进YOLO11-neck.
    项目视频百度云链接:20250612版本更新说明

343. ultralytics/cfg/models/11/yolo11-SOEP-RFPN.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE对原创改进SOEP再次创新.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250612版本更新说明

344. ultralytics/cfg/models/11/yolo11-SOEP-MFM.yaml

    使用[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对原创改进SOEP再次创新.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250612版本更新说明

345. ultralytics/cfg/models/11/yolo11-SOEP-RFPN-MFM.yaml

    使用[ECCV2024 rethinking-fpn](https://github.com/AlanLi1997/rethinking-fpn)的SNI和GSConvE和[CVPR2024 DCMPNet](https://github.com/zhoushen1/DCMPNet)中的MFM对原创改进SOEP再次创新.
    该模块需要在train.py中关闭amp、且在ultralytics/engine/validator.py 115行附近的self.args.half设置为False、跑其余改进记得修改回去！
    项目视频百度云链接:20250612版本更新说明

346. ultralytics/cfg/models/11/yolo11-C3k2-SFSConv.yaml

    使用[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv改进C3k2.
    项目视频百度云链接:20250612版本更新说明

347. ultralytics/cfg/models/11/yolo11-C3k2-MambaOut-SFSC.yaml

    使用[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv与[CVPR2025 MambaOut](https://github.com/yuweihao/MambaOut)中的MambaOutBlock二次创新后改进C3k2.
    项目视频百度云链接:20250612版本更新说明

348. ultralytics/cfg/models/11/yolo11-C3k2-PSFSConv.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的PConv与[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv二次创新后改进C3k2.
    项目视频百度云链接:20250612版本更新说明

349. ultralytics/cfg/models/11/yolo11-C3k2-FasterSFSC.yaml

    使用[FasterNet CVPR2023](https://github.com/JierunChen/FasterNet)中的FasterBlock与[CVPR2024 SFSConv](https://github.com/like413/SFS-Conv)的SFSConv二次创新后改进C3k2.
    项目视频百度云链接:20250612版本更新说明

350. ultralytics/cfg/models/11/yolo11-FCM.yaml

    使用[AAAI2025 FBRT-YOLO](https://github.com/galaxy-oss/FCM)的模块改进yolo11.
    项目视频百度云链接:20250612版本更新说明

351. ultralytics/cfg/models/12/yolo12-FCM.yaml

    使用[AAAI2025 FBRT-YOLO](https://github.com/galaxy-oss/FCM)的模块改进yolo12.
    项目视频百度云链接:20250612版本更新说明

352. ultralytics/cfg/models/11/yolo11-C3k2-GroupMamba.yaml

    使用[CVPR2025 GroupMamba](https://github.com/Amshaker/GroupMamba)中的GroupMambaLayer改进C3k2.(编译教程请看:20240219版本更新说明)
    项目视频百度云链接:20250612版本更新说明

353. ultralytics/cfg/models/11/yolo11-C3k2-GroupMambaBlock.yaml

    使用[CVPR2025 GroupMamba](https://github.com/Amshaker/GroupMamba)中的GroupMambaBlock改进C3k2.(编译教程请看:20240219版本更新说明)
    项目视频百度云链接:20250612版本更新说明

354. ultralytics/cfg/models/11/yolo11-C3k2-MambaVision.yaml

    使用[CVPR2025 MambaVision](https://github.com/NVlabs/MambaVision)中的MambaVision改进C3k2.(编译教程请看:20240219版本更新说明)
    项目视频百度云链接:20250612版本更新说明

# 常见错误和解决方案(如果是跑自带的一些配置文件报错可以先看看对应的配置文件是否有提示需要修改内容)
1. RuntimeError: xxxxxxxxxxx does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'.....

    解决方案：在ultralytics/utils/torch_utils.py中init_seeds函数中把torch.use_deterministic_algorithms里面的True改为False

2. ModuleNotFoundError：No module named xxx

    解决方案：缺少对应的包，先把YOLO11环境配置的安装命令进行安装一下，如果还是缺少显示缺少包，安装对应的包即可(xxx就是对应的包).

3. OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.  

    解决方案：https://zhuanlan.zhihu.com/p/599835290

4. 训练过程中loss出现nan.

    可以尝试关闭AMP混合精度训练.(train.py中加amp=False)

<a id="a"></a>

5. 固定640x640尺寸的解决方案.

    运行train.py中的时候需要在ultralytics/models/yolo/detect/train.py的DetectionTrainer class中的build_dataset函数中的rect=mode == 'val'改为rect=False.其他模型可以修改回去.  
    运行val.py的时候,把val.py的rect=False注释取消即可.其他模型可以修改回去.  
    运行detect.py中的时候需要在ultralytics/engine/predictor.py找到函数def pre_transform(self, im),在LetterBox中的auto改为False,其他模型可以修改回去.  

6. 多卡训练问题.[参考链接](https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training/#multi-gpu-dataparallel-mode-not-recommended:~:text=just%201%20GPU.-,Multi%2DGPU%20DistributedDataParallel%20Mode%20(%E2%9C%85%20recommended),-You%20will%20have)

    假设我的机器有四张卡
    我想用第二第三张卡来跑，则命令：CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --nproc_per_node 2 train.py
    我想用第一第二第三张卡来跑，则命令：CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run --nproc_per_node 3 train.py
    我想用全部卡跑，则命令：CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 train.py
    百度云视频链接：https://pan.baidu.com/s/1q7Yap6WSflAUtMIvssFn0w?pwd=cy8r

7. 指定显卡训练.

    百度云视频链接：https://pan.baidu.com/s/1q7Yap6WSflAUtMIvssFn0w?pwd=cy8r

8. ValueError: Expected more than 1 value per channel when training, got input size torch.Size...

    请注意看控制台输出信息，如果是训练阶段的最后一个batch出现这个报错，证明最后一个batch的大小为1，可以去掉训练集的一张图或者加一张图避免出现这个情况，又或者更改batch。
    同样地，如果在训练时候的验证阶段出现这个情况，跟上述处理一样，可以去掉验证集的一张图或者加一张图避免出现这个情况，又或者更改batch。

9. AttributeError: Can't pickle local object 'EMASlideLoss.__init__.<locals>.<lambda>'

    可以在ultralytics/utils/loss.py中添加import dill as pickle,然后装一下dill这个包.  
    pip install dill -i https://pypi.tuna.tsinghua.edu.cn/simple

10. RuntimeError: Dataset 'xxxxx' error ❌

    将data.yaml中的路径都改为绝对路径.

11. WARNING  NMS time limit 2.100s exceeded

    在ultralytics/utils/ops.py中non_max_suppression函数里面找到这个语句：
        time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    前面的2.0自己改大点即可，大到不会出现这个NMS time limit即可.

12. OSError: [WinError 1455] 页面文件太小，无法完成操作。

    此问题常见于windows训练.一般情况下有两种解决方案:
    1. 把workers设置小点直接不会报错.最小为0
    2. 扩大虚拟内存(可百度).

13. train.py运行的精度正常，但是val.py全是0.

    选用fp32后缀的模型去计算就可以，这是因为部分模块对fp16很敏感，所以导致用fp16格式的权重去验证的时候会出现精度全为0的情况。