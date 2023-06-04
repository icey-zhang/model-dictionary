# model-dictionary
The dictionary for us to conveniently record some codes

1.去噪声的可逆网络MPRNet

2.用来在yolov5上画热图的代码grad-cam

代码原址：[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

使用并且修改代码：[EigenCAM%20for%20YOLO5.ipynb](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/EigenCAM%20for%20YOLO5.ipynb)


3.多模态Multimodal Prompting with Missing Modalities for Visual Recognition

代码原址：[missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts)

需要下载的一个权重文件：[vilt_200k_mlm_itm.ckpt](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)

数据集：[Food101]

一些bug修改：
（1）[_sync_params](https://blog.csdn.net/qq_33854260/article/details/129037203)
（2）from pytorch_lightning.metrics import Metric修改成from torchmetrics import Metric，因为pytorch_lightning.metrics这个库被删了

4.多模态融合的可信度分析Uncertainty-Aware Multi-View Representation Learning

代码原址：[daunet](http://cic.tju.edu.cn/faculty/zhangchangqing/research.html)

5.生成高低频图像的代码HFC

代码原址：[HFC](https://github.com/HaohanWang/HFC)

6.多模态融合Trusted Multi-View Classification with Dynamic Eidential Fusion

代码原址：[ETMC](https://github.com/hanmenghan/TMC)

数据集下载： [NYUD2](https://drive.google.com/file/d/1M-EvhVfQ0HXEpTrDcqVrNK6C8CHPP0Yo/view?usp=sharing)

