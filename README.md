# model-dictionary
The dictionary for us to conveniently record some codes

1.去噪声的可逆网络MPRNet

2.用来在yolov5上画热图的代码grad-cam

3.多模态Multimodal Prompting with Missing Modalities for Visual Recognition
代码原址：[missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts)
一些bug修改：
（1）[_sync_params](https://blog.csdn.net/qq_33854260/article/details/129037203)
（2）from pytorch_lightning.metrics import Metric修改成from torchmetrics import Metric，因为pytorch_lightning.metrics这个库被删了

