# model-dictionary
The dictionary for us to conveniently record some codes

## 1.去噪声的可逆网络MPRNet

## 2.用来在yolov5上画热图的代码grad-cam

代码原址：[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

使用并且修改代码：[EigenCAM%20for%20YOLO5.ipynb](https://github.com/jacobgil/pytorch-grad-cam/blob/master/tutorials/EigenCAM%20for%20YOLO5.ipynb)


## 3.多模态Multimodal Prompting with Missing Modalities for Visual Recognition

代码原址：[missing_aware_prompts](https://github.com/YiLunLee/missing_aware_prompts)

需要下载的一个权重文件：[vilt_200k_mlm_itm.ckpt](https://github.com/dandelin/ViLT/releases/download/200k/vilt_200k_mlm_itm.ckpt)

数据集：[Food101]

一些bug修改：
（1）[_sync_params](https://blog.csdn.net/qq_33854260/article/details/129037203)
（2）from pytorch_lightning.metrics import Metric修改成from torchmetrics import Metric，因为pytorch_lightning.metrics这个库被删了

## 4.多模态融合的可信度分析Uncertainty-Aware Multi-View Representation Learning

代码原址：[daunet](http://cic.tju.edu.cn/faculty/zhangchangqing/research.html)

## 5.生成高低频图像的代码HFC

代码原址：[HFC](https://github.com/HaohanWang/HFC)

## 6.多模态融合Trusted Multi-View Classification with Dynamic Eidential Fusion

代码原址：[ETMC](https://github.com/hanmenghan/TMC)

数据集下载： [NYUD2](https://drive.google.com/file/d/1M-EvhVfQ0HXEpTrDcqVrNK6C8CHPP0Yo/view?usp=sharing)

修改数据集路径即可

## 7.Bridging Cross-task Protocol Inconsistency for Distillation in Dense Object Detection

代码原址：[BCKD](https://github.com/TinyTigerPan/BCKD)

数据集下载： [coco](https://blog.csdn.net/qq_44554428/article/details/122597358)

安装环境、下载数据集、下载权重文件、运行测试

## 8.QWen_VL | [Code](https://github.com/icey-zhang/model-dictionary/tree/main/Qwen)

[【下载模型】](https://huggingface.co/Qwen/Qwen-VL/tree/main) 放到model路径下

修改路径

做目标检测任务

## 9.miniGPT4 | [Code](https://github.com/icey-zhang/miniGPT4_guide)

[【使用教程】](https://github.com/icey-zhang/miniGPT4_guide)

## 10. YOLOWORD
### 问题一
```bash
You should set `PYTHONPATH` to make `sys.path` include the directory which contains your custom module
```
在simple_demo.py文件最开始添加project的路径
```python
import sys
sys.path.append('/home/zjq/YOLO-World')
```
### 问题二
```bash
OSError: Incorrect path_or_model_id: '../pretrained_models/clip-vit-base-patch32-projection'. Please provide either the path to a local folder or the repo_id of a model on the Hub.
```
#### 1. 修改config文件里面
change ../pretrained_models/clip-vit-base-patch32-projection into openai/clip-vit-base-patch32 (不行)

#### 2. 直接通过镜像下载权重 (不行)
```bash
git clone https://hf-mirror.com/openai/clip-vit-base-patch32 
```
#### 3. 直接下载放目标目录下

### 问题三
下载文件
https://huggingface.co/GLIPModel/GLIP/blob/main/lvis_v1_minival_inserted_image_name.json

### 问题四
创建txt文件,文件内容
```bash
person
bus
```

运行代码 image_demo.py

