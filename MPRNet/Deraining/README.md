
## Training
- Download the [Datasets](https://pan.baidu.com/s/1Xd_gVcgHftzirD6aQlwimQ) code: nlkb

- Train the model with default arguments by running

```
cd MPRNet\Deraining
python train.py
```


## Evaluation

1. The pretrained model has download in `./pretrained_models/`

2. I have download Test100 dataset in [Datasets](https://pan.baidu.com/s/1Xd_gVcgHftzirD6aQlwimQ) code: nlkb
other dataset (Rain100H, Rain100L, Test1200, Test2800) you can download from [here](https://drive.google.com/drive/folders/1PDWggNh8ylevFmrjo-JEvlmqsDlWWvZs?usp=sharing) and place them in `./Datasets/test/`

3. Run
```
python test.py
```

#### To reproduce PSNR/SSIM scores of the paper, run
```
evaluate_PSNR_SSIM.m 
```
