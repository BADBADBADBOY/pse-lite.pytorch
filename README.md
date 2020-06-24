# pse-lite-pytorch


***

#### data format
follow icdar15 dataset format, x1,y1,x2,y2,x3,y3,x4,y4,label
```
image
│   1.jpg
│   2.jpg   
│		...
label
│   gt_1.txt
│   gt_2.txt
|		...
```

### Compression model mode one,use lite basemodel

***
Support switching basemodel,(mobilenet,squeezenet,shufflenet,resnet)

#### train 


```
python3 train.py --backbone mobile 
```

#### test

```
python3 inference.py
```

***


### Compression model mode two,Channel clipping

#### Sparse training

```
python3 train.py --backbone resnet --sr_lr 0.00001
```

#### prune model

```
python3 prune.py 
```

#### fintune

```
python3 train_prune_finetune.py 
```

#### prune test

```
python3 inference_prune.py 
```
#### performance
|Method| precision|     recall  |   hmean|prune ratio|modelsize（M）|infer time(v100)(ms)|
| :-------- | --------:| :------: |
| PSENet-1s (ResNet50)|   |  ||0|114.5|
| PSENet-1s (ResNet50)|0.8179|   0.7958|  0.8067|0.8|25.1|
| PSENet-1s (ResNet50)|0.8124|   0.7862|  0.7991|0.9|16.6|7
***
### Compression model mode three, Model distillation



# reference

 1. https://github.com/whai362/PSENet
 2. https://github.com/xiaolai-sqlai/mobilenetv3
 3. https://github.com/MhLiao/DB
 4. https://github.com/tanluren/yolov3-channel-and-layer-pruning

