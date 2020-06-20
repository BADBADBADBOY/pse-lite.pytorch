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
#### test

python3 inference.py

Support switching basemodel,(mobilenet,squeezenet,shufflenet,resnet)

#### train 


```
python3 train.py --backbone mobile 
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

# reference

 1. https://github.com/whai362/PSENet
 2. https://github.com/xiaolai-sqlai/mobilenetv3
 3. https://github.com/MhLiao/DB
 4. https://github.com/tanluren/yolov3-channel-and-layer-pruning

