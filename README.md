
Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a
[**Python>=3.7.0**](https://www.python.org/) environment, including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/).


```
```bash
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
```

# requirements
```bash
conda env create -f yolo.yaml -y
```
# source env 
```bash
conda activate yolo 
```

`detect.py` runs inference on a variety of sources, downloading [models](https://github.com/ultralytics/yolov5/tree/master/models) automatically from
the latest YOLOv5 [release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.

```bash
python detect.py --source 0  # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
                          'path/*.jpg'  # glob
                          'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                          'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
```

</details>

<details>
<summary>Training</summary>

The commands below reproduce YOLOv5 [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)
results. [Models](https://github.com/ultralytics/yolov5/tree/master/models)
and [datasets](https://github.com/ultralytics/yolov5/tree/master/data) download automatically from the latest
YOLOv5 [release](https://github.com/ultralytics/yolov5/releases). Training times for YOLOv5n/s/m/l/x are
1/2/4/6/8 days on a V100 GPU ([Multi-GPU](https://github.com/ultralytics/yolov5/issues/475) times faster). Use the
largest `--batch-size` possible, or pass `--batch-size -1` for
YOLOv5 [AutoBatch](https://github.com/ultralytics/yolov5/pull/5092). Batch sizes shown for V100-16GB.

```bash
python train.py --data coco.yaml --cfg yolov5n.yaml --weights '' --batch-size 128
                                       yolov5s                                64
                                       yolov5m                                40
                                       yolov5l                                24
                                       yolov5x                                16
```

### Train
YOLOv5 classification training supports auto-download of MNIST, Fashion-MNIST, CIFAR10, CIFAR100, Imagenette, Imagewoof, and ImageNet datasets with the `--data` argument. To start training on MNIST for example use `--data mnist`.
```bash
python train.py --weights models/weights/xxxx.pt --batch-size 16 --project landSlideByPoly/TrainYolov5n --data data/xxx.yaml
```
#### 一般要使用weights.pt 文件进行预训练，如果已经训练完毕的文件也可以继续训练， 不过要在训练时添加“-resume”才能保证下一次能恢复训练；下一次想要继续上一次训练结果继续训练时，将--weights 中.pt文件改成最last.pt即可
#### data中存放训练的数据集的yaml文件，里面要包含训练集、测试集和标签 ，验证集可以不需要，专门有验证的.py文件，如果需要测试，再改一下地址就行。

```bash
# Single-GPU
python classify/train.py --model yolov5s-cls.pt --data cifar100 --epochs 5 --img 224 --batch 128

# Multi-GPU DDP
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 classify/train.py --model yolov5s-cls.pt --data imagenet --epochs 5 --img 224 --device 0,1,2,3
```

### Val
Validate YOLOv5m-cls accuracy on ImageNet-1k dataset:
```bash
bash data/scripts/get_imagenet.sh --val  # download ImageNet val split (6.3G, 50000 images)
python classify/val.py --weights yolov5m-cls.pt --data ../datasets/imagenet --img 224  # validate
```

### Predict
Use pretrained YOLOv5s-cls.pt to predict bus.jpg:
```bash
python classify/predict.py --weights yolov5s-cls.pt --data data/images/bus.jpg
```
```python
model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5s-cls.pt')  # load from PyTorch Hub
```

### Export
Export a group of trained YOLOv5s-cls, ResNet and EfficientNet models to ONNX and TensorRT:
```bash
python export.py --weights yolov5s-cls.pt resnet50.pt efficientnet_b0.pt --include onnx engine --img 224
```



