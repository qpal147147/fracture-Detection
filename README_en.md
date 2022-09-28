<div align="center">

[中文](https://github.com/qpal147147/fracture-Detection/blob/main/README.md) | English
</div>

# fracture-Detection

 Detection fresh and old fracture on spine CT image using [YOLOR](https://github.com/WongKinYiu/yolor)

## Example

![Example](https://github.com/qpal147147/fracture-Detection/blob/main/example/example.png)

## Other models

* Object detection
  * [EfficientDet (Keras)](https://github.com/qpal147147/fracture-Detection/tree/main/efficientDet)
  * [RetinaNet (Pytorch)](https://github.com/qpal147147/fracture-Detection/tree/main/retinaNet)
  * [YOLOv4 (Pytorch)](https://github.com/qpal147147/fracture-Detection/tree/main/yolov4)
  * [Scaled-YOLOv4 (Pytorch)](https://github.com/qpal147147/fracture-Detection/tree/main/scaled-yolov4)

* Segmentation
  * [Segmentation (Keras)](https://github.com/qpal147147/fracture-Detection/tree/main/segmentation)

## Method

<img src="https://github.com/qpal147147/fracture-Detection/blob/main/example/System%20flow.png" height="300">

> Vertebral compression fractures caused by osteoporosis are one of the reasons of pain and disability in the elderly. The early detection and treatment are very important. Although MRI can effectively diagnose symptoms, it requires a higher diagnostic cost. CT image has a lower diagnostic cost compared to MRI. However, CT detecting vertebral fractures is not as accurate as MRI. In order to speed up diagnosis and grasp the golden period of treatment, we proposed a YOLO-based object detection method to localize old and fresh fractures on spine CT images. We replaced the backbone of CSPDarknet53 in the native YOLOR model with MobileViT and EfficientNet_NS. Three YOLOR models with different backbone are trained separately, and finally the three YOLOR models are ensemble to improve the ability of feature extraction. Experimental results show that the accuracy of the three YOLOR models are 89%, 89.8%, and 89.2%, respectively. On this basis, these three improved networks are replaced convolution layer by Involution layer and integrated by the model ensemble method, the accuracy rate is increased to 93.4%. The proposed method achieves the purpose of being fast and accurate, and provides good advice and reference for physicians.

### Experiment result

| Model | Backbone | Precision | Recall | AP<sub>fresh</sub> | AP<sub>old</sub> | mAP@0.5
| :--: | :--: | :--: | :--: | :--: | :--: | :--:
| EfficientDet | EfficientNetB0 | 17.5% | 96.2% | 87.7% | 83% | 85.4%
| RetinaNet | ResNet50 | 38.2% | 94% | 90.1% | 82.3% | 86.2%
| YOLOv4 | CSPDarknet53 | 53.4% | 92.2% | 92% | 84.6% | 88.4%
| Scaled-YOLOv4 | CSPDarknet53 | 62% | 90.4% | 93.2% | 84.6% | 88.9%
| | | | | | |
| YOLOR | CSPDarknet53 | 65% | 91.1% | 92.6% | 85.4% | 89%
| YOLOR | MobileViT | 60.6% | 92.1% | 92.9% | 86.7% | 89.8%
| YOLOR | EfficientNet_NS | 69.1% | 89.9% | 92.9% | 85.6% | 89.2%
| | | | | | |
| YOLOR | CSPDarknet53<sub>invo</sub> | 71.3% | 91.8% | 93.7% | 88.2% | 90.9%
| YOLOR | MobileViT<sub>invo</sub> | 61.8% | 92.2% | 93.1% | 87.5% | 90.3%
| YOLOR | EfficientNet_NS<sub>invo</sub> | 65.6% | 91.1% | 92.7% | 86.3% | 89.5%
| | | | | | |
| YOLOR | Ensemble | 63.4% | 95.1% | 95.4% | 91.5% | 93.4%

## Environment

* Python >= 3.7
* Pytorch >= 1.7.0

```bash
pip install -r requirements.txt
```

## Dataset

### DICOM to BMP

``` python
python utils/preprocess.py dicom2img datasets/dicoms datasets/bmps --HU
```

### Labeling

You can use [LabelImg](https://github.com/heartexlabs/labelImg) label your data, it uses YOLO format with `.txt` extension:  
`<object-class> <x_center> <y_center> <width> <height>`  
Please see [Train Custom Data tutorial of YOLOv5](https://docs.ultralytics.com/tutorials/train-custom-datasets/) for more details  
**Images without objects can be used as background images for training and it doesn't require labels.**

### Data structure

```text
datasets/
    -project_name/
        -images/
            -train/
                -*.bmp(or others format)
            -val/
                -*.bmp(or others format)
            -test/
                -*.bmp(or others format)
        -labels/
            -train/
                -*.txt
            -val/
                -*.txt
            -test/
                -*.txt
# for example
datasets/
    -fracture/
        -images/
            -train/
                -00001.bmp
                -00002.bmp
                -00003.bmp
            -val/
                -00004.bmp
            -test/
                -00005.bmp
        -labels/
            -train/
                -00001.txt
                -00002.txt
            -val/
                -00004.txt
            -test/
                -00005.txt
```

## Training

* Original YOLOR (CSPDarknet53)
  
``` python
python train.py --data data/spine.yaml --cfg models/spine_yolor-p6.yaml --img-size 640 --weights yolor_p6.pt --device 0 --batch 32 --cache --epochs 300 --name yolor_p6
```

* MobileViT

``` python
python train.py --data data/spine.yaml --cfg models/spine_yolor-mobileViT.yaml --img-size 640 --weights yolor_p6.pt --device 0 --batch 32 --cache --epochs 300 --name yolor_mobilevit
```

* Efficient_NS

``` python
python train.py --data data/spine.yaml --cfg models/spine_yolor-efficientB2ns.yaml --img-size 640 --weights yolor_p6.pt --device 0 --batch 32 --cache --epochs 300 --name yolor_efficient_ns
```

### Track training

use `wandb`

```python
pip install wandb
```

After the installation is complete, the training command is the same as above, but you need to enter the API Key in the terminal. Key can be obtained from <https://wandb.ai/authorize>.  
Please see [Weights & Biases with YOLOv5](https://github.com/ultralytics/yolov5/issues/1289) for more details.

## Evaluation

* Single model

``` python
python test.py --data data/spine.yaml --weights yolor_p6.pt --batch 16 --img-size 640 --task test --device 0
```

* Model ensembl

``` python
python test.py --data data/spine.yaml --weights yolor_p6.pt yolor_mobilevit.pt yolor_efficient_ns.pt --batch 16 --img-size 640 --task test --device 0
```

## Detection

* Single model

``` python
# --source can detect from different sources
python detect.py --source datasets/images/fracture.jpg --weights yolor_p6.pt --img-size 640 --device 0 --save-txt
                          0 # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
```

* Model ensemble

``` python
python detect.py --source datasets/images/fracture.jpg --weights yolor_p6.pt yolor_mobilevit.pt yolor_efficient_ns.pt --img-size 640 --device 0 --save-txt
```

## Second stage classifier

Using a second stage classifier improves model accuracy and reduces false positives, but increases detection time  
You can add `--classifier`、`--classifier-weights`、`--classifier-size`、`--classifier-thres` after the evaluation and detection commands.

More information on classifiers can be found [here](https://github.com/qpal147147/fracture-Detection/tree/main/classifier)

``` python
# for example
# evaluation
python test.py --data data/spine.yaml --weights yolor_p6.pt --batch 16 --img-size 640 --task test --device 0 --classifier --classifier-weights model_best.pth.tar --classifier-size 96 --classifier-thres 0.6

# detect
python detect.py --source datasets/images/fracture.jpg --weights yolor_p6.pt --img-size 640 --device 0 --save-txt --classifier --classifier-weights model_best.pth.tar --classifier-size 96 --classifier-thres 0.6
```

| Command | Description
| ------------- | -------------
| classifier | Enable classifier
| classifier-weights | Classifier weight path
| classifier-size | Input image size of classifier
| classifier-thres | Change the threshold for detection classes. When the classification probability exceeds this threshold, it means that the classifier is highly confident, so the original detection class is changed to the classification class.

## UI

![UI](https://github.com/qpal147147/fracture-Detection/blob/main/example/ui.gif)

### Run UI

```python
python ui.py
```

For more usage, please refer to [gradio](https://www.gradio.app/)

## Reference

* <https://github.com/WongKinYiu/yolor>
* <https://github.com/ultralytics/yolov5>
* <https://github.com/gradio-app/gradio>
* <https://github.com/ChristophReich1996/Involution>
* <https://github.com/rwightman/pytorch-image-models>
