<div align="center">

中文 | [English](https://github.com/qpal147147/fracture-Detection/blob/main/README_en.md)
</div>

# fracture-Detection

 Detection fresh and old fracture on spine CT image using YOLOR  
 使用[YOLOR](https://github.com/WongKinYiu/yolor)檢測脊椎CT影像上的新舊骨折  

## 範例

![Example](https://github.com/qpal147147/fracture-Detection/blob/main/example/example.png)

## 其他模型

* 物件檢測
  * [使用EfficientDet檢測 (Keras)](https://github.com/qpal147147/fracture-Detection/tree/main/efficientDet)
  * [使用RetinaNet檢測 (Pytorch)](https://github.com/qpal147147/fracture-Detection/tree/main/retinaNet)
  * [使用YOLOv4檢測 (Pytorch)](https://github.com/qpal147147/fracture-Detection/tree/main/yolov4)
  * [使用Scaled-YOLOv4檢測 (Pytorch)](https://github.com/qpal147147/fracture-Detection/tree/main/scaled-yolov4)

* 分割
  * [分割模型 (Keras)](https://github.com/qpal147147/fracture-Detection/tree/main/segmentation)

## 方法

<img src="https://github.com/qpal147147/fracture-Detection/blob/main/example/System%20flow.png" height="300">
骨質疏鬆症所引起的脊椎壓迫性骨折是造成老年人疼痛與失能的原因之一，及早發現並採取治療是非常重要的事情。在MRI影像上儘管能有效的診斷出症狀，但需要較高的診斷成本，相較於MRI，CT影像診斷成本比MRI低，且在許多疾病上的診斷也相當良好，但在脊椎骨折上檢測效果卻不如MRI準確。綜合上述，為了能夠使診斷加速及把握治療的黃金期，我們提出了基於YOLOR物件檢測方法，對脊椎CT影像定位骨折區域並判別新舊骨折。實驗結果顯示，在基礎的YOLOR物件檢測方法上，獲得了92.6%的準確率，依照YOLOR架構進行Backbone的替換，將原架構中的CSPDarknet53替換為MobileViT以及EfficientNet_NS，訓練出三種不同Backbone的YOLOR模型，用以提升提取特徵的能力，分別獲得了89%、89.8%及89.2%的準確度。在此基礎上，替換卷積層為Involution層以及結合模型集成方式，集成改進的三種網路，準確率能夠提升至93.4%。

### 實驗比較

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

## 環境

* Python >= 3.7
* Pytorch >= 1.7.0

``` shell
pip install -r requirements.txt
```

## 數據準備

### DICOM to BMP

``` python
python utils/preprocess.py dicom2img datasets/dicoms datasets/bmps --HU
```

### 標記骨折

可使用[LabelImg](https://github.com/heartexlabs/labelImg)進行標記，使用YOLO格式，`.txt`格式如下:  
`<object-class> <x_center> <y_center> <width> <height>`  
詳細可參考[YOLOv5自定義數據教學](https://docs.ultralytics.com/tutorials/train-custom-datasets/)  
 **未包含物件的影像可當作背景影像訓練，無須標籤**

### 數據結構

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

## 訓練

* 原始YOLOR

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

### 追蹤訓練情形

使用`wandb`

```python
pip install wandb
```

安裝完畢後，訓練指令如上述一樣，但需要在終端輸入API Key儲存到你的帳號底下，Key可從<https://wandb.ai/authorize>獲得  
更多詳細資訊請參考[Weights & Biases with YOLOv5](https://github.com/ultralytics/yolov5/issues/1289)

## 評估

* 單模型

``` python
python test.py --data data/spine.yaml --weights yolor_p6.pt --batch 16 --img-size 640 --task test --device 0
```

* 模型集成

``` python
python test.py --data data/spine.yaml --weights yolor_p6.pt yolor_mobilevit.pt yolor_efficient_ns.pt --batch 16 --img-size 640 --task test --device 0
```

## 檢測

* 單模型

``` python
# --source 可以從各種來源進行檢測
python detect.py --source datasets/images/fracture.jpg --weights yolor_p6.pt --img-size 640 --device 0 --save-txt
                          0 # webcam
                          img.jpg  # image
                          vid.mp4  # video
                          path/  # directory
```

* 模型集成

``` python
python detect.py --source datasets/images/fracture.jpg --weights yolor_p6.pt yolor_mobilevit.pt yolor_efficient_ns.pt --img-size 640 --device 0 --save-txt
```

## 二階段分類器

使用二階段分類器可以提高模型的準確率，降低模型的誤判率，但會使檢測時間提高  
可在評估及檢測命令後面增加`--classifier`、`--classifier-weights`、`--classifier-size`、`--classifier-thres`

關於分類器的詳細資訊，請參考[這裡](https://github.com/qpal147147/fracture-Detection/tree/main/classifier)

``` python
# for example
# evaluation
python test.py --data data/spine.yaml --weights yolor_p6.pt --batch 16 --img-size 640 --task test --device 0 --classifier --classifier-weights model_best.pth.tar --classifier-size 96 --classifier-thres 0.6

# detect
python detect.py --source datasets/images/fracture.jpg --weights yolor_p6.pt --img-size 640 --device 0 --save-txt --classifier --classifier-weights model_best.pth.tar --classifier-size 96 --classifier-thres 0.6
```

| 參數 | 描述
| ------------- | -------------
| classifier | 是否啟用分類器
| classifier-weights | 分類器權重路徑
| classifier-size | 分類器檢測的影像尺寸
| classifier-thres | 用於改變檢測類別的閥值。當分類機率超過該閥值時，代表分類器高度自信，因此將原始的檢測類別更改為分類的類別

## UI

![UI](https://github.com/qpal147147/fracture-Detection/blob/main/example/ui.gif)

### 啟動UI

```python
python ui.py
```

更多用法請參考[gradio](https://www.gradio.app/)

## 參考

* <https://github.com/WongKinYiu/yolor>
* <https://github.com/ultralytics/yolov5>
* <https://github.com/gradio-app/gradio>
* <https://github.com/ChristophReich1996/Involution>
* <https://github.com/rwightman/pytorch-image-models>
