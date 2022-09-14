<div align="center">

[中文](https://github.com/qpal147147/fracture-Detection/blob/main/yolov4/README.md) | English
</div>

# fracture-Detection

This document uses **YOLOv4**

Environment, training, testing and detection methods refer to [PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)

## Dataset

Use YOLO format :  
`<object-class> <x_center> <y_center> <width> <height>`

Please see [Train Custom Data tutorial of YOLOv5](https://docs.ultralytics.com/tutorials/train-custom-datasets/) for more details.  
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

## training setting

Change the filter in the cfg file based on your number of categories :  
`filters = (classes + 5) * 3`  

If your number of categories is 2, the filter is `(2 + 5) * 3 = 21`. You can refer [here](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)  
<https://github.com/qpal147147/fracture-Detection/blob/main/yolov4/cfg/yolov4.cfg#L961>  
<https://github.com/qpal147147/fracture-Detection/blob/main/yolov4/cfg/yolov4.cfg#L1048>  
<https://github.com/qpal147147/fracture-Detection/blob/main/yolov4/cfg/yolov4.cfg#L1135>

## Reference

* <https://github.com/WongKinYiu/PyTorch_YOLOv4>
* <https://github.com/ultralytics/yolov5>
