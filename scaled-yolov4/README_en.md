<div align="center">

[中文](https://github.com/qpal147147/fracture-Detection/blob/main/scaled-yolov4/README.md) | English
</div>

# fracture-Detection

This document uses **Scaled-YOLOv4**

Environment, training, testing and detection methods refer to [ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)

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

## Reference

* <https://github.com/WongKinYiu/ScaledYOLOv4>
* <https://github.com/ultralytics/yolov5>
