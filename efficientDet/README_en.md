<div align="center">

[中文](https://github.com/qpal147147/fracture-Detection/blob/main/efficientDet/README.md) | English
</div>

# fracture-Detection

This document uses **EfficientDet**

## Environment

* python=3.6.13
* tensorflow=1.14.0
* keras=2.2.4
* keras-applications=1.0.8
* keras-preprocessing=1.1.2
* opencv-python=4.5.3.56
* pillow=8.3.2
* tqdm=4.62.3
* progressbar2

  ```bash
  pip install Cython
  python setup.py build_ext --inplace
  ```

## Custom dataset

  Use CSV format:
  
  ```text
  path/to/img.bmp, x1, y1, x2, y2, class_name
  ```

  Where `x1` and `y1` are the coordinates of the upper left corner, and `x2` and `y2` are the coordinates of the lower right corner.
  
  If the image without objects, the format is:
  
  ```text
  path/to/img.bmp,,,,,
  ```

  If you have label file of YOLO format, you can use [yolo2csv.py](https://github.com/qpal147147/fracture-Detection/blob/main/util/yolo2csv.py) to convert to csv format:

  ```python
  python yolo2csv.py --images datasets/images --labels datasets/labels
  ```
  
  Output `annotations.csv` and `class.csv` after convert

### Example

* annotations.csv

  ```text
  000001.jpg,128,47,173,82,old
  000001.jpg,347,431,363,455,fresh
  000002.jpg,48,13,79,34,fresh
  000003.jpg,,,,,
  ```

* class.csv

  ```text
  fresh,0
  old,1
  ```

If you need to change the classes, edit the [default classes](https://github.com/qpal147147/fracture-Detection/blob/main/util/yolo2csv.py#L10)

## Training

* Step1:

  ```python
  python train.py --snapshot imagenet --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --random-transform --compute-val-loss --freeze-backbone --batch-size 16 --steps 1000 csv train.csv classes.csv --val-annotations val.csv
  ```

  Where `--phi` is the [EfficientNet](https://arxiv.org/abs/1905.11946) model number(B0~B6)
  
* Step2: When validation scores can't keep going up

  ```python
  python train.py --snapshot checkpoints/xxx.h5 --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --random-transform --compute-val-loss --freeze-bn --batch-size 4 --steps 1000 csv train.csv classes.csv --val-annotations val.csv
  ```

## Evaluation

```python
cd eval
python common.py --snapshot checkpoints/xxx.h5 --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 csv test.csv classes.csv
```

## Reference

* <https://github.com/xuannianz/EfficientDet>
