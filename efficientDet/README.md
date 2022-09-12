<div align="center">

中文 | [English](https://github.com/qpal147147/fracture-Detection/blob/main/efficientDet/README_en.md)
</div>

# fracture-Detection

此文件使用**EfficientDet**

## 環境

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

## 自訂數據集

  使用csv格式
  
  ```text
  path/to/img.bmp, x1, y1, x2, y2, class_name
  ```

  其中`x1`與`y1`為左上角座標，`x2`與`y2`為右下角座標
  
  如果影像沒有標記物件，格式為:
  
  ```text
  path/to/img.bmp,,,,,
  ```

  如果你有yolo格式的label檔案，可使用[yolo2csv.py](https://github.com/qpal147147/fracture-Detection/blob/main/util/yolo2csv.py)進行轉換:  

  ```python
  python yolo2csv.py --images datasets/images --labels datasets/labels
  ```
  
  轉換後輸出`annotations.csv`與`class.csv`

### 範例

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

  如果要修改class，請更改[預設類別](https://github.com/qpal147147/fracture-Detection/blob/main/util/yolo2csv.py#L10)

## 訓練

* Step1:

  ```python
  python train.py --snapshot imagenet --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --random-transform --compute-val-loss --freeze-backbone --batch-size 16 --steps 1000 csv train.csv classes.csv --val-annotations val.csv
  ```

  其中`--phi`為指定[EfficientNet](https://arxiv.org/abs/1905.11946)型號(B0~B6)
  
* Step2: 當驗證分數無法繼續上升時

  ```python
  python train.py --snapshot checkpoints/xxx.h5 --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 --random-transform --compute-val-loss --freeze-bn --batch-size 4 --steps 1000 csv train.csv classes.csv --val-annotations val.csv
  ```

## 評估

```python
cd eval
python common.py --snapshot checkpoints/xxx.h5 --phi {0, 1, 2, 3, 4, 5, 6} --gpu 0 csv test.csv classes.csv
```

## 參考

* <https://github.com/xuannianz/EfficientDet>
