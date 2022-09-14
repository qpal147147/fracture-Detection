<div align="center">

中文 | [English](https://github.com/qpal147147/fracture-Detection/blob/main/retinaNet/README_en.md)
</div>

# fracture-Detection

此文件使用**RetinaNet**

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

如果你有yolo格式的label檔案，可使用[yolo2csv.py](https://github.com/qpal147147/fracture-Detection/blob/main/utils/yolo2csv.py)進行轉換:  

```python
python yolo2csv.py --images datasets/images --labels datasets/labels
```

轉換後輸出`annotations.csv`與`class.csv`

### 範例

* annotations.csv

    ``` text
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

```python
python train.py --dataset csv --csv_train train_annots.csv  --csv_classes class.csv  --csv_val val_annots.csv --batch-size 16 --epochs 100
```

## 評估

```python
python csv_validation.py --csv_annotations_path test_annots.csv --class_list_path class.csv --model_path weights/model.pt
```

## 可視化

```python
python visualize_single_image.py --image_dir path/to/images --model_path weights/model.pt --class_list class.csv
```

## 參考

* <https://github.com/yhenon/pytorch-retinanet>
