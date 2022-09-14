<div align="center">

[中文](https://github.com/qpal147147/fracture-Detection/blob/main/retinaNet/README.md) | English
</div>

# fracture-Detection

This document uses **RetinaNet**

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

If you have label file of YOLO format, you can use [yolo2csv.py](https://github.com/qpal147147/fracture-Detection/blob/main/utils/yolo2csv.py) to convert to csv format:

```python
python yolo2csv.py --images datasets/images --labels datasets/labels
```

Output `annotations.csv` and `class.csv` after convert

### Example

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

If you need to change the classes, edit the [default classes](https://github.com/qpal147147/fracture-Detection/blob/main/util/yolo2csv.py#L10)

## Training

```python
python train.py --dataset csv --csv_train train_annots.csv  --csv_classes class.csv  --csv_val val_annots.csv --batch-size 16 --epochs 100
```

## Evaluation

```python
python csv_validation.py --csv_annotations_path test_annots.csv --class_list_path class.csv --model_path weights/model.pt
```

## Visualization

```python
python visualize_single_image.py --image_dir path/to/images --model_path weights/model.pt --class_list class.csv
```

## Reference

* <https://github.com/yhenon/pytorch-retinanet>
