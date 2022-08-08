# Classifier

## 訓練
### 數據集
    datasets/
        -project_name/
            -train_name/
                -classes_id/
                    -*.jpg
            -val_name/
                -classes_id/
                    -*.jpg


    # for example
    datasets/
        -fracture/
            -train/
                -0/
                    -00001.jpg
                    -00002.jpg
                -1/
                    -00003.jpg
            -val/
                -0/
                    -00004.jpg
                -1/
                    -00005.jpg

### 開始訓練
```python
python train.py ./datasets/fracture --model tf_efficientnet_b1_ns --pretrained --num-classes 2 --img-size 96 --batch-size 128 --opt AdamP --epochs 300
```
更多**模型**選擇請查看[文檔](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv)  
更多**影像增強**介紹請參閱[教學](https://timm.fast.ai/training)

## 驗證
```python
python validate.py ./datasets/fracture --model tf_efficientnet_b1_ns --num-classes 2 --img-size 96 --batch-size 256 --checkpoint ./output/train/model_best.pth.tar
```

## 推論
```python
python inference.py path/to/data --model tf_efficientnet_b1_ns --num-classes 2 --img-size 96 --checkpoint ./output/train/model_best.pth.tar
```

## 參考
* https://github.com/rwightman/pytorch-image-models
