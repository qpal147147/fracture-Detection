<div align="center">

[中文](https://github.com/qpal147147/fracture-Detection/blob/main/classifier/README.md) | English
</div>

# Classifier

## Data structure

```text
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
```

## Training

```python
python train.py ./datasets/fracture --model tf_efficientnet_b1_ns --pretrained --num-classes 2 --img-size 96 --batch-size 128 --opt AdamP --epochs 300
```

Please see [documentation](https://github.com/rwightman/pytorch-image-models/blob/master/results/results-imagenet.csv) for more **models**.

Please refer to the [tutorial]((https://timm.fast.ai/training)) for more **image augment**.

## Evaluation

```python
python validate.py ./datasets/fracture --model tf_efficientnet_b1_ns --num-classes 2 --img-size 96 --batch-size 256 --checkpoint ./output/train/model_best.pth.tar
```

## Inference

```python
python inference.py path/to/data --model tf_efficientnet_b1_ns --num-classes 2 --img-size 96 --checkpoint ./output/train/model_best.pth.tar
```

## Reference

* <https://github.com/rwightman/pytorch-image-models>
