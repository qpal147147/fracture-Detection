<div align="center">

[中文](https://github.com/qpal147147/fracture-Detection/blob/main/segmentation/README.md) | English
</div>

# Fracture segmentation

This document uses **Segmentation model**

## Example

![segmentation](https://github.com/qpal147147/fracture-Detection/blob/main/segmentation/segmentation.png)  

## Environment

* python=3.8.11
* tensorflow=2.4.0
* keras-applications=1.0.8
* keras-preprocessing=1.1.2
* opencv-python=4.5.4.6
* numpy=1.19.5
* numba=0.41.0
* Classification models

```python
pip install image-classifiers==0.2.2
```

## Available models (Not sorted by year)

* CBAM_ASPP_ResUNet
* CPF-Net
* CE-Net
* PSPNet
* ResUNet++
* UNet3+
* VNet

## Dataset

**Notice**: The number of images and annotations must be the same

```text
datasets/
    -project_name/
        -train_name/
            -image_name/
                -image/
                    -*.bmp(Formats supported by tensorflow)
            -mask_name/
                -mask/
                    -*.bmp(Formats supported by tensorflow)

# for example
datasets/
    -fracture/
        -train/
            -image_set/
                -image/
                    -00001.bmp
                    -00002.bmp
                    -00003.bmp
            -mask_set/
                -mask/
                    -00001.bmp
                    -00002.bmp
                    -00003.bmp
```

## Training

If your images path is :
`C:\datasets\fracture\train\image_set\image\00001.bmp`, the parameter of [trainX_dir、trainY_dir](https://github.com/qpal147147/fracture-Detection/blob/main/segmentation/train.py#L52) is `C:\datasets\fracture\train\image_set`

```python
python train.py
```

## Evaluation

```python
python ./eval/main.py -s path/to/seg -g path/to/gt
```

## Testing

```python
python inference.py
```

## Reference

* <https://github.com/qubvel/classification_models>
* <https://github.com/mlyg/unified-focal-loss>
* <https://github.com/kochlisGit/Unet3-Plus>
* <https://github.com/FENGShuanglang/CPFNet_Project>
* <https://github.com/FENGShuanglang/2D-Vnet-Keras>
* <https://github.com/billymoonxd/ResUNet>
* <https://github.com/billymoonxd/ResUNet>
* <https://www.kaggle.com/code/momincks/medical-image-segmentation-with-ce-net/notebook>
* <https://github.com/FabioXimenes/SegmentationMetrics>
