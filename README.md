# fracture-Detection
 此儲存庫使用**分割模型**
 
## 範例
 ![segmentation](https://github.com/qpal147147/fracture-Detection/blob/segmentation/segmentation.png)  
 
## 環境
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

## 可用模型(未按照年分排序)
* CBAM_ASPP_ResUNet
* CPF-Net
* CE-Net
* PSPNet
* ResUNet++
* UNet3+
* VNet

## 訓練
 ### 數據集
 **注意**: 影像與註釋的數量必須一樣
 
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
                  
   ### 開始訓練
   如果你的影像路徑為如下所示:
   ```C:\datasets\fracture\train\image_set\image\00001.bmp```  
   則train.py的[trainX_dir、trainY_dir](https://github.com/qpal147147/fracture-Detection/blob/segmentation/train.py#L52)參數為:
   ```C:\datasets\fracture\train\image_set```
   
   * 訓練
   ```python
   python train.py
   ```

## 評估
 ```python
 python ./eval/main.py -s path/to/seg -g path/to/gt
 ```
## 測試
 ```python
 python inference.py
 ```
 
## 參考
* <https://github.com/qubvel/classification_models>
* <https://github.com/mlyg/unified-focal-loss>
* <https://github.com/kochlisGit/Unet3-Plus>
* <https://github.com/FENGShuanglang/CPFNet_Project>
* <https://github.com/FENGShuanglang/2D-Vnet-Keras>
* <https://github.com/billymoonxd/ResUNet>
* <https://github.com/billymoonxd/ResUNet>
* <https://www.kaggle.com/code/momincks/medical-image-segmentation-with-ce-net/notebook>
* <https://github.com/FabioXimenes/SegmentationMetrics>
