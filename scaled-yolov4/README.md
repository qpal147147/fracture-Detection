# fracture-Detection
此文件使用**Scaled-YOLOv4**

環境、訓練、測試、檢測方式請參考[ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)

## 數據集
 使用yolo格式，如以下形式:  
 ```<object-class> <x_center> <y_center> <width> <height>```

 詳細可參考[YOLOv5自定義數據教學](https://docs.ultralytics.com/tutorials/train-custom-datasets/)  
 **未包含物件的影像可當作背景影像訓練，無須標籤**
 
 ### 數據結構
 
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
 
## 參考
* https://github.com/WongKinYiu/ScaledYOLOv4
* https://github.com/ultralytics/yolov5
