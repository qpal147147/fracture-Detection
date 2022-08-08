import os
import glob

import cv2
import numpy as np
from models import unet3_plus, cpfnet, pspnet, vnet, cbam_aspp_resUnet, resUnet_plus_plus, cenet

class DataLoader:
    def __init__(self, test_dir):
        self.size = 512
        self.ch = 3
        self.test_dir = test_dir

    def preprocess(self, img):
        img = cv2.resize(img, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.
        return img

    def load_data(self):
        test_list = glob.glob(self.test_dir + "/*.*")
        assert len(test_list)!=0, "The Number of image can't be 0 !"
        print(f"Test Number: {len(test_list)}")

        test = []
        for test_path in test_list:
            img = cv2.imread(test_path)
            img = self.preprocess(img)
            test.append(img)

        return np.asarray(test)

class PredNet:
    def __init__(self, test_dir, model_path):
        self.saveImg = "./predict_img"
        self.saveMask = "./predict_mask"
        self.size = 512
        self.ch = 3
        self.test_dir = test_dir
        self.model_path = model_path
        # -------------------

        os.makedirs(self.saveImg, exist_ok=True)
        os.makedirs(self.saveMask, exist_ok=True)
        # -------------------

        self.test = DataLoader(self.test_dir).load_data()
        self.model = cpfnet()
        self.model.load_weights(self.model_path)
    

    def predict(self):
        file_name = os.listdir(self.test_dir)

        for index, test in enumerate(self.test):
            x = test[np.newaxis, :, :, :]
            pred = self.model.predict(x)
            pred = pred[0] * 255
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2GRAY)

            # 二值化
            pred = pred.astype("uint8")
            _, pred = cv2.threshold(pred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite((self.saveMask + "/" + file_name[index]), pred)
            
            # 預測遮罩上色 - 橘色
            pred_blue = pred / 255 * 13
            pred_green = pred / 255 * 126
            pred_red = pred / 255 * 230
            pred_color = cv2.merge((pred_blue, pred_green, pred_red)).astype("uint8")

            # 疊合
            final_img = cv2.addWeighted((test * 255).astype("uint8"), 1, pred_color, 1, gamma=0)

            cv2.imwrite((self.saveImg + "/" + file_name[index]), final_img)
            print(index+1, end="\r")
        
if __name__ == "__main__":
    model = PredNet(test_dir="./image", model_path="./last.h5")
    model.predict()