from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from loss_functions import bce_dice_loss, dice_coefficient, iou_metric
from models import unet3_plus, cpfnet, pspnet, vnet, cbam_aspp_resUnet, resUnet_plus_plus, cenet


class SegNet:
    def __init__(self, trainX_dir=None, trainY_dir=None):
        self.size = 512
        self.ch = 3
        self.batch_size = 16
        self.trainX_dir = trainX_dir
        self.trainY_dir = trainY_dir
        # -----------------------------

        self.trainX_gen, self.trainY_gen, self.trainX_num, self.valX_num = self.load_data(self.trainX_dir, self.trainY_dir)
        self.model = cpfnet()
        # -----------------------------

        self.model.compile(optimizer=Adam(lr=1e-4), loss=bce_dice_loss(), metrics=[iou_metric(), 'accuracy'])

    # ------------------------------------------
    
    def load_data(self, trainX_dir, trainY_dir):
        trainX_dataGen = ImageDataGenerator(horizontal_flip=True, rescale=1./255, validation_split=0.3)
        trainY_dataGen = ImageDataGenerator(horizontal_flip=True, rescale=1./255, validation_split=0.3)

        trainX_gen = trainX_dataGen.flow_from_directory(trainX_dir, target_size=(self.size, self.size), class_mode=None, batch_size=self.batch_size, seed=920, subset="training")
        trainY_gen = trainY_dataGen.flow_from_directory(trainY_dir, target_size=(self.size, self.size), class_mode=None, batch_size=self.batch_size, seed=920, subset="training")
        valX_gen = trainX_dataGen.flow_from_directory(trainX_dir, target_size=(self.size, self.size), class_mode=None, batch_size=self.batch_size, seed=920, subset="validation")
        valY_gen = trainY_dataGen.flow_from_directory(trainY_dir, target_size=(self.size, self.size), class_mode=None, batch_size=self.batch_size, seed=920, subset="validation")

        return zip(trainX_gen, trainY_gen), zip(valX_gen, valY_gen), trainX_gen.samples, valX_gen.samples

    def train(self, epoch, save_path):
        checkPoint = ModelCheckpoint("./best-{epoch:02d}-{val_loss:.4f}.h5", monitor='val_loss', mode='min', save_weights_only=True, save_best_only=True)

        self.model.fit_generator(generator=self.trainX_gen,
                                steps_per_epoch= self.trainX_num // self.batch_size,
                                epochs=epoch,
                                callbacks=[checkPoint],
                                validation_data=self.trainY_gen,
                                validation_steps=self.valX_num // self.batch_size,
                                workers=1)
        
        self.model.save_weights(save_path)


if __name__ == "__main__":
    model = SegNet(trainX_dir="./image",
                    trainY_dir="./mask")

    model.train(epoch=300, save_path="./last.h5")