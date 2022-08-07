import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import Model

import numpy as np
from classification_models.keras import Classifiers


def unet3_plus(shape=(512, 512, 3)):
    # UNet3+ arxiv: https://arxiv.org/abs/2004.08790
    # https://github.com/kochlisGit/Unet3-Plus

    def encoder_block(inputs, n_filters, kernel_size, strides):
        encoder = Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
        encoder = BatchNormalization()(encoder)
        encoder = Activation("relu")(encoder)
        encoder = Conv2D(filters=n_filters, kernel_size=kernel_size, padding='same', use_bias=False)(encoder)
        encoder = BatchNormalization()(encoder)
        encoder = Activation("relu")(encoder)

        return encoder
        
    def upscale_blocks(inputs):
        n_upscales = len(inputs)
        upscale_layers = []

        for i, inp in enumerate(inputs):
            p = n_upscales - i
            u = Conv2DTranspose(filters=64, kernel_size=3, strides=2**p, padding='same')(inp)

            for i in range(2):
                u = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(u)
                u = BatchNormalization()(u)
                u = Activation("relu")(u)
                u = Dropout(rate=0.4)(u)

            upscale_layers.append(u)

        return upscale_layers

    def decoder_block(layers_to_upscale, inputs):
        upscaled_layers = upscale_blocks(layers_to_upscale)

        decoder_blocks = []

        for i, inp in enumerate(inputs):
            d = Conv2D(filters=64, kernel_size=3, strides=2**i, padding='same', use_bias=False)(inp)
            d = BatchNormalization()(d)
            d = Activation("relu")(d)
            d = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False)(d)
            d = BatchNormalization()(d)
            d = Activation("relu")(d)

            decoder_blocks.append(d)

        decoder = concatenate(upscaled_layers + decoder_blocks)
        decoder = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', use_bias=False)(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Activation("relu")(decoder)
        decoder = Dropout(rate=0.4)(decoder)

        return decoder

    inputs_layer = Input(shape=shape)

    e1 = encoder_block(inputs_layer, n_filters=32, kernel_size=3, strides=1)
    e2 = encoder_block(e1, n_filters=64, kernel_size=3, strides=2)
    e3 = encoder_block(e2, n_filters=128, kernel_size=3, strides=2)
    e4 = encoder_block(e3, n_filters=256, kernel_size=3, strides=2)
    e5 = encoder_block(e4, n_filters=512, kernel_size=3, strides=2)

    d4 = decoder_block(layers_to_upscale=[e5], inputs=[e4, e3, e2, e1])
    d3 = decoder_block(layers_to_upscale=[e5, d4], inputs=[e3, e2, e1])
    d2 = decoder_block(layers_to_upscale=[e5, d4, d3], inputs=[e2, e1])
    d1 = decoder_block(layers_to_upscale=[e5, d4, d3, d2], inputs=[e1])

    output = Conv2D(filters=shape[2], kernel_size=1, padding='same', activation='tanh')(d1)

    return Model(inputs_layer, output)

def cpfnet(shape=(512, 512, 3)):
    # CPFNet https://ieeexplore.ieee.org/document/9049412
    # https://github.com/FENGShuanglang/CPFNet_Project

    def UpSampling2D_Bilinear(size):
            return Lambda(lambda x: tf.compat.v1.image.resize_bilinear(x, size, align_corners=True))

    def gpg_2(s2, s3, s4, s5, out_channel):
        (_, height, width, _) = K.int_shape(s2)

        f2 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s2)
        f2 = BatchNormalization()(f2)
        f2 = ReLU()(f2)

        f3 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s3)
        f3 = BatchNormalization()(f3)
        f3 = ReLU()(f3)
        f3 = UpSampling2D_Bilinear((height, width))(f3)

        f4 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s4)
        f4 = BatchNormalization()(f4)
        f4 = ReLU()(f4)
        f4 = UpSampling2D_Bilinear((height, width))(f4)

        f5 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s5)
        f5 = BatchNormalization()(f5)
        f5 = ReLU()(f5)
        f5 = UpSampling2D_Bilinear((height, width))(f5)

        concat1 = Concatenate()([f2, f3, f4, f5])

        d1 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=1)(concat1)
        d1 = BatchNormalization()(d1)
        d1 = ReLU()(d1)

        d2 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=2)(concat1)
        d2 = BatchNormalization()(d2)
        d2 = ReLU()(d2)

        d3 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=4)(concat1)
        d3 = BatchNormalization()(d3)
        d3 = ReLU()(d3)

        d4 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=8)(concat1)
        d4 = BatchNormalization()(d4)
        d4 = ReLU()(d4)

        concta2 = Concatenate()([d1, d2, d3, d4])

        output = Conv2D(filters=out_channel, kernel_size=1, padding='same')(concta2)
        output = BatchNormalization()(output)

        return output
    
    def gpg_3(s3, s4, s5, out_channel):
        (_, height, width, _) = K.int_shape(s3)

        f3 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s3)
        f3 = BatchNormalization()(f3)
        f3 = ReLU()(f3)

        f4 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s4)
        f4 = BatchNormalization()(f4)
        f4 = ReLU()(f4)
        f4 = UpSampling2D_Bilinear((height, width))(f4)

        f5 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s5)
        f5 = BatchNormalization()(f5)
        f5 = ReLU()(f5)
        f5 = UpSampling2D_Bilinear((height, width))(f5)

        concat1 = Concatenate()([f3, f4, f5])

        d1 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=1)(concat1)
        d1 = BatchNormalization()(d1)
        d1 = ReLU()(d1)

        d2 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=2)(concat1)
        d2 = BatchNormalization()(d2)
        d2 = ReLU()(d2)

        d3 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=4)(concat1)
        d3 = BatchNormalization()(d3)
        d3 = ReLU()(d3)

        concta2 = Concatenate()([d1, d2, d3])

        output = Conv2D(filters=out_channel, kernel_size=1, padding='same')(concta2)
        output = BatchNormalization()(output)

        return output

    def gpg_4(s4, s5, out_channel):
        (_, height, width, _) = K.int_shape(s4)

        f4 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s4)
        f4 = BatchNormalization()(f4)
        f4 = ReLU()(f4)

        f5 = Conv2D(filters=out_channel, kernel_size=3, padding='same')(s5)
        f5 = BatchNormalization()(f5)
        f5 = ReLU()(f5)
        f5 = UpSampling2D_Bilinear((height, width))(f5)

        concat1 = Concatenate()([f4, f5])

        d1 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=1)(concat1)
        d1 = BatchNormalization()(d1)
        d1 = ReLU()(d1)

        d2 = SeparableConv2D(filters=out_channel, kernel_size=3, padding='same', dilation_rate=2)(concat1)
        d2 = BatchNormalization()(d2)
        d2 = ReLU()(d2)

        concta2 = Concatenate()([d1, d2])

        output = Conv2D(filters=out_channel, kernel_size=1, padding='same')(concta2)
        output = BatchNormalization()(output)

        return output

    def sapf(input_layer):
        def sa_module(input1, input2, ch):
            concat = Concatenate()([input1, input2])

            concat = Conv2D(filters=ch, kernel_size=1)(concat)
            concat = ReLU()(concat)

            concat = Conv2D(filters=ch//2, kernel_size=3, padding='same')(concat)
            concat = ReLU()(concat)

            att = Conv2D(filters=2, kernel_size=3, padding='same')(concat)
            att = Softmax()(att)

            att1 = Lambda(lambda x: K.expand_dims(x[:, :, :, 0]))(att)
            att2 = Lambda(lambda x: K.expand_dims(x[:, :, :, 1]))(att)

            mul1 = Multiply()([input1, att1])
            mul2 = Multiply()([input2, att2])
            fushion = Add()([mul1, mul2])

            return fushion

        (_, height, width, ch) = K.int_shape(input_layer)

        branch_1 = Conv2D(filters=ch, kernel_size=3, padding='same', dilation_rate=1, name="branch_1")(input_layer)
        branch_1 = BatchNormalization()(branch_1)

        branch_2 = Conv2D(filters=ch, kernel_size=3, padding='same', dilation_rate=2, name="branch_2")(input_layer)
        branch_2 = BatchNormalization()(branch_2)

        branch_3 = Conv2D(filters=ch, kernel_size=3, padding='same', dilation_rate=4, name="branch_3")(input_layer)
        branch_3 = BatchNormalization()(branch_3)

        fushion1 = sa_module(branch_1, branch_2, ch)
        fushion2 = sa_module(fushion1, branch_3, ch)

        alpha = K.variable(K.zeros(shape=1))

        # mul1 = Lambda(lambda x: x * alpha)(fushion2)
        # mul2 = Lambda(lambda x: x * (1-alpha))(input_layer)
        # output = Add()([mul1, mul2])
        output = Add()([fushion2, input_layer])
        return output

    def decoder(input_layer,in_ch, out_ch, last_layer):
        (_, height, width, _) = K.int_shape(input_layer)

        if last_layer:
            input_layer = Conv2D(filters=in_ch, kernel_size=3, strides=1, padding='same')(input_layer)
            input_layer = BatchNormalization()(input_layer)
            input_layer = ReLU()(input_layer)
        
        up1 = UpSampling2D_Bilinear((height*2, width*2))(input_layer)

        conv = Conv2D(filters=out_ch, kernel_size=1, strides=1)(up1)
        conv = BatchNormalization()(conv)
        output = ReLU()(conv)

        return output

    ResNet34, _ = Classifiers.get('resnet34')
    ResNet34_model = ResNet34(input_shape=shape, weights='imagenet', include_top=False)

    ''' Encoder '''
    s1 = ResNet34_model.get_layer(name="relu0").output                    # 256, 256, 64
    s2 = ResNet34_model.get_layer(name="stage2_unit1_relu1").output       # 128, 128, 64
    s3 = ResNet34_model.get_layer(name="stage3_unit1_relu1").output       # 64, 64, 128
    s4 = ResNet34_model.get_layer(name="stage4_unit1_relu1").output       # 32, 32, 256
    s5 = ResNet34_model.output                                            # 16, 16, 512

    g2 = gpg_2(s2, s3, s4, s5, out_channel=64) # 64
    g3 = gpg_3(s3, s4, s5, out_channel=128) # 128
    g4 = gpg_4(s4, s5, out_channel=256)
    
    s5 = sapf(s5)   # 16, 16, 512

    ''' Decoder '''
    d4 = decoder(s5, 512, 256, last_layer=True) # 32, 32, 256
    d4 = Add()([d4, g4])
    d4 = ReLU(name="d4")(d4)

    d3 = decoder(d4, 256, 128, last_layer=False)  # 64, 64, 128
    d3 = Add()([d3, g3])
    d3 = LeakyReLU(name="d3")(d3)   # Relu

    d2 = decoder(d3, 128, 64, last_layer=False) # 128, 128, 64
    d2 = Add()([d2, g2])
    d2 = LeakyReLU(name="d2")(d2)   # Relu

    d1 = decoder(d2, 64, 64, last_layer=False)  # 256, 256, 64
    d1 = Add()([d1, s1])

    output = UpSampling2D_Bilinear((shape[0], shape[1]))(d1)
    output = Conv2D(filters=32, kernel_size=1, strides=1)(output)
    output = BatchNormalization()(output)
    output = ReLU()(output)

    output = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(output)
    output = BatchNormalization()(output)
    output = ReLU()(output)

    output = Conv2D(filters=shape[2], kernel_size=1, strides=1)(output)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[ResNet34_model.input], outputs=[output])
    weight = model.get_layer(name="branch_1").get_weights()
    model.get_layer(name="branch_2").set_weights(weight)
    model.get_layer(name="branch_3").set_weights(weight)

    return model

def pspnet(shape=(512, 512, 3)):
    # Pyramid Scene Parsing Network https://arxiv.org/abs/1612.01105

    def conv_block(input_tensor, filters, strides, d_rates):
        x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters[1], kernel_size=3, strides=strides, padding='same', dilation_rate=d_rates[1])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
        x = BatchNormalization()(x)

        shortcut = Conv2D(filters[2], kernel_size=1, strides=strides)(input_tensor)
        shortcut = BatchNormalization()(shortcut)

        x = add([x, shortcut])
        x = Activation('relu')(x)

        return x
    
    def identity_block(input_tensor, filters, d_rates):
        x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters[1], kernel_size=3, padding='same', dilation_rate=d_rates[1])(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
        x = BatchNormalization()(x)

        x = add([x, input_tensor])
        x = Activation('relu')(x)

        return x
    
    def pyramid_pooling_block(input_tensor, bin_sizes):
        concat_list = [input_tensor]
        h = input_tensor.shape[1]
        w = input_tensor.shape[2]

        for bin_size in bin_sizes:
            x = AveragePooling2D(pool_size=(h//bin_size, w//bin_size), strides=(h//bin_size, w//bin_size))(input_tensor)
            x = Conv2D(512, kernel_size=1)(x)
            x = Lambda(lambda x: tf.image.resize(x, (h, w)))(x)

            concat_list.append(x)

        return concatenate(concat_list)
    
    
    img_input = Input(shape)

    x = Conv2D(64, kernel_size=3, strides=(2, 2), padding='same')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, filters=[64, 64, 256], strides=(1, 1), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[64, 64, 256], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[64, 64, 256], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[128, 128, 512], strides=(2, 2), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])

    x = conv_block(x, filters=[256, 256, 1024], strides=(1, 1), d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 2, 1])

    x = conv_block(x, filters=[512, 512, 2048], strides=(1, 1), d_rates=[1, 4, 1])
    x = identity_block(x, filters=[512, 512, 2048], d_rates=[1, 4, 1])
    x = identity_block(x, filters=[512, 512, 2048], d_rates=[1, 4, 1])

    x = pyramid_pooling_block(x, [1, 2, 3, 6])

    x = Conv2D(512, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(shape[2], kernel_size=1)(x)
    x = Conv2DTranspose(shape[2], kernel_size=(16, 16), strides=(8, 8), padding='same')(x)
    x = Activation('sigmoid')(x)

    return Model(img_input, x)

def vnet(shape=(512, 512, 3),stage_num=5):
    # VNet https://arxiv.org/abs/1606.04797
    # https://github.com/FENGShuanglang/2D-Vnet-Keras

    def resBlock(conv,stage,keep_prob,stage_num=5):
        inputs=conv
        
        for _ in range(3 if stage>3 else stage):
            conv=PReLU()(BatchNormalization()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))

        conv_add=PReLU()(add([inputs,conv]))
        conv_drop=Dropout(keep_prob)(conv_add)
        
        if stage<stage_num:
            conv_downsample=PReLU()(BatchNormalization()(Conv2D(16*(2**stage), 2, strides=(2, 2),activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv_drop)))
            return conv_downsample,conv_add
        else:
            return conv_add,conv_add


    def up_resBlock(forward_conv,input_conv,stage):
        conv=concatenate([forward_conv,input_conv],axis = -1)

        for _ in range(3 if stage>3 else stage):
            conv=PReLU()(BatchNormalization()(Conv2D(16*(2**(stage-1)), 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv)))

        conv_add=PReLU()(add([input_conv,conv]))
        if stage>1:
            conv_upsample=PReLU()(BatchNormalization()(Conv2DTranspose(16*(2**(stage-2)),2,strides=(2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(conv_add)))
            return conv_upsample
        else:
            return conv_add
    

    keep_prob = 0.2
    features=[]
    input_model = Input(shape)
    x=PReLU()(BatchNormalization()(Conv2D(16, 5, activation = None, padding = 'same', kernel_initializer = 'he_normal')(input_model)))
    
    for s in range(1,stage_num+1):
        x,feature=resBlock(x,s,keep_prob,stage_num)
        features.append(feature)
        
    conv_up=PReLU()(BatchNormalization()(Conv2DTranspose(16*(2**(s-2)),2,strides=(2, 2),padding='valid',activation = None,kernel_initializer = 'he_normal')(x)))
    
    for d in range(stage_num-1,0,-1):
        conv_up=up_resBlock(features[d-1],conv_up,d)

    conv_out=Conv2D(shape[2], 1, activation = 'softmax', padding = 'same', kernel_initializer = 'he_normal')(conv_up)
    
    return Model(inputs=input_model,outputs=conv_out)

def cbam_aspp_resUnet(shape=(512, 512, 3)):
    # https://github.com/billymoonxd/ResUNet

    def ca_block(inputs, ratio=16):
        """
        Channel Attention Module exploiting the inter-channel relationship of features.
        """
        shape = inputs.shape
        filters = shape[-1]

        # avg_pool = Lambda(lambda x: K.mean(x, axis=[1, 2], keepdims=True))(inputs)
        # max_pool = Lambda(lambda x: K.max(x, axis=[1, 2], keepdims=True))(inputs)
        # avg_pool = AveragePooling2D(pool_size=(shape[1], shape[2]))(inputs)
        # max_pool = MaxPooling2D(pool_size=(shape[1], shape[2]))(inputs)
        avg_pool = K.mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = K.max(inputs, axis=[1, 2], keepdims=True)

        x1 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(avg_pool)
        x1 = Dense(filters, activation=None, kernel_initializer='he_normal', use_bias=False)(x1)

        x2 = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(max_pool)
        x2 = Dense(filters, activation=None, kernel_initializer='he_normal', use_bias=False)(x2)

        x = Add()([x1, x2])
        x = Activation('sigmoid')(x)

        outputs = Multiply()([inputs, x])
        return outputs

    def sa_block(inputs):
        """
        Spatial Attention Module utilizing the inter-spatial relationship of features.
        """
        kernel_size = 7

        # avg_pool = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inputs)
        # max_pool = Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inputs)
        avg_pool = K.mean(inputs, axis=-1, keepdims=True)
        max_pool = K.max(inputs, axis=-1, keepdims=True)

        x = Concatenate()([avg_pool, max_pool])

        x = Conv2D(1, kernel_size, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(x)

        outputs = Multiply()([inputs, x])
        return outputs

    def cbam_block(inputs):
        """
        CBAM: Convolutional Block Attention Module, which combines Channel Attention Module and Spatial Attention Module,
        focusing on `what` and `where` respectively. The sequential channel-spatial order proves to perform best.
        See: https://arxiv.org/pdf/1807.06521.pdf
        """
        x = ca_block(inputs)
        x = sa_block(x)
        return x

    def ca_stem_block(inputs, filters, strides=1):
        """
        Residual block for the first layer of Deep Residual U-Net.
        See: https://arxiv.org/pdf/1711.10684.pdf
        Code from: https://github.com/dmolony3/ResUNet
        """
        # Conv
        x = Conv2D(filters, (3, 3), padding="same", strides=strides)(inputs)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)

        # CA
        x = ca_block(x)

        # Shortcut
        s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
        s = BatchNormalization()(s)

        # Add
        outputs = Add()([x, s])
        return outputs

    def ca_resblock(inputs, filters, strides=1):
        """
        Residual block with Channel Attention Module.
        """
        # Conv
        x = BatchNormalization()(inputs)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

        # CA
        x = ca_block(x)

        # Shortcut
        s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
        s = BatchNormalization()(s)

        # Add
        outputs = Add()([x, s])
        return outputs

    def cbam_resblock(inputs, filters, strides=1):
        """
        Residual block with Convolutional Block Attention Module.
        """
        # Conv
        x = BatchNormalization()(inputs)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

        # CBAM
        x = cbam_block(x)

        # Shortcut
        s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
        s = BatchNormalization()(s)

        # Add
        outputs = Add()([x, s])
        return outputs

    def sa_resblock(inputs, filters, strides=1):
        """
        Residual block with Spatial Attention Module.
        """
        # Conv
        x = BatchNormalization()(inputs)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same", strides=strides)(x)

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same", strides=1)(x)

        # SA
        x = sa_block(x)

        # Shortcut
        s = Conv2D(filters, (1, 1), padding="same", strides=strides)(inputs)
        s = BatchNormalization()(s)

        # Add
        outputs = Add()([x, s])
        return outputs

    def feature_fusion(high, low):
        """
        Low- and high-level feature fusion, taking advantage of multi-level contextual information.
        Args:
            high: high-level semantic information in the contracting path.
            low: low-level feature map in the symmetric expanding path.
        See: https://arxiv.org/pdf/1804.03999.pdf
        """
        filters = low.shape[-1]

        x1 = UpSampling2D(size=(2, 2))(high)
        x1 = BatchNormalization()(x1)
        x1 = Activation("relu")(x1)
        x1 = Conv2D(filters, (3, 3), padding="same")(x1)

        x2 = BatchNormalization()(low)
        x2 = Activation("relu")(x2)
        x2 = Conv2D(filters, (3, 3), padding="same")(x2)

        x = Add()([x1, x2])

        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters, (3, 3), padding="same")(x)

        outputs = Multiply()([x, low])
        return outputs

    n_filters = [32, 64, 128, 256, 512]

    inputs = Input(shape)

    # Encoder
    c0 = ca_stem_block(inputs, n_filters[0])

    c1 = cbam_resblock(c0, n_filters[1], strides=1)
    c1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c1)

    c2 = cbam_resblock(c1, n_filters[2], strides=1)
    c2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c2)

    c3 = cbam_resblock(c2, n_filters[3], strides=1)
    c3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(c3)

    # Bridge
    b1 = sa_resblock(c3, n_filters[4])

    # Decoder
    # Nearest-neighbor UpSampling followed by Conv2D & ReLU to dampen checkerboard artifacts.
    # See: https://distill.pub/2016/deconv-checkerboard/

    d1 = UpSampling2D(size=(2, 2))(b1)
    d1 = Conv2D(n_filters[3], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d1)
    d1 = feature_fusion(c3, d1)
    d1 = Concatenate()([d1, c2])
    d1 = cbam_resblock(d1, n_filters[3])

    d2 = UpSampling2D(size=(2, 2))(d1)
    d2 = Conv2D(n_filters[2], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d2)
    d2 = feature_fusion(c2, d2)
    d2 = Concatenate()([d2, c1])
    d2 = cbam_resblock(d2, n_filters[2])

    d3 = UpSampling2D(size=(2, 2))(d2)
    d3 = Conv2D(n_filters[1], (3, 3), padding="same", activation='relu', kernel_initializer='he_normal')(d3)
    d3 = feature_fusion(c1, d3)
    d3 = Concatenate()([d3, c0])
    d3 = cbam_resblock(d3, n_filters[1])

    # Output
    outputs = ca_resblock(d3, n_filters[0])
    outputs = Conv2D(shape[2], (1, 1), padding="same")(outputs)
    outputs = Activation("sigmoid")(outputs)

    # Model
    model = Model(inputs, outputs)
    return model

def resUnet_plus_plus(shape=(512, 512, 3)):
    # https://github.com/billymoonxd/ResUNet

    def squeeze_excite_block(inputs, ratio=8):
        init = inputs
        channel_axis = -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

        x = Multiply()([init, se])
        return x

    def stem_block(x, n_filter, strides):
        x_init = x

        ## Conv 1
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same")(x)

        ## Shortcut
        s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        ## Add
        x = Add()([x, s])
        x = squeeze_excite_block(x)
        return x

    def resnet_block(x, n_filter, strides=1):
        x_init = x

        ## Conv 1
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=strides)(x)
        ## Conv 2
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(n_filter, (3, 3), padding="same", strides=1)(x)

        ## Shortcut
        s  = Conv2D(n_filter, (1, 1), padding="same", strides=strides)(x_init)
        s = BatchNormalization()(s)

        ## Add
        x = Add()([x, s])
        x = squeeze_excite_block(x)
        return x
    
    def aspp_block(x, num_filters, rate_scale=1):
        x1 = Conv2D(num_filters, (3, 3), dilation_rate=(6 * rate_scale, 6 * rate_scale), padding="same")(x)
        x1 = BatchNormalization()(x1)

        x2 = Conv2D(num_filters, (3, 3), dilation_rate=(12 * rate_scale, 12 * rate_scale), padding="same")(x)
        x2 = BatchNormalization()(x2)

        x3 = Conv2D(num_filters, (3, 3), dilation_rate=(18 * rate_scale, 18 * rate_scale), padding="same")(x)
        x3 = BatchNormalization()(x3)

        x4 = Conv2D(num_filters, (3, 3), padding="same")(x)
        x4 = BatchNormalization()(x4)

        y = Add()([x1, x2, x3, x4])
        y = Conv2D(num_filters, (1, 1), padding="same")(y)
        return y
    
    def attetion_block(g, x):
        """
            g: Output of Parallel Encoder block
            x: Output of Previous Decoder block
        """

        filters = x.shape[-1]

        g_conv = BatchNormalization()(g)
        g_conv = Activation("relu")(g_conv)
        g_conv = Conv2D(filters, (3, 3), padding="same")(g_conv)

        g_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(g_conv)

        x_conv = BatchNormalization()(x)
        x_conv = Activation("relu")(x_conv)
        x_conv = Conv2D(filters, (3, 3), padding="same")(x_conv)

        gc_sum = Add()([g_pool, x_conv])

        gc_conv = BatchNormalization()(gc_sum)
        gc_conv = Activation("relu")(gc_conv)
        gc_conv = Conv2D(filters, (3, 3), padding="same")(gc_conv)

        gc_mul = Multiply()([gc_conv, x])
        return gc_mul
    
    n_filters = [16, 32, 64, 128, 256]
    inputs = Input(shape)

    c0 = inputs
    c1 = stem_block(c0, n_filters[0], strides=1)

    ## Encoder
    c2 = resnet_block(c1, n_filters[1], strides=2)
    c3 = resnet_block(c2, n_filters[2], strides=2)
    c4 = resnet_block(c3, n_filters[3], strides=2)

    ## Bridge
    b1 = aspp_block(c4, n_filters[4])

    ## Decoder
    d1 = attetion_block(c3, b1)
    d1 = UpSampling2D((2, 2))(d1)
    d1 = Concatenate()([d1, c3])
    d1 = resnet_block(d1, n_filters[3])

    d2 = attetion_block(c2, d1)
    d2 = UpSampling2D((2, 2))(d2)
    d2 = Concatenate()([d2, c2])
    d2 = resnet_block(d2, n_filters[2])

    d3 = attetion_block(c1, d2)
    d3 = UpSampling2D((2, 2))(d3)
    d3 = Concatenate()([d3, c1])
    d3 = resnet_block(d3, n_filters[1])

    ## output
    outputs = aspp_block(d3, n_filters[0])
    outputs = Conv2D(shape[2], (1, 1), padding="same")(outputs)
    outputs = Activation("sigmoid")(outputs)

    ## Model
    model = Model(inputs, outputs)
    return model

def cenet(shape=(512, 512, 3)):
    # CE-Net https://arxiv.org/abs/1903.02740
    # https://www.kaggle.com/code/momincks/medical-image-segmentation-with-ce-net/notebook

    keep_scale = 0.2
    l1l2 = tf.keras.regularizers.l1_l2(l1=0, l2=0.0005)

    def resblock(x,level='en_l1',filters=64,keep_scale=keep_scale,l1l2=l1l2,downsample=False,bn_act=True,first_layer=False):
        if downsample:
            if not first_layer:
                x_H = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_Hconv')(x)
                x = Conv2D(filters/2,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
            else:
                x_H = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_Hconv')(x)
                x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
        else:
            x_H = x
            x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv1')(x)
        x = BatchNormalization(name=level+'_conv1bn')(x)
        x = ReLU(name=level+'_conv1relu')(x)
        if downsample:
            x = Conv2D(filters,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name=level+'_conv2')(x)
        else:
            x = Conv2D(filters,(3,3),padding='same',kernel_regularizer=l1l2,name=level+'_conv2')(x)
        x = BatchNormalization(name=level+'_conv2bn')(x)
        x = ReLU(name=level+'_conv2relu')(x)
        x = Add(name=level+'_add')([x_H*keep_scale,x*(1-keep_scale)])
        if bn_act:
            x = BatchNormalization(name=level+'_finalbn')(x)
            x = ReLU(name=level+'_finalrelu')(x)       
        return x

    inputs = tf.keras.Input(shape=shape,name='input')

    layer_0 = Conv2D(64,(7,7),strides=(2,2),padding='same',name='input_conv1',activation=None)(inputs)
    layer = BatchNormalization(name='input_conv1bn')(layer_0)
    layer = ReLU(name='input_conv1relu')(layer)

    ## Encoder
    layer = resblock(layer,'en_l1',64,keep_scale,l1l2,downsample=True,bn_act=True,first_layer=True)
    layer = resblock(layer,'en_l2',64,keep_scale,l1l2,downsample=False,bn_act=True)
    encoder_1 = resblock(layer,'en_l3',64,keep_scale,l1l2,downsample=False,bn_act=False)
    layer = BatchNormalization(name='en_l3_finalbn')(encoder_1)
    layer = ReLU(name='en_l3_finalrelu')(layer)

    layer = resblock(layer,'en_l4',128,keep_scale,l1l2,downsample=True,bn_act=True)
    layer = resblock(layer,'en_l5',128,keep_scale,l1l2,downsample=False,bn_act=True)
    layer = resblock(layer,'en_l6',128,keep_scale,l1l2,downsample=False,bn_act=True)
    encoder_2 = resblock(layer,'en_l7',128,keep_scale,l1l2,downsample=False,bn_act=False)
    layer = BatchNormalization(name='en_l7_finalbn')(encoder_2)
    layer = ReLU(name='en_l7_finalrelu')(layer)

    layer = resblock(layer,'en_l8',256,keep_scale,l1l2,downsample=True,bn_act=True)
    layer = resblock(layer,'en_l9',256,keep_scale,l1l2,downsample=False,bn_act=True)
    layer = resblock(layer,'en_l10',256,keep_scale,l1l2,downsample=False,bn_act=True)
    layer = resblock(layer,'en_l11',256,keep_scale,l1l2,downsample=False,bn_act=True)
    layer = resblock(layer,'en_l12',256,keep_scale,l1l2,downsample=False,bn_act=True)
    encoder_3 = resblock(layer,'en_l13',256,keep_scale,l1l2,downsample=False,bn_act=False)
    layer = BatchNormalization(name='en_l13_finalbn')(encoder_3)
    layer = ReLU(name='en_l13_finalrelu')(layer)

    layer = resblock(layer,'en_l14',512,keep_scale,l1l2,downsample=True,bn_act=True)
    layer = resblock(layer,'en_l15',512,keep_scale,l1l2,downsample=False,bn_act=True)
    layer = resblock(layer,'en_l16',512,keep_scale,l1l2,downsample=False,bn_act=True)

    ## DAC block
    b1 = Conv2D(512,(3,3),padding='same',dilation_rate=1,name='dac_b1_conv1',activation=None)(layer)
    # b1 = BatchNormalization()(b1)
    # b1 = ReLU(name='dac_b1_relu')(b1)

    b2 = Conv2D(512,(3,3),padding='same',dilation_rate=3,name='dac_b2_conv1',activation=None)(layer)
    b2 = Conv2D(512,(1,1),padding='same',dilation_rate=1,name='dac_b2_conv2',activation=None)(b2)
    # b2 = BatchNormalization()(b2)
    # b2 = ReLU(name='dac_b2_relu')(b2)

    b3 = Conv2D(512,(3,3),padding='same',dilation_rate=1,name='dac_b3_conv1',activation=None)(layer)
    b3 = Conv2D(512,(3,3),padding='same',dilation_rate=3,name='dac_b3_conv2',activation=None)(b3)
    b3 = Conv2D(512,(1,1),padding='same',dilation_rate=1,name='dac_b3_conv3',activation=None)(b3)
    # b3 = BatchNormalization()(b3)
    # b3 = ReLU(name='dac_b3_relu')(b3)

    b4 = Conv2D(512,(3,3),padding='same',dilation_rate=1,name='dac_b4_conv1',activation=None)(layer)
    b4 = Conv2D(512,(3,3),padding='same',dilation_rate=3,name='dac_b4_conv2',activation=None)(b4)
    b4 = Conv2D(512,(3,3),padding='same',dilation_rate=5,name='dac_b4_conv3',activation=None)(b4)
    b4 = Conv2D(512,(1,1),padding='same',dilation_rate=1,name='dac_b4_conv4',activation=None)(b4)
    # b4 = BatchNormalization()(b4)
    # b4 = ReLU(name='dac_b4_relu')(b4)

    layer = Add(name='dac_add')([layer,b1,b2,b3,b4])
    # layer = BatchNormalization(name='dac_bn')(layer)
    layer = ReLU(name='dac_relu')(layer)
    
    
    ## RMP block
    size = layer.shape[1]
    b1 = MaxPool2D((2,2),strides=(2,2),padding='valid',name='rmp_b1_pool')(layer)
    b1 = Conv2D(1,(1,1),padding='valid',name='rmb_b1_conv1',activation=None)(b1)
    b1 = Conv2DTranspose(1,(1,1),(2,2),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b1)
    b1 = tf.image.resize(b1, [size,size], method=tf.image.ResizeMethod.BILINEAR)

    b2 = MaxPool2D((3,3),strides=(3,3),padding='valid',name='rmp_b2_pool')(layer)
    b2 = Conv2D(1,(1,1),padding='valid',name='rmb_b2_conv1',activation=None)(b2)
    b2 = Conv2DTranspose(1,(1,1),(3,3),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b2)
    b2 = tf.image.resize(b2, [size,size], method=tf.image.ResizeMethod.BILINEAR)

    b3 = MaxPool2D((5,5),strides=(5,5),padding='valid',name='rmp_b3_pool')(layer)
    b3 = Conv2D(1,(1,1),padding='valid',name='rmb_b3_conv1',activation=None)(b3)
    b3 = Conv2DTranspose(1,(1,1),(5,5),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b3)
    b3 = tf.image.resize(b3, [size,size], method=tf.image.ResizeMethod.BILINEAR)

    b4 = MaxPool2D((6,6),strides=(6,6),padding='valid',name='rmp_b4_pool')(layer)
    b4 = Conv2D(1,(1,1),padding='valid',name='rmb_b4_conv1',activation=None)(b4)
    b4 = Conv2DTranspose(1,(1,1),(6,6),padding='valid',kernel_regularizer=l1l2,output_padding=0,activation=None)(b4)
    b4 = tf.image.resize(b4, [size,size], method=tf.image.ResizeMethod.BILINEAR)

    layer = Concatenate(name='rmp_concat')([layer,b1,b2,b3,b4])
    layer = ReLU(name='rmp_relu')(layer)

    layer = Conv2D(256,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l1_conv1',activation=None)(layer)
    layer = BatchNormalization(name='de_l1_conv1bn')(layer)
    layer = ReLU(name='de_l1_conv1relu')(layer)
    layer = Conv2DTranspose(256,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l1_deconv2',activation=None)(layer)
    layer = BatchNormalization(name='de_l1_conv2bn')(layer)
    layer = ReLU(name='de_l1_deconv2relu')(layer)
    layer = Conv2D(256,(3,3),padding='same',kernel_regularizer=l1l2,name='de_l1_conv3',activation=None)(layer)
    layer = Add(name='de_l1_add')([encoder_3*keep_scale,layer*(1-keep_scale)])
    layer = BatchNormalization(name='de_l1_conv3bn')(layer)
    layer = ReLU(name='de_l1_conv3relu')(layer)

    layer = Conv2D(128,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l2_conv1',activation=None)(layer)
    layer = BatchNormalization(name='de_l2_conv1bn')(layer)
    layer = ReLU(name='de_l2_conv1relu')(layer)
    layer = Conv2DTranspose(128,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l2_deconv2',activation=None)(layer)
    layer = BatchNormalization(name='de_l2_conv2bn')(layer)
    layer = ReLU(name='de_l2_deconv2relu')(layer)
    layer = Conv2D(128,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l2_conv3',activation=None)(layer)
    layer = Add(name='de_l2_add')([encoder_2*keep_scale,layer*(1-keep_scale)])
    layer = BatchNormalization(name='de_l2_conv3bn')(layer)
    layer = ReLU(name='de_l2_conv3relu')(layer)

    layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l3_conv1',activation=None)(layer)
    layer = BatchNormalization(name='de_l3_conv1bn')(layer)
    layer = ReLU(name='de_l3_conv1relu')(layer)
    layer = Conv2DTranspose(64,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l3_deconv2',activation=None)(layer)
    layer = BatchNormalization(name='de_l3_conv2bn')(layer)
    layer = ReLU(name='de_l3_deconv2relu')(layer)
    layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l3_conv3',activation=None)(layer)
    layer = Add(name='de_l3_add')([encoder_1*keep_scale,layer*(1-keep_scale)])
    layer = BatchNormalization(name='de_l3_conv3bn')(layer)
    layer = ReLU(name='de_l3_conv3relu')(layer)

    layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l4_conv1',activation=None)(layer)
    layer = BatchNormalization(name='de_l4_conv1bn')(layer)
    layer = ReLU(name='de_l4_conv1relu')(layer)
    layer = Conv2DTranspose(64,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,output_padding=1,name='de_l4_deconv2',activation=None)(layer)
    layer = BatchNormalization(name='de_l4_conv2bn')(layer)
    layer = ReLU(name='de_l4_deconv2relu')(layer)
    layer = Conv2D(64,(1,1),padding='same',kernel_regularizer=l1l2,name='de_l4_conv3',activation=None)(layer)
    layer = Add(name='de_l4_add')([layer_0*keep_scale,layer*(1-keep_scale)])
    layer = BatchNormalization(name='de_l4_conv3bn')(layer)
    layer = ReLU(name='de_l4_conv3relu')(layer)

    layer = Conv2DTranspose(32,(3,3),(2,2),padding='same',kernel_regularizer=l1l2,name='final_deconv1',activation=None)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(32,(3,3),padding='same',kernel_regularizer=l1l2,name='final_conv1',activation=None)(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    outputs = Conv2D(shape[2],(3,3),padding='same',name='output',activation=None)(layer)
    outputs = Activation("sigmoid")(outputs)
    
    return Model(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    model = cenet()

    inputs = np.zeros((1, 512, 512, 3))
    output = model.predict(inputs)
    print(output.shape)
