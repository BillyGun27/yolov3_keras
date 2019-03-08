"""YOLO_v3 Model Defined in Keras."""

from functools import wraps

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2

from model.utils import compose


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

def resblock_body(x, num_filters, num_blocks):
    '''A series of resblocks starting with a downsampling Convolution2D'''
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Leaky(num_filters//2, (1,1)),
                DarknetConv2D_BN_Leaky(num_filters, (3,3)))(x)
        x = Add()([x,y])
    return x

def darknet_body(x):
    '''Darknent body having 52 Convolution2D layers'''
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x

def make_last_layers(x, num_filters, out_filters):
    '''6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer'''
    x = compose(
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)),
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D_BN_Leaky(num_filters, (1,1)))(x)
    y = compose(
            DarknetConv2D_BN_Leaky(num_filters*2, (3,3)),
            DarknetConv2D(out_filters, (1,1)))(x)
    return x, y


def mobilenet_yolo_body(inputs, num_anchors, num_classes):
    #net, endpoint = inception_v2.inception_v2(inputs)
    mobilenet = MobileNet(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256

    f1 = mobilenet.get_layer('conv_pw_13_relu').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenet.get_layer('conv_pw_11_relu').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenet.get_layer('conv_pw_5_relu').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])

def yolo_body(inputs, num_anchors, num_classes):
     #net, endpoint = inception_v2.inception_v2(inputs)
    mobilenet = MobileNetV2(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256

    f1 = mobilenet.get_layer('out_relu').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenet.get_layer('block_13_expand_relu').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenet.get_layer('block_6_expand_relu').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])

def mobilenetv2_yolo_body(inputs, num_anchors, num_classes):
     #net, endpoint = inception_v2.inception_v2(inputs)
    mobilenet = MobileNetV2(input_tensor=inputs,weights='imagenet')

    # input: 416 x 416 x 3
    # conv_pw_13_relu :13 x 13 x 1024
    # conv_pw_11_relu :26 x 26 x 512
    # conv_pw_5_relu : 52 x 52 x 256

    f1 = mobilenet.get_layer('out_relu').output
    # f1 :13 x 13 x 1024
    x, y1 = make_last_layers(f1, 512, num_anchors * (num_classes + 5))

    x = compose(
            DarknetConv2D_BN_Leaky(256, (1,1)),
            UpSampling2D(2))(x)

    f2 = mobilenet.get_layer('block_13_expand_relu').output
    # f2: 26 x 26 x 512
    x = Concatenate()([x,f2])

    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))

    x = compose(
            DarknetConv2D_BN_Leaky(128, (1,1)),
            UpSampling2D(2))(x)

    f3 = mobilenet.get_layer('block_6_expand_relu').output
    # f3 : 52 x 52 x 256
    x = Concatenate()([x, f3])
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))

    return Model(inputs = inputs, outputs=[y1,y2,y3])

