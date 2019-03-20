"""
Retrain the YOLO model for your own dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda ,Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.core import preprocess_true_boxes, yolo_loss
from model.yolo3 import yolo_body, tiny_yolo_body
from model.utils import get_random_data


def _main():
    train_path = '2007_train.txt'
    val_path = '2007_val.txt'
   # test_path = '2007_test.txt'
    log_dir = 'logs/000/'
    classes_path = 'class/voc_classes.txt'
    anchors_path = 'anchors/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    num_anchors = len(anchors)

    input_shape = (416,416) # multiple of 32, hw

    with open(train_path) as f:
        train_lines = f.readlines()

    with open(val_path) as f:
        val_lines = f.readlines()

   # with open(test_path) as f:
   #     test_lines = f.readlines()
    '''
    image_input = Input(shape=(416, 416, 3))
    model = yolo_body(image_input, num_anchors//3, num_classes)
    model.load_weights("model_data/trained_weights_final.h5")

    
    # class+5
    yolo3 = Reshape((13, 13, 3, 25))(model.layers[-3].output)
    yolo2 = Reshape((26, 26, 3, 25))(model.layers[-2].output)
    yolo1 = Reshape((52, 52, 3, 25))(model.layers[-1].output)
    

    model = Model( input= model.input , output=[yolo3,yolo2,yolo1] )
    #model.summary()

    model.compile(
        optimizer=Adam(lr=1e-3) , 
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    '''
    
    batch_size = 16
    
    train_logits = {}
    val_logits = {}

    print( "total "+ str(len(train_lines)) + " loop "+ str( len(train_lines)//batch_size +1 ) )
    i = 0 #step
    for  logits in data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes) : 
        #x , y = dat
        train_logits[i] = logits
        #trainY[i] = dat
        #print(x.shape)
        #print(logits[0][0].shape)
        #print( logits[1] )
        #print( train_logits[0][0][1].shape)
        #print( len( train_logits[0][1] ) )
        #print(i)
        #print(img.shape)
        #print(dat)
        i+=1
        if i>= (len(train_lines)//batch_size+1) :
            break

    print( "total "+ str(len(val_lines)) + " loop "+ str( len(val_lines)//batch_size +1 ) )
    i = 0 #step
    for  logits in data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes) : 
        #x , y = dat
        val_logits[i] = logits
        i+=1
        if i>= (len(val_lines)//batch_size+1) :
            break
    

    np.save('train_logits.npy', train_logits)
    np.save('val_logits.npy', val_logits)

    train_log = np.load('train_logits.npy')[()]

    print(train_log[0][0][0].shape)
    print( len(train_log[0][1]) )

   # data_train_generator = ImageDataGenerator()

    #   test_generator = data_train_generator.flow(trainX, trainY, batch_size=batch_size), #data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes)

  #  eva = model.evaluate_generator(test_generator, 80)

   # print(eva)

def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    '''data generator for fit_generator'''
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)

if __name__ == '__main__':
    _main()