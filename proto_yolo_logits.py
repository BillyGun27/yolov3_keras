"""
Retrain the YOLO model for your own dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda ,Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.core import preprocess_true_boxes, yolo_loss
from model.yolo3 import yolo_body, tiny_yolo_body
from model.utils import get_random_data

from tqdm import tqdm
import time

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

    num_anchors = len(anchors)
    image_input = Input(shape=(416, 416, 3))
    model = yolo_body(image_input, num_anchors//3, num_classes)
    model.load_weights("model_data/trained_weights_final.h5")
    
    yolo3 = Reshape((13, 13, 3, 25))(model.layers[-3].output)
    yolo2 = Reshape((26, 26, 3, 25))(model.layers[-2].output)
    yolo1 = Reshape((52, 52, 3, 25))(model.layers[-1].output)
    

    model = Model( inputs= model.input , outputs=[yolo3,yolo2,yolo1] )
    
    batch_size = 1
    
    print( "total "+ str(len(train_lines)) + " loop "+ str( len(train_lines) ) )

        # create an hdf5 file
    train_size = len(train_lines)
    with h5py.File("train_logits.h5",'w') as f:
        # create a dataset for your movie
        img = f.create_dataset("img_data", shape=(  train_size, 416, 416, 3)) #len(train_lines)
        bbox = f.create_dataset("big_logits", shape=( train_size, 13, 13, 3, 25))
        mbox = f.create_dataset("medium_logits", shape=( train_size , 26, 26, 3, 25))
        sbox = f.create_dataset("small_logits", shape=(  train_size , 52, 52, 3, 25))

        # fill the 10 frames with a random image
        i = 0
        for logits in tqdm( data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes,model) ) : 
            #print(logits[0][0])
            #print(logits[0][0].shape)
            #trat = logits[0][0]

            #print("box")
            #print(logits[1][0])
            #print(logits[1][0].shape)

            img[i] = logits[0][0] # np.random.randint(255, size=(416, 416, 3)) #        
            bbox[i] = logits[1][0]
            mbox[i] = logits[2][0]
            sbox[i] = logits[3][0]
        
            i+=1
            if i>= train_size:#(len(train_lines)) :
                break


    #fp = h5py.File('train_logits.h5','r')
    #train_logits = []
    #print(fp["img_data"][0].shape)
    #print(fp["big_logits"][1].shape)
    #print(fp["medium_logits"][1].shape)

    #print("o1")
    #print(fp["img_data"][0])
    #print("o2")
    #print(fp["img_data"][1])
    #print("ddd")
    #print(trat)

    #print("b1")
    #print(fp["big_logits"][2])
    #print("b2")
    #print(fp["big_logits"][1])
    #print("ddd")
    #print(trat)


    #train_logits.append(fp["img_data"][0])
    #train_logits.append(fp["img_data"][1])

    #train_logits = np.array(train_logits)
    #print(train_logits.shape)

    val_size = len(val_lines)
    with h5py.File("val_logits.h5",'w') as f:
        # create a dataset for your movie
        img = f.create_dataset("img_data", shape=(  val_size, 416, 416, 3)) #
        bbox = f.create_dataset("big_logits", shape=( val_size, 13, 13, 3, 25))
        mbox = f.create_dataset("medium_logits", shape=( val_size, 26, 26, 3, 25))
        sbox = f.create_dataset("small_logits", shape=(  val_size, 52, 52, 3, 25))

        # fill the 10 frames with a random image
        i = 0
        for logits in tqdm( data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes,model) ) : 

            img[i] = logits[0][0] # np.random.randint(255, size=(416, 416, 3)) #        
            bbox[i] = logits[1][0]
            mbox[i] = logits[2][0]
            sbox[i] = logits[3][0]
        
            i+=1
            if i>= val_size:#(len(val_lines)) :
                break

'''
    i = 0 #step
    for  logits in data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes) : 
        #x , y = dat
        #train_logits[i] = logits
        print(logits[0].shape)
        print(logits[1].shape)
        f = h5py.File('train_logits.h5','a')
        f.create_dataset('data_logits', data=logits)
        f.close()    


        i+=1
        if i>= 5:#(len(train_lines)) :
            break
    
    fp = h5py.File('train_logits.h5','r')
    print(fp[0][1].shape)
    print(fp[1][2].shape)

    # print(train_logits[0][1].shape)
    #np.save('train_logits.npy', train_logits)
    #train_log = np.load('train_logits.npy')[()]

    #print(train_log[0][0].shape)
    #print(train_log[1][0].shape)
    
    print( "total "+ str(len(val_lines)) + " loop "+ str( len(val_lines)//batch_size +1 ) )
    i = 0 #step
    for  logits in data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes) : 
        #x , y = dat
        val_logits[i] = logits

        i+=1
        if i>= (len(val_lines)) :
            break
    

    '''
    #np.save('val_logits.npy', val_logits)

    


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


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,model):
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
        #box_data = np.array(box_data)
        #y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
       # print(image_data.shape)
        y_true = model.predict(image_data)
       # print("d")
      #  print(y_true[0].shape)
      #  print(y_true[1].shape)
      #  print(y_true[2].shape)
       # y_true[0] = y_true[0].reshape(y_true[0].shape[0], y_true[0].shape[1], y_true[0].shape[2], 3 , y_true[0].shape[3]//3 ) 
       # y_true[1] = y_true[1].reshape(y_true[1].shape[0], y_true[1].shape[1], y_true[1].shape[2], 3 , y_true[1].shape[3]//3 ) 
       # y_true[2] = y_true[2].reshape(y_true[2].shape[0], y_true[2].shape[1], y_true[2].shape[2], 3 , y_true[2].shape[3]//3 ) 

        yield [image_data, *y_true]#, np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,model):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,model)

if __name__ == '__main__':
    _main()