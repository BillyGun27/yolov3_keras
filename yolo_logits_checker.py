"""
Retrain the YOLO model for your own dataset.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import keras.backend as K
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Lambda ,Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.core import yolo_head,preprocess_true_boxes, yolo_loss
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

    i = 0 #step
    for  img,flogits,mlogits in data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes,model) : 
        #x , y = dat
        #train_logits[i] = logits
        print("fakers")
        print(flogits[0].shape)

        box_xy, box_wh , box_confidence, box_class_probs = fake_head(flogits[0], anchors, num_classes, input_shape, calc_loss=False)
        print(box_xy)
        print(box_wh)
        print(box_confidence)
        print(box_class_probs)
       # for attr in flogits:

       #    print(flogits[0][0][0][0])
       
       #  print("models")
       # print(mlogits[0].shape)
       # print(flogits[0][0][0][0][0])
        print("---------------")
        

        i+=1
        if i>= 5:#(len(train_lines)) :
            break
    
    #fp = h5py.File('train_logits.h5','r')
    #print(fp[0][1].shape)
    #print(fp[1][2].shape)

    # print(train_logits[0][1].shape)
    #np.save('train_logits.npy', train_logits)
    #train_log = np.load('train_logits.npy')[()]

    #print(train_log[0][0].shape)
    #print(train_log[1][0].shape)
    '''
    print( "total "+ str(len(val_lines)) + " loop "+ str( len(val_lines)//batch_size +1 ) )
    i = 0 #step
    for  logits in data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes) : 
        #x , y = dat
        val_logits[i] = logits

        i+=1
        if i>= (len(val_lines)) :
            break
    '''


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
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
       # print(image_data.shape)
        m_true = model.predict(image_data)
     

        yield image_data , y_true , m_true  #, np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,model):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,model)

def fake_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Convert final layer features to bounding box parameters."""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    grid_shape = K.shape(feats)[1:3] # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

   # feats = K.reshape(
   #     feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

if __name__ == '__main__':
    _main()