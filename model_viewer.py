import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import Input, Lambda ,Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.core import preprocess_true_boxes, yolo_loss
from model.mobilenet import mobilenetv2_yolo_body
from model.yolo3 import yolo_body, tiny_yolo_body
from model.utils  import get_random_data
from keras.utils.vis_utils import plot_model as plot
from model.squeezenet import squeezenet_body,squeezenet_yolo_body

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
    
train_path = '2007_train.txt'
val_path = '2007_val.txt'
# test_path = '2007_test.txt'
log_dir = 'logs/logits_only_000/'
classes_path = 'class/voc_classes.txt'
anchors_path = 'anchors/yolo_anchors.txt'
class_names = get_classes(classes_path)
num_classes = len(class_names)
anchors = get_anchors(anchors_path)

input_shape = (416,416) # multiple of 32, hw
    
  
image_input = Input(shape=(416, 416, 3))
h, w = input_shape
num_anchors = len(anchors)


#model = mobilenetv2_yolo_body(image_input, num_anchors//3, num_classes)
#plot(model, to_file='{}.png'.format("mobilenetv2_yolo"), show_shapes=True)

#squeezenet_model = squeezenet_body( input_tensor = image_input )
#squeezenet_model.summary()
squeezenet_model = squeezenet_yolo_body(image_input, num_anchors//3, num_classes)
plot(squeezenet_model , to_file='{}.png'.format("squeezenet_yolo"), show_shapes=True)


