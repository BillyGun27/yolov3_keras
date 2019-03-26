"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import Input, Lambda ,Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from model.core import preprocess_true_boxes,yolo_head
from model.distillation import apprentice_distill_loss
from model.mobilenet import mobilenetv2_yolo_body
from model.yolo3 import yolo_body, tiny_yolo_body
from model.utils  import get_random_data
from model.evaluation import AveragePrecision

import argparse

def _main():
    train_path = '2007_train.txt'
    val_path = '2007_val.txt'
   # test_path = '2007_test.txt'
    log_dir = 'logs/with_loss_000/'
    classes_path = 'class/voc_classes.txt'
    anchors_path = 'anchors/yolo_anchors.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    input_shape = (416,416) # multiple of 32, hw
    
  
    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/trained_weights_final_mobilenetv2.h5') # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    with open(train_path) as f:
        train_lines = f.readlines()

    with open(val_path) as f:
        val_lines = f.readlines()

# with open(test_path) as f:
#     test_lines = f.readlines()

# train_lines = np.load('train_logits.npy')[()]
# val_lines = np.load('val_logits.npy')[()]

    num_val = int(len(train_lines))
    num_train = int(len(val_lines))

    #declare model
    num_anchors = len(anchors)
    image_input = Input(shape=(416, 416, 3))
    teacher = yolo_body(image_input, num_anchors//3, num_classes)
    teacher.load_weights("model_data/trained_weights_final.h5")
    
    # return the constructed network architecture
    # class+5
    yolo3 = Reshape((13, 13, 3, 25))(teacher.layers[-3].output)
    yolo2 = Reshape((26, 26, 3, 25))(teacher.layers[-2].output)
    yolo1 = Reshape((52, 52, 3, 25))(teacher.layers[-1].output)
    
    teacher = Model( inputs= teacher.input , outputs=[yolo3,yolo2,yolo1] )
    teacher._make_predict_function()
    
    '''    
    batch_size = 1
    v_dat , dero = next(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes,teacher))
    obj = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    arrpmod = v_dat[1]
    boxmod = np.where(arrpmod[...,4] > 0.3 )
    boxmod = np.transpose(boxmod)
    print(boxmod)

    box = boxmod
    arrp= arrpmod
    for i in range(len(box)):

        print( "({})".format(box[i]) )
        print( "true" )
        print( v_dat[1][tuple(box[i])][0:5] )
        print( v_dat[1][tuple(box[i])][5:25] )
        true_label =  np.argmax( v_dat[1][tuple(box[i])][5:25]) 
        print( "{} = {}".format(true_label, obj[ true_label ] ) )
        print("-------------------------------------------------------")
        print( "({})".format(box[i]) )
        print( "modified model" )
        print( v_dat[4][tuple(box[i])][0:5] )
        print( v_dat[4][tuple(box[i])][5:25] )
        true_label =  np.argmax( v_dat[4][tuple(box[i])][5:25]) 
        print( "{} = {}".format(true_label, obj[ true_label ] ) )
        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    
    print(v_dat[4])
    '''   
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'apprentice_distill_loss': lambda y_true, y_pred: y_pred})
    
        batch_size = 16#32

        meanAP = AveragePrecision(data_generator_wrapper(val_lines, 1 , input_shape, anchors, num_classes ,teacher ) ,num_val,, input_shape , len(anchors)//3 , anchors ,num_classes)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes,teacher),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes,teacher),
                validation_steps=max(1, num_val//batch_size),
                epochs=30,
                initial_epoch=0,
                callbacks=[logging, checkpoint,  meanAP])
        model.save_weights(log_dir + 'distillation_loss_appr_trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'apprentice_distill_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size =  8#32 note that more GPU memory is required after unfreezing the body

        meanAP = AveragePrecision(data_generator_wrapper(val_lines, 1 , input_shape, anchors, num_classes ,teacher) ,num_val , input_shape , len(anchors)//3 , anchors ,num_classes)

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes,teacher),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes,teacher),
            validation_steps=max(1, num_val//batch_size),
            epochs=60,
            initial_epoch=30,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping ,  meanAP])
        model.save_weights(log_dir + 'distillation_loss_appr_trained_weights_final.h5')

# Further training if needed.


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

def sigmoid(x):
        """sigmoid.

        # Arguments
            x: Tensor.

        # Returns
            numpy ndarray.
        """
        return 1 / (1 + np.exp(-x))
        
def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    l_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = mobilenetv2_yolo_body(image_input, num_anchors//3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            num = (185, len(model_body.layers)-3)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))
    
    for y in range(-3, 0):
        model_body.layers[y].name = "conv2d_output_" + str(h//{-3:32, -2:16, -1:8}[y])

    model_loss = Lambda(apprentice_distill_loss, output_shape=(1,), name='apprentice_distill_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true , *l_true])
    model = Model([model_body.input, *y_true , *l_true ], model_loss)

    return model

def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/tiny_yolo_weights.h5'):
    '''create the training model, for Tiny YOLOv3'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16}[l], w//{0:32, 1:16}[l], \
        num_anchors//2, num_classes+5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors//2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers)-2)[freeze_body-1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(apprentice_distill_loss, output_shape=(1,), name='apprentice_distill_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true, *l_true])
    model = Model([model_body.input, *y_true, *l_true], model_loss)

    return model

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
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
        m_true = teacher.predict(image_data)
         
        h, w = input_shape
        num_anchors = len(anchors)
        
        l_true =  [ np.zeros( shape=( batch_size ,416//{0:32, 1:16, 2:8}[l], 416//{0:32, 1:16, 2:8}[l], 9//3, 20+5) ) for l in range(3) ]

        #print(len(y_true))
        #print(len(m_true))
        #print(len(l_true))
        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 

        for l in range( len(m_true) ) : 
            '''
            pred_output = tf.Variable(m_true[l]) 
            anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if len(m_true)==3 else [[3,4,5], [1,2,3]] 
            pred_xy, pred_wh , pred_conf , pred_class = yolo_head( pred_output ,anchors[anchor_mask[l]], num_classes, input_shape, calc_loss=False)
            pred_model = K.concatenate([pred_xy, pred_wh, pred_conf ,pred_class  ])

            with tf.Session() as sess:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                    
                    pred_model = pred_model.eval()
            '''
            anchors_tensor = np.reshape( anchors[anchor_mask[l]] , [1, 1, 1, len( anchors[anchor_mask[l]] ) , 2] )

            grid_shape = m_true[l].shape[1:3] # height, width
            grid_shape
            grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                [1, grid_shape[1], 1, 1])
            grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                [grid_shape[0], 1, 1, 1])
            grid = np.concatenate([grid_x, grid_y],axis=-1)

            #print(l)
            m_true[l][...,:2] = (sigmoid(m_true[l][...,:2]) + grid ) / np.array( grid_shape[::-1] )
            m_true[l][..., 2:4] = np.exp(m_true[l][..., 2:4]) * anchors_tensor  / np.array( input_shape[::-1] )
            m_true[l][..., 4] = sigmoid(m_true[l][..., 4])
            m_true[l][..., 5:] = sigmoid(m_true[l][..., 5:])
            
            #print("inside")
            box = np.where(y_true[l][...,4] > 0.5 )
            box = np.transpose(box)

            for i in range(len(box)):
                l_true[l][tuple(box[i])] = m_true[l][tuple(box[i])] #pred_model[tuple(box[i])]
        
        
        yield [image_data, *y_true , *l_true ], np.zeros(batch_size)

def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher):
    n = len(annotation_lines)
    if n==0 or batch_size<=0: return None
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes,teacher)

if __name__ == '__main__':
    _main()
