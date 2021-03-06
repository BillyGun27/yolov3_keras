"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping , LambdaCallback
import tensorflow as tf
from model.core import preprocess_true_boxes, yolo_loss
from model.yolo3 import yolo_body
from model.yolo3 import tiny_yolo_body
from model.utils  import get_random_data
from model.evaluation import AveragePrecision

import argparse

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

    input_shape = (416,416) # multiple of 32, hw

    is_tiny_version = len(anchors)==6 # default setting
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/xtiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
            freeze_body=2, weights_path='model_data/trained_weights_final.h5') #trained_weights_final_mobilenetv2.h5 # make sure you know what you freeze

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'mobep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # Print the batch number at the beginning of every batch.
    #batch_print_callback = LambdaCallback(on_epoch_end=lambda batch,logs: print(batch))

    # Function to display the target and prediciton
    i=0
    def testmodel(epoch, logs):
        #logito = next(data_iterator)
        '''
        predout = model.predict(
            logito[0],
            batch_size=1
        )

        print("Input\n")
        print(predx)
        print("Target\n")
        print(predy)
        print("Prediction\n")
        print(predout)
        '''
        #print(logs)
        #print(i)
        #i+=1

    # Callback to display the target and prediciton
    testmodelcb = keras.callbacks.LambdaCallback(on_epoch_end=testmodel)


    class LossHistory(keras.callbacks.Callback):
        def __init__(self, val_data, batch_size = 20):
            super().__init__()
            self.validation_data = val_data
            self.batch_size = batch_size

        def caller(self , cel):
            return cel

        def on_epoch_begin(self, epoch, logs={}):
            self.losses = []
            #print(self.validation_data)
            print( self.caller("b") )
            print( self.batch_size)

        def on_epoch_end(self, epoch, logs={}):
            #self.losses.append(logs.get('loss'))
            #print(   K.shape( self.model.input[0] )[0]  )
            valdar = next( self.validation_data )
            print( valdar[0][0].shape )

            image_input = Input(shape=(None, None, 3))

            #model_body = yolo_body(image_input, num_anchors//3, num_classes)
            #model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)

            #self.model.layers.pop()
            self.model.summary()
            #testmodel =  Model(  self.model.get_layer('input_1').input , self.model.get_layer('conv2d_23').output )
            testmodel =  Model(  self.model.layers[0].input , self.model.layers[-5].output )
            res= testmodel.predict( valdar[0][0] )
            print( res.shape )
            print( res )
            
            #with tf.Session() as sess:
            #    init = tf.global_variables_initializer()
            #    sess.run(init)
                #print(pred_box.eval().shape)
           #     print ( self.model.get_layer('conv2d_23').output.eval() )
           #     print ( self.model.get_layer('conv2d_23').output.eval().shape )
            #self.batch_size +=1


    with open(train_path) as f:
        train_lines = f.readlines()

    with open(val_path) as f:
        val_lines = f.readlines()

   # with open(test_path) as f:
   #     test_lines = f.readlines()

    num_train = 4# int(len(train_lines))
    num_val = 4 #int(len(val_lines))

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        model.compile(optimizer=Adam(lr=1e-3), loss={
            # use custom yolo_loss Lambda layer.
            'yolo_loss': lambda y_true, y_pred: y_pred}
            ,metrics=[mean_iou]
            )

        batch_size = 1#32

        meanAP = AveragePrecision(data_generator_wrapper(val_lines, 1 , input_shape, anchors, num_classes) ,batch_size, input_shape , len(anchors)//3 , anchors ,num_classes)
        
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                steps_per_epoch=max(1, num_train//batch_size),
                validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes),
                validation_steps=max(1, num_val//batch_size),
                epochs=50,
                initial_epoch=0,
                callbacks=[logging, checkpoint , meanAP ])
        model.save_weights(log_dir + 'trained_weights_stage_1_mobilenetv2.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if False:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred}) # recompile to apply the change
        print('Unfreeze all of the layers.')

        batch_size =  8#32 note that more GPU memory is required after unfreezing the body
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors, num_classes),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=50,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final_mobilenetv2.h5')

    # Further training if needed.
def mean_iou( y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return y_true


'''
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        #y_predict = np.asarray(model.predict(X_val))

        #y_val = np.argmax(y_val, axis=1)
        #y_predict = np.argmax(y_predict, axis=1)
        
        self._data.append({
            'val_rocauc': roc_auc_score(y_val, y_predict),
        })
        
        print(X_val)

    def get_data(self):
        return self._data
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


def create_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
            weights_path='model_data/yolo_weights.h5'):
    '''create the training model'''
    K.clear_session() # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors//3, num_classes)
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


    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

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

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model

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
