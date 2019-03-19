import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.callbacks import Callback
import tensorflow as tf
from model.core import preprocess_true_boxes, yolo_loss, yolo_head,box_iou
from model.mobilenet import yolo_body
from model.yolo3 import tiny_yolo_body
from model.utils  import get_random_data


class AveragePrecision(Callback):
        def __init__(self, val_data ,batch_size,input_shape,num_layers,anchors,num_classes):
            super().__init__()
            self.validation_data = val_data
            self.batch_size = batch_size
            self.input_shape = input_shape
            self.num_layers = num_layers
            self.anchors = anchors
            self.num_classes = num_classes

        def on_epoch_begin(self, epoch, logs={}):
            self.losses = []
            #print(self.validation_data)
            #print( self.caller("b") )
            #print( self.batch_size )
        
        def on_epoch_end(self, epoch, logs={}):
            #self.losses.append(logs.get('loss'))
            #print(   K.shape( self.model.input[0] )[0]  )
            
            batch_map = []
            for b in range(self.batch_size):
                layers_map = []
                print("batch" + str(b) )
                val_dat , zeros = next( self.validation_data )
                image_data = val_dat[0] 
                true_label = val_dat[1:4] if self.num_layers==3 else val_dat[1:3]

                for l in range(self.num_layers):
                    #print(self.num_layers)
                    print("layer" + str(l) )
                    arrp = true_label[l]
                    box = np.where(arrpl[...,4] > 0 )
                    box = np.transpose(box)
                    #print(box)
                    #print(len(box))
                    if( len(box) > 0 ):
                        testmodel =  Model(  self.model.layers[0].input ,  self.model.layers[-7+l].output  )
                        pred_output= testmodel.predict( image_data )
                        print(pred_output.shape)
                        
                        pred_output = tf.Variable(pred_output) 

                        image_input = Input(self.input_shape)

                        anchor_mask = [[6,7,8], [3,4,5], [0,1,2]] if self.num_layers==3 else [[3,4,5], [1,2,3]]

                        pred_xy, pred_wh , pred_conf , pred_class = yolo_head( pred_output ,self.anchors[anchor_mask[l]], self.num_classes, self.input_shape, calc_loss=False)
                        pred_box = K.concatenate([pred_xy, pred_wh])
                        
                        object_mask = true_label[l][..., 4:5]
                        object_mask_bool = K.cast(object_mask, 'bool')
                        true_box = tf.boolean_mask(true_label[l][0,...,0:4], object_mask_bool[0,...,0])
                        iou = box_iou(pred_box, true_box)
                        best_iou = K.max(iou, axis=-1)
                        
                        #convert tf variable to real varable
                        with tf.Session() as sess:
                            init = tf.global_variables_initializer()
                            sess.run(init)
                            
                            #pred_box = pred_box.eval()
                            pred_conf = pred_conf.eval()
                            pred_class = pred_class.eval()
                            best_iou = best_iou.eval()
                        
                        iou_thres = 0.5
                        conf_thres = 0.5
                        false_positives = np.zeros((0,))
                        true_positives  = np.zeros((0,))
            
                        for i in range(len(box)):
                            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                            print( "({})".format(box[i]) )
                            print( arrp[tuple(box[i])][0:5] )
                            print( arrp[tuple(box[i])][5:25] )
                            true_label =  np.argmax( arrp[tuple(box[i])][5:25]) 
                            print( "{} = {}".format(true_label, obj[ true_label ] ) )
                            print("-------------------------------------------------------")
                            print( pred_box[ tuple(box[i]) ] )
                            print( pred_conf[ tuple(box[i]) ] )
                            print( pred_class[ tuple(box[i]) ] )
                            pred_label =  np.argmax( pred_class[tuple(box[i])]) 
                            print( "{} = {}".format(pred_label, obj[ pred_label] ) )
                            print("-------------------------------------------------------")
                            if( best_iou[tuple(box[i])] > iou_thres and  pred_conf[tuple(box[i])] > conf_thres and (true_label and pred_label) ):
                                #print( best_iou[tuple(box[i])] )
                                false_positives = np.append(false_positives, 0)
                                true_positives   = np.append(true_positives, 1)
                            else:
                                false_positives = np.append(false_positives, 1)
                                true_positives  = np.append(true_positives, 0)
                            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

                            
            #print( image_data.shape )
            #print( image_data[0,0,0] )
            #print( true_label[0].shape )
            #print( true_label[1].shape )
            #print( true_label[2].shape )

            #image_input = Input(self.input_shape)

            #testmodel =  Model(  self.model.get_layer('input_1').input , self.model.get_layer('conv2d_23').output )
            #testmodel =  Model(  self.model.layers[0].input , [ self.model.layers[-7].output, self.model.layers[-6].output,self.model.layers[-5].output ] )
            #res= testmodel.predict( image_data )
            #print( res[0].shape )
            #print( res[1].shape )
            #print( res[2].shape )
            #print( res )
            
            #with tf.Session() as sess:
            #    init = tf.global_variables_initializer()
            #    sess.run(init)
                #print(pred_box.eval().shape)
           #     print ( self.model.get_layer('conv2d_23').output.eval() )
           #     print ( self.model.get_layer('conv2d_23').output.eval().shape )
           # self.batch_size +=1

        #def caller(self , cel):
        #    return cel