from model.yolo3 import yolo_body, tiny_yolo_body
import keras.backend as K
from keras.layers import Input, Lambda ,Reshape
from keras.models import Model
from model.utils import get_random_data
import numpy as np

'''
with tf.Session(graph=K.get_session().graph) as session:
    session.run(tf.global_variables_initializer())
    model = load_model('model.h5')
    predictions = model.predict(input)
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

num_anchors = len(anchors)
image_input = Input(shape=(416, 416, 3))
model = yolo_body(image_input, num_anchors//3, num_classes)
model.load_weights("model_data/trained_weights_final.h5")

yolo3 = Reshape((13, 13, 3, 25))(model.layers[-3].output)
yolo2 = Reshape((26, 26, 3, 25))(model.layers[-2].output)
yolo1 = Reshape((52, 52, 3, 25))(model.layers[-1].output)


model = Model( inputs= model.input , outputs=[yolo3,yolo2,yolo1] )

sodel = Model( inputs= model.input , outputs=[yolo3,yolo2,yolo1] )

with open(train_path) as f:
        train_lines = f.readlines()

input_shape = (416,416) # multiple of 32, hw
batch_size = 1
image_data = []
box_data = []
    
image, box = get_random_data(train_lines[0], input_shape, random=True)
image_data.append(image)
box_data.append(box)
image_data = np.array(image_data)
        
m_true = model.predict(image_data)
#print(m_true)

k_true = sodel.predict(image_data)
#print(k_true)

print("combination")
xvel = m_true+k_true
print(xvel)