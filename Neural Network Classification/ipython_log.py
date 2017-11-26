# IPython log file

from keras.models import load_model
model = load_model('fullmodel3cl50epVGG19.h5')
model.summary()
from keras.prepreocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
img_width, img_height = 256
img_width, img_height = 256,256
train_data_dir = "data/train"
train_data_dir = "data/test"
from os import listdir
len(listdir('data/train'))
#[Out]# 3
len(listdir('data/train/apple') + listdir('data/train/banana') + listdir('data/train/hamburger'))
#[Out]# 3581
len(listdir('data/train/apple'))
#[Out]# 1100
num_train = 3581
len(listdir('data/test/apple') + listdir('data/test/banana') + listdir('data/test/hamburger'))
#[Out]# 982
num_test = 982
epochs =50
from keras import applications
newmodel = applications.VGG19(weights="imagenet",include_top=False,input_shape=(img_width,img_height))
newmodel = applications.VGG19(weights="imagenet",include_top=False,input_shape=(img_width,img_height,3))
newmodel.summary()
for layer in model.layers[:5]:
    layer.trainable=False
for layer in newmodel.layers[:5]:
    layer.trainable=False
newmodel.summary()
for layer in newmodel.layers:
    layer.trainable=False
newmodel.summary())
newmodel.summary()
x=newmodel.output
x = Flatten()(x)
x = Dense(3,activation="relu")(x)
x = Dropout(0.5)(x)
x=Dense(3,activation="relu")(x)
from keras.regularizers import l2
x = l2(0.01)(x)
predictions = Dense(3,activation="softmax")(x)
y = newmodel.output
y = Flatten()(y)
y = Dense(3,activation="relu")(y)
y = Dropout(0.5)(y)
y = Dense(3,activation="relu")(y)
predictions = Dense(3,activation="softmax",W_regularizer=l2(0.01))(y)
model_final = Model(input=newmodel.input,output=predictions)
model_final.summary()
model_final.compile(loss="hinge",optimizer=optimizers.SGD(lr=0.0001,momentum=0.9),metrics=["accuracy"])
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)
batch_size=32
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")
test_data_dir = "data/test"
train_data_dir = "data/train"
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
model_final.fit_generator(
train_generator,
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = num_test,
callbacks = [checkpoint, early])
model_final.summary()
batch_size=16
model_final.fit_generator(
train_generator,
batch_size=batch_size
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = num_test,
callbacks = [checkpoint, early])
model_final.fit_generator(
train_generator,
batch_size=batch_size,
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = num_test,
callbacks = [checkpoint, early])
model_final.fit_generator(
train_generator,
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = num_test,
callbacks = [checkpoint, early])
train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")
model_final.fit_generator(
train_generator,
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = num_test,
callbacks = [checkpoint, early])
resnet = keras.applications.resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=(256,256,3))
from keras.applications import resnet50
resnet = ResNet50(include_top=False,weights='imagenet'input_shape=(256,256,3))
resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(256,256,3))
resnet = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=(256,256,3))
resnet.summary()
for layer in resnet.layers:
for layer in resnet.layers:
    layer.trainable=False

x=resnet.output
x=Flatten()(x)
x=Dense(3,activation="relu")(x)
x=Dropout(0.5)(x)
x=Dense(3,activation="relu")(x)
predictions = Dense(3,activation="softmax",W_regularizer=l2(0.01))(x)
resnet_final = Model(input=resnet.input,output=predictions)
resnet_final.compile(loss='hinge',optimizer=optimizers.SGD(lr=0.0001,momentum=0.9),metrics=["accuracy"])
batch_size=32
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
test_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")
checkpoint = ModelCheckpoint("resnet_3cl.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')
model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early])
model_final.fit_generator(
train_generator,
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nunm_test = nb_validation_samples,
callbacks = [checkpoint, early])
model_final.fit_generator(
train_generator,
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = num_test,
callbacks = [checkpoint, early])
resnet_final.fit_generator(
train_generator,
samples_per_epoch = num_train,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = num_test,
callbacks = [checkpoint, early])
resnet = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=(256,256,3))
extModel = Sequential()
extModel.add(Dense(3,activation="relu",W_regularizer=l2(0.001))
extModel.add(Dense(3,activation="relu",W_regularizer=l2(0.001)))
resnet.output
#[Out]# <tf.Tensor 'avg_pool_1/AvgPool:0' shape=(?, 1, 1, 2048) dtype=float32>
resnet.add(InputLayer(None,1,1,2048))
extModel.add(InputLayer(1,1,2048))
from keras.layers import InputLayer
extModel.add(InputLayer(1,1,2048))
extModel.add(InputLayer((1,1,2048)))
extModel.summary()
extModel.add(Flatten())
extModel.add(Dense(3,activation="relu",W_regularizer=l2(0.001)))
extModel.summary()
extModel.compile(loss='hinge',optimizer=optimizers.SGD(lr=0.0001,momentum=0.9),metrics=["accuracy"])
extModel=Sequential()
extModel.add(InputLayer((1,1,2048)))
extModel.add(Flatten())
extModel.add(Dense(3,activation="softmax",W_regularizer=l2(0.001)))
extModel.compile(loss='hinge',optimizer=optimizers.SGD(lr=0.0001,momentum=0.9),metrics=["accuracy"])
extModel.summary()
import numpy as np
apple_generator =  train_generator = train_datagen.flow_from_directory(
'data/train/apple',
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")
apple_generator =  train_generator = train_datagen.flow_from_directory(
'data/train/apple',
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode=None)
apple_generator =  train_generator = train_datagen.flow_from_directory(
'data/train',
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode=None)
apple_generator =  train_generator = train_datagen.flow_from_directory(
'data/train/apple',
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode=None)
banana_generator =  train_generator = train_datagen.flow_from_directory(
'data/train/banana',
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode=None)
hamburger_generator =  train_generator = train_datagen.flow_from_directory(
'data/train/hamburger',
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode=None)
apples = resnet.predict_generator(apple_generator,3000)
apples = resnet.predict_generator(apple_generator,10)
apples.shape
#[Out]# (320, 1, 1, 2048)
apples.reshape((320,2048))
#[Out]# array([[ 0.        ,  0.        ,  0.00648986, ...,  1.87498069,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.00576222, ...,  1.66337895,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.01023097, ...,  1.56111908,
#[Out]#          0.        ,  0.        ],
#[Out]#        ..., 
#[Out]#        [ 0.        ,  0.        ,  0.00959528, ...,  1.51725924,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.00717544, ...,  1.4454571 ,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.01027201, ...,  1.87963963,
#[Out]#          0.        ,  0.        ]], dtype=float32)
apples.shape
#[Out]# (320, 1, 1, 2048)
apples = resnet.predict_generator(apple_generator,100)
apples.shape
#[Out]# (3140, 1, 1, 2048)
apples = np.reshape(apples,(3140,2048))
apples.shape
#[Out]# (3140, 2048)
np.max(apples[0])
#[Out]# 11.17968
np.mean(apples)
#[Out]# 0.14813209
np.argmax(apples[0])
#[Out]# 851
np.mean(apples[0])
#[Out]# 0.14941689
maxes = np.argmax(apples,axis=1)
maxes.shape
#[Out]# (3140,)
un = np.unique(maxes)
un
#[Out]# array([249, 851], dtype=int64)
apples.save('apple_features')
np.save('apples.npy',apples)
resnet.input_shape
#[Out]# (None, 256, 256, 3)
s = resnet.input_shape
s[1:3]
#[Out]# (256, 256)
apples = resnet.predict_generator(apple_generator,1)
apples.shape
#[Out]# (32, 1, 1, 2048)
import TestClassifier as tf
import TestClassifier as tc
tc.generate_features_from_directory('data/train',10)
from importlib import reload
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
from TestClassifier import generate_features_from_directory
generate_features_from_directory('data/train',10)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
generate_features_from_directory('data/train',10)
tc.generate_features_from_directory('data/train',10)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
tc.generate_features_from_directory('data/train',10)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
tc.generate_features_from_directory('data/train',10)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
tc.generate_features_from_directory('data/train',1)
tc.generate_features_from_directory('data/train',10)
from os.path import listdir
from os import listdir
listdir('data/train')
#[Out]# ['apple', 'banana', 'hamburger']
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
tc.generate_features_from_directory('data/train',10)
app = np.load('data/train/apples.npy')
app.shape
#[Out]# (32, 1, 1, 2048)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
tc.generate_features_from_directory('data/train',3000)
arr1 = [1,2,3]
arr2 = [2,3,4]
arr3 = [3,4,5]
arrays = [arr1,arr2,arr3]
x = np.append(arr for arr in arrays)
x = np.concatenate(arr for arr in arrays)
x = np.concatenate([arr for arr in arrays])
x
#[Out]# array([1, 2, 3, 2, 3, 4, 3, 4, 5])
x = np.concatenate([arr for arr in arrays],axis=0)
x
#[Out]# array([1, 2, 3, 2, 3, 4, 3, 4, 5])
arr1 = np.array([1,2,3])
arr1
#[Out]# array([1, 2, 3])
arr2 = np.array([2,3,4])
arr3 = np.array([3,4,5])
arrays = [arr1,arr2,arr3]
arrays
#[Out]# [array([1, 2, 3]), array([2, 3, 4]), array([3, 4, 5])]
x = np.concatenate([arr for arr in arrays])
x
#[Out]# array([1, 2, 3, 2, 3, 4, 3, 4, 5])
x = np.concatenate([arr for arr in arrays],axis=0)
x
#[Out]# array([1, 2, 3, 2, 3, 4, 3, 4, 5])
x = np.concatenate([arr for arr in arrays],axis=1)
x = np.vstack([arr for arr in arrays])
x
#[Out]# array([[1, 2, 3],
#[Out]#        [2, 3, 4],
#[Out]#        [3, 4, 5]])
l = np.array([0,1])
l2 = [l]*3
l2
#[Out]# [array([0, 1]), array([0, 1]), array([0, 1])]
l3 = np.vstack([l]*4)
l3
#[Out]# array([[0, 1],
#[Out]#        [0, 1],
#[Out]#        [0, 1],
#[Out]#        [0, 1]])
label_names = ['apple','banana','ham']
with open("labels.txt","w") as output:
    output.write([label + '\n' for label in label_names])
with open("labels.txt","w") as output:
    output.write(str.join([label + '\n' for label in label_names]))
with open("labels.txt","w") as output:
    output.write("".join([label + '\n' for label in label_names]))
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
X,y,labels = tc.create_training_set('data/train')
labelcodes = np.load('label_codes.npy')
labelcodes.shape
#[Out]# (8889, 3)
y.shape
#[Out]# (8889, 3)
y[0]
#[Out]# array([ 1.,  0.,  0.])
labelcodes[0]
#[Out]# array([ 1.,  0.,  0.])
X.shape
#[Out]# (8889, 1, 1, 2048)
x.reshape(8889,2048)
X.reshape(8889,2048)
#[Out]# array([[ 0.        ,  0.        ,  0.00666931, ...,  1.89847219,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.01179638, ...,  1.52907813,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.00756826, ...,  1.73048151,
#[Out]#          0.        ,  0.        ],
#[Out]#        ..., 
#[Out]#        [ 0.        ,  0.        ,  0.0352179 , ...,  1.68864942,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.01412456, ...,  1.6645937 ,
#[Out]#          0.        ,  0.        ],
#[Out]#        [ 0.        ,  0.        ,  0.00722377, ...,  1.68252814,
#[Out]#          0.        ,  0.        ]], dtype=float32)
X.shape
#[Out]# (8889, 1, 1, 2048)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
tc.generate_features_from_directory('data/train',3000)
tc.create_training_set('data/train')
#[Out]# (array([[ 0.        ,  0.        ,  0.01065386, ...,  1.59584415,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01274862, ...,  2.08245325,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.00753007, ...,  1.66746128,
#[Out]#           0.        ,  0.        ],
#[Out]#         ..., 
#[Out]#         [ 0.        ,  0.        ,  0.        , ...,  1.73320055,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.06124046, ...,  1.90971136,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01843298, ...,  1.65652096,
#[Out]#           0.        ,  0.        ]], dtype=float32), array([[ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         ..., 
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.]]), array([[ 1.,  0.,  0.],
#[Out]#         [ 0.,  1.,  0.],
#[Out]#         [ 0.,  0.,  1.]]))
tc.create_training_set('data/train')
#[Out]# (array([[ 0.        ,  0.        ,  0.01065386, ...,  1.59584415,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01274862, ...,  2.08245325,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.00753007, ...,  1.66746128,
#[Out]#           0.        ,  0.        ],
#[Out]#         ..., 
#[Out]#         [ 0.        ,  0.        ,  0.        , ...,  1.73320055,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.06124046, ...,  1.90971136,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01843298, ...,  1.65652096,
#[Out]#           0.        ,  0.        ]], dtype=float32), array([[ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         ..., 
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.]]), array([[ 1.,  0.,  0.],
#[Out]#         [ 0.,  1.,  0.],
#[Out]#         [ 0.,  0.,  1.]]))
tc.create_training_set('data/train','data')
#[Out]# (array([[ 0.        ,  0.        ,  0.01065386, ...,  1.59584415,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01274862, ...,  2.08245325,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.00753007, ...,  1.66746128,
#[Out]#           0.        ,  0.        ],
#[Out]#         ..., 
#[Out]#         [ 0.        ,  0.        ,  0.        , ...,  1.73320055,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.06124046, ...,  1.90971136,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01843298, ...,  1.65652096,
#[Out]#           0.        ,  0.        ]], dtype=float32), array([[ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         ..., 
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.]]), array([[ 1.,  0.,  0.],
#[Out]#         [ 0.,  1.,  0.],
#[Out]#         [ 0.,  0.,  1.]]))
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
tc.create_training_set('data/train','data')
#[Out]# (array([[ 0.        ,  0.        ,  0.01065386, ...,  1.59584415,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01274862, ...,  2.08245325,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.00753007, ...,  1.66746128,
#[Out]#           0.        ,  0.        ],
#[Out]#         ..., 
#[Out]#         [ 0.        ,  0.        ,  0.        , ...,  1.73320055,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.06124046, ...,  1.90971136,
#[Out]#           0.        ,  0.        ],
#[Out]#         [ 0.        ,  0.        ,  0.01843298, ...,  1.65652096,
#[Out]#           0.        ,  0.        ]], dtype=float32), array([[ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         [ 1.,  0.,  0.],
#[Out]#         ..., 
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.],
#[Out]#         [ 0.,  0.,  1.]]), array([[ 1.,  0.,  0.],
#[Out]#         [ 0.,  1.,  0.],
#[Out]#         [ 0.,  0.,  1.]]))
X = np.load('training_set.npy')
y = np.load('label_codes.npy')
labels = [word for line in open('data/labels.txt','r')]
labels = [line for line in open('data/labels.txt','r')]
labels
#[Out]# ['apple\n', 'banana\n', 'hamburger\n']
labels = [label.split('\n')[0] for label in labels]
labels
#[Out]# ['apple', 'banana', 'hamburger']
extModel = Sequential()
extModel.add(InputLayer(input_shape=(None,1,1,2048)))
extModel.add(Flatten())
extmodel = Sequential()
extModel.add(InputLayer(input_shape=(1,1,2048)))
extModel.add(Flatten())
extmodel.add(InputLayer(input_shape=(1,1,2048)))
extmodel.add(Flatten())
extmodel.summary()
resnet.summary()
reload(tc)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
train_dir = 'data/train'
val_dir = 'data/test'
train_size = 3000
val_size = 300
output_dir = 'data/output'
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
model = tc.train_classifier_from_directory(train_dir,train_size,val_dir,val_size,output_dir)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
model = tc.train_classifier_from_directory(train_dir,train_size,val_dir,val_size,output_dir)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
model = tc.train_classifier_from_directory(train_dir,train_size,val_dir,val_size,output_dir)
reload(tc)
#[Out]# <module 'TestClassifier' from 'E:\\KPImageProcessing\\KP-ImageProcessing\\Neural Network Classification\\TestClassifier.py'>
model = tc.train_classifier_from_directory(train_dir,train_size,val_dir,val_size,output_dir)
