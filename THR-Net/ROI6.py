from cv2 import cv2
import os
import numpy as np
from keras_preprocessing import image
from sklearn.preprocessing import normalize
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from keras import backend as K
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from keras import metrics,losses

fps=25

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def loaddata(filearr):

    featurearr = []
    labelarr = []
    updategtarr=[]
    for j in range(filearr.shape[0]):
        videofolder = filearr[j][1]
        count = int(filearr[j][2])
        if os.path.exists(videofolder):
            print(videofolder)
            recording = []

            for i in range(int(fps * (13 + 10 * (count))), int(fps * (23 + 10 * (count)))-2):

                noseimg = cv2.imread(os.path.join(videofolder, 'thermal_nose_{0}.jpg'.format(i)))
                noseimg = cv2.resize(noseimg, (65, 54))
                avg_nose = np.zeros((int(54 / 4) * int(65 / 4), 3))
                avg_nose_count = 0

                for p in range(int(54 / 4)):
                    for q in range(int(65 / 4)):
                        nose_clip = noseimg[p * 4:(p + 1) * 4, q * 4:(q + 1) * 4]
                        avg_nose[avg_nose_count, 0] = np.mean(nose_clip[:, :, 0])
                        avg_nose[avg_nose_count, 1] = np.mean(nose_clip[:, :, 1])
                        avg_nose[avg_nose_count, 2] = np.mean(nose_clip[:, :, 2])
                        avg_nose_count += 1

                recording.append(avg_nose)

            recording = np.array(recording)

            # recording = recording.reshape((4, int(recording.shape[0] /4), recording.shape[1], recording.shape[2]))

            recording = recording.transpose((1,0,2))
            print(recording.shape)
            recording= cv2.resize(recording, (int(recording.shape[1]/2),int(recording.shape[0]/2)))
            recording = cv2.resize(recording, (124, 130))
            print(recording.shape)

            featurearr.append(recording)
            labelarr.append(float(filearr[j][3]))
            updategtarr.append(filearr[j])

    featurearr = np.array(featurearr)
    labelarr = np.array(labelarr)
    updategtarr=np.array(updategtarr)

    return featurearr, labelarr,updategtarr

def pearson_r(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x, axis=0)
    my = K.mean(y, axis=0)
    xm, ym = x - mx, y - my
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num / r_den
    return K.mean(r)

def label_preprocessing(labelarr):

    newlabel = np.zeros((labelarr.shape[0], 1))
    for i in range(labelarr.shape[0]):
        newlabel[i,:] = labelarr[i]
    print("新label维度", newlabel.shape)
    return newlabel

def img_preprocessing(timearr):

    timearr = timearr.astype('float32')
    timearr /= 255
    # mean= timearr.mean(axis=0)
    # timearr -= mean
    return timearr


def LA_block(x,filters, kernel_size,strides):
    x = Conv3D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', strides=strides)(x)
    y=Permute((1,4,3,2))(x)
    y=GlobalAveragePooling3D()(y)
    num=K.int_shape(y)[1]
    y = Dense(units=128)(y)
    y= Activation('relu')(y)
    y = Dense(units=num)(y)
    y = Activation('sigmoid')(y)
    y = Reshape((1, K.int_shape(y)[1], 1, 1))(y)
    x=multiply([x,y])
    x=BatchNormalization()(x)
    return x




def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def TA_block(x,nb_filter, kernel_size,strides):
    x = Conv2d_BN(x,nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    tempchannelnum=K.int_shape(x)[-2]
    y=Permute((1,3,2))(x)
    kernalsize = K.int_shape(y)[2]
    kernalnum=K.int_shape(y)[-1]
    y=Reshape((K.int_shape(y)[1]*K.int_shape(y)[2],K.int_shape(y)[3]))(y)
    # num = K.int_shape(y)[2]
    y=Conv1D(filters=kernalnum,kernel_size=kernalsize, strides=kernalsize,activation='relu')(y)
    y=GlobalAveragePooling1D()(y)

    y = Dense(units=64)(y)
    y= Activation('relu')(y)
    y = Dense(units=tempchannelnum)(y)
    y = Activation('sigmoid')(y)
    y = Reshape((1, K.int_shape(y)[1], 1))(y)
    z=multiply([x,y])
    x=add([x,z])
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        att=TA_block(x, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, inpt])
        x=add([x, att])
        return x



def models(labelarr,imgarr):
    pretrainmodel = load_model('/home/som/lab/seed-yzj/newpaper4/laboratory/model/labotary_nose_att.hdf5',
                               compile=False)
    pretrainmodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error,
                          metrics=[metrics.MeanAbsoluteError(),
                                   metrics.MeanAbsolutePercentageError(), metrics.RootMeanSquaredError(), pearson_r])

    x_train, x_test, y_train, y_test = train_test_split(imgarr, labelarr, test_size=0.4, random_state=3)
    testlen = x_test.shape[0]
    x_val = x_test[:int(testlen / 2)]
    y_val = y_test[:int(testlen / 2)]
    x_test = x_test[int(testlen / 2):]
    y_test = y_test[int(testlen / 2):]

    # input
    input = Input(shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]))

    x = ZeroPadding2D((3, 3))(input)
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # (56,56,64)
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
    # (28,28,128)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
    # (14,14,256)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
    # (7,7,512)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(3, 3))(x)
    x = Flatten()(x)

    fc6 = Dense(units=labelarr.shape[1], weights=pretrainmodel.get_layer('dense_27').get_weights(), trainable=False)(x)

    model = Model(input=input, output=fc6)
    model.summary()

    # model = multi_gpu_model(model, gpus=4)

    model.compile(optimizer=Adam(lr=1e-4),loss=losses.mean_squared_error, metrics=[metrics.MeanSquaredError(),
                                                                metrics.MeanAbsoluteError(),
                                                                metrics.MeanAbsolutePercentageError(),
                                                                metrics.RootMeanSquaredError(),pearson_r])

    model_checkpoint = ModelCheckpoint('/home/som/lab/seed-yzj/newpaper4/laboratory/model/labotary_nose_thermal_att.hdf5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto',factor=0.7,min_lr=1e-6)

    # csv_logger = CSVLogger('training.csv')

    model.fit(x_train, y_train,batch_size=32,epochs=200,verbose=1,callbacks=[model_checkpoint,reduce_lr],
                        validation_data=(x_val, y_val))

    return 0

def loaddata1(filearr,imgnpy):
    featurearr = np.load(imgnpy)
    labelarr=[]
    for j in range(filearr.shape[0]):
        labelarr.append(float(filearr[j][3]))
    labelarr = np.array(labelarr)

    return featurearr, labelarr

if __name__ == '__main__':
    labelnpy = '/home/som/lab/seed-yzj/newpaper4/laboratory/datanpy/hr-gt-each10s-laboratory.npy'
    imgnpy='/home/som/lab/seed-yzj/newpaper4/laboratory/datanpy/labotary-nose-thermal-2D.npy'

    filearr = np.load(labelnpy)

    if os.path.exists(imgnpy):
        dataarr, labelarr = loaddata1(filearr,imgnpy)
    else:
        dataarr, labelarr,updategtarr = loaddata(filearr)
        np.save(imgnpy, dataarr)
        np.save(labelnpy,updategtarr)

    labelarr = label_preprocessing(labelarr)
    print('标签',labelarr)
    dataarr= img_preprocessing(dataarr)
    print('输入维度',dataarr.shape)
    models(labelarr, dataarr)