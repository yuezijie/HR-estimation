import os
import numpy as np
import csv
from keras_preprocessing import image
from sklearn.preprocessing import normalize
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils import multi_gpu_model
from sklearn.model_selection import train_test_split
from keras import metrics,losses

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"


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


def turntofrequencysequence(arr):
    interarr=np.zeros((arr.shape[0],int(arr.shape[1]/2)+1,arr.shape[2]))
    for i in range(arr.shape[0]):
        for j in range(arr.shape[2]):
            before_fft_signal=arr[i,:,j]
            fft = np.abs(np.fft.rfft(before_fft_signal))
            interarr[i,:,j]=fft
    return interarr

def turntime_frequencysequence(arr1,arr2):
    timelen=arr1.shape[0]
    interarr=np.zeros((arr1.shape[1]*arr1.shape[2]+arr2.shape[1]*arr2.shape[2],timelen,3))

    for i in range(timelen):
        count1 = 0
        for j in range(arr1.shape[1]):
            for k in range(arr1.shape[2]):
                interarr[count1,i,:]=arr1[i,j,k,:]
                count1+=1

    for i in range(timelen):
        count2 = arr1.shape[1] * arr1.shape[2]
        for j in range(arr2.shape[1]):
            for k in range(arr2.shape[2]):
                interarr[count2,i,:]=arr2[i,j,k,:]
                count2+=1

    timearr=interarr
    frequencyarr=turntofrequencysequence(timearr)

    return timearr,frequencyarr

def load_imgs(imgspath,num):
    foreheadarr=[]
    nosearr=[]
    imgsfolder=imgspath+'/'+num
    imgs=os.listdir(imgsfolder)
    splitimglen=len(imgs)/2
    count = 0
    while True:
        i=int(splitimglen/2)-450

        foreheadimgpath = os.path.join(imgsfolder, 'forehead_{0}.jpg'.format(i))
        noseimgpath = os.path.join(imgsfolder, 'nose_{0}.jpg'.format(i))
        if count<900:
            if os.path.exists(foreheadimgpath):
                foreheadimg = image.load_img(foreheadimgpath, target_size=(7, 9))
                foreheadimg = image.img_to_array(foreheadimg)
                foreheadarr.append(foreheadimg)
                noseimg = image.load_img(noseimgpath, target_size=(15, 31))
                noseimg = image.img_to_array(noseimg)
                nosearr.append(noseimg)
                # print('OK')
                count+=1
                i+=1
            else:
                i+=1
                continue
        else:
            break

    foreheadarr=np.array(foreheadarr)
    foreheadarr=foreheadarr.reshape((5,int(foreheadarr.shape[0]/5),foreheadarr.shape[1],foreheadarr.shape[2],foreheadarr.shape[3]))
    nosearr = np.array(nosearr)
    nosearr = nosearr.reshape((5, int(nosearr.shape[0] / 5), nosearr.shape[1], nosearr.shape[2], nosearr.shape[3]))
    timeseries=[]
    frequencyseries=[]
    for j in range(5):
        time3Ddimension,frequency3Ddimension=turntime_frequencysequence(foreheadarr[j],nosearr[j])
        timeseries.append(time3Ddimension)
        frequencyseries.append(frequency3Ddimension)
    timeseries=np.array(timeseries)
    frequencyseries=np.array(frequencyseries)

    print('OVER')

    return timeseries,frequencyseries


def load_all_data(ecgdatapath,imgsdatapath):
    labelarr= []
    ecgarr=[]
    timearr=[]
    frequencyarr=[]
    with open(ecgdatapath, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            eachrow = [float(i) for i in row]
            labelarr.append(eachrow[-8])
            ecgarr.append(eachrow[1:-8])
            timedata,frequencydata=load_imgs(imgsdatapath,str(int(eachrow[0])))
            timearr.append(timedata)
            frequencyarr.append(frequencydata)

    labelarr=np.array(labelarr)
    print("标签维度是",labelarr.shape)
    ecgarr=np.array(ecgarr)
    print("ecg信号维度是",ecgarr.shape)
    timearr=np.array(timearr)
    print("图像时序数据维度是", timearr.shape)
    frequencyarr=np.array(frequencyarr)
    print("图像频域数据维度是", frequencyarr.shape)

    return labelarr,ecgarr,timearr,frequencyarr

def label_preprocessing(labelarr):

    newlabel = np.zeros((labelarr.shape[0], 1))
    for i in range(labelarr.shape[0]):
        newlabel[i,:] = labelarr[i]
    print("新label维度", newlabel.shape)
    # newlabel = normalize(newlabel, axis=0, norm='max')

    return newlabel

def img_preprocessing(timearr,frequencyarr):
    timearr = timearr.astype('float32')
    timearr /= 255
    # mean= timearr.mean(axis=0)
    # timearr -= mean

    frequencyarr= frequencyarr.astype('float32')
    frequencyarr /= np.max(frequencyarr,axis=0)
    # mean_fr = frequencyarr.mean(axis=0)
    # frequencyarr-=mean_fr

    return timearr,frequencyarr

def models(labelarr,ecgarr,timearr,frequencyarr):

    ecg_train, ecg_test, y_train, y_test = train_test_split(ecgarr, labelarr, test_size=0.2, random_state=4)
    time_train, time_test, y_train, y_test = train_test_split(timearr, labelarr, test_size=0.2, random_state=4)
    frequency_train, frequency_test, y_train, y_test = train_test_split(frequencyarr, labelarr, test_size=0.2, random_state=4)

    ecg_val=ecg_train[:int(0.25*ecg_train.shape[0])]
    ecg_train = ecg_train[int(0.25 * ecg_train.shape[0]):]

    time_val=time_train[:int(0.25*time_train.shape[0])]
    time_train = time_train[int(0.25 * time_train.shape[0]):]

    frequency_val=frequency_train[:int(0.25*frequency_train.shape[0])]
    frequency_train = frequency_train[int(0.25 * frequency_train.shape[0]):]

    y_val=y_train[:int(0.25*y_train.shape[0])]
    y_train= y_train[int(0.25 * y_train.shape[0]):]
    print(ecg_val.shape)
    print(ecg_train.shape)

    print(time_val.shape)
    print(time_train.shape)

    print(frequency_val.shape)
    print(frequency_train.shape)

    print(y_val.shape)
    print(y_train.shape)

    # input
    timeinput = Input(shape=(timearr.shape[1], timearr.shape[2], timearr.shape[3],timearr.shape[4]))

    frequencyinput = Input(shape=(frequencyarr.shape[1], frequencyarr.shape[2], frequencyarr.shape[3],frequencyarr.shape[4]))

    ecginput=Input(shape=(ecgarr.shape[1],1))

    conv1 = Conv3D(filters=64, kernel_size=(1, 11,11), padding='same', activation='relu',strides=(1,4,4))(timeinput)
    bn1 = BatchNormalization()(conv1)
    pool1=MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn1)
    conv2 = Conv3D(filters=96, kernel_size=(1,5, 5), padding='same', activation='relu', strides=(1,1, 1))(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2=MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn2)
    conv3 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(pool2)
    bn3 = BatchNormalization()(conv3)
    conv4 = Conv3D(filters=96, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn3)
    bn4 = BatchNormalization()(conv4)
    conv5= Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn4)
    bn5 = BatchNormalization()(conv5)
    pool3 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn5)
    fla1=Flatten()(pool3)
    fc1=Dense(units=1024,activation='relu')(fla1)

    conv6 = Conv3D(filters=64, kernel_size=(1, 11,11), padding='same', activation='relu',strides=(1,4,4))(frequencyinput)
    bn6 = BatchNormalization()(conv6)
    pool4=MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn6)
    conv7 = Conv3D(filters=96, kernel_size=(1,5, 5), padding='same', activation='relu', strides=(1,1, 1))(pool4)
    bn7 = BatchNormalization()(conv7)
    pool5=MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn7)
    conv8 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(pool5)
    bn8 = BatchNormalization()(conv8)
    conv9 = Conv3D(filters=96, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn8)
    bn9 = BatchNormalization()(conv9)
    conv10= Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn9)
    bn10 = BatchNormalization()(conv10)
    pool6 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn10)
    fla2=Flatten()(pool6)
    fc2=Dense(units=1024,activation='relu')(fla2)

    conv11=Conv1D(filters=16,kernel_size=(16), strides=2, activation='relu')(ecginput)
    conv12=Conv1D(filters=16,kernel_size=(16), strides=2, activation='relu')(conv11)
    pool7=MaxPooling1D(pool_size=2)(conv12)
    conv13=Conv1D(filters=64,kernel_size=(8), strides=2, activation='relu')(pool7)
    conv14 = Conv1D(filters=64, kernel_size=(8), strides=2, activation='relu')(conv13)
    pool8= MaxPooling1D(pool_size=2)(conv14)
    conv15=Conv1D(filters=128,kernel_size=(4), strides=2, activation='relu')(pool8)
    conv16 = Conv1D(filters=128, kernel_size=(4), strides=2, activation='relu')(conv15)
    pool9 = MaxPooling1D(pool_size=2)(conv16)
    conv17=Conv1D(filters=256,kernel_size=(2), strides=1, activation='relu')(pool9)
    conv18 = Conv1D(filters=256, kernel_size=(2), strides=1, activation='relu')(conv17)
    pool10 = MaxPooling1D(pool_size=2)(conv18)
    fc3 = Dense(units=1024, activation='relu')(pool10)

    add1 = add([fc1, fc2,fc3])
    fla2 = Flatten()(add1)
    fc4=Dense(units=128, activation='relu')(fla2)
    fc5=Dense(units=labelarr.shape[1])(fc4)

    model = Model(input=[timeinput,frequencyinput,ecginput], output=fc5)

    model = multi_gpu_model(model, gpus=4)

    model.compile(optimizer=Adam(lr=1e-4),loss=losses.mean_squared_error, metrics=[metrics.MeanSquaredError(),
                                                                metrics.MeanAbsoluteError(),
                                                                metrics.MeanAbsolutePercentageError(),
                                                                metrics.RootMeanSquaredError(),pearson_r])

    model_checkpoint = ModelCheckpoint('/home/som/lab/seed-yzj/paper4/model/gitnet.hdf5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True)


    history = model.fit([time_train,frequency_train,ecg_train], y_train,batch_size=32,epochs=300,verbose=1,callbacks=[model_checkpoint],
                        validation_data=([time_val,frequency_val,ecg_val], y_val))

    return history


def ecg_preprocessing(ecgarr):
    newecg=np.zeros((ecgarr.shape[0],ecgarr.shape[1], 1))
    for i in range(ecgarr.shape[0]):
        for j in range(ecgarr.shape[1]):
            newecg[i,j,:]=ecgarr[i,j]
    print("新ecg维度",newecg.shape)
    return newecg


def drawlines(history):
    history_dict=history.history
    csv_file = open('/home/som/lab/seed-yzj/paper4/model/gitnet.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    for key in history_dict:
        writer.writerow([key, history_dict[key]])
    csv_file.close()

if __name__ == '__main__':
    ecg_label_path='/home/som/lab/seed-yzj/paper4/fusion.csv'
    imgsdatapath='/home/som/lab-data/seed-yzj/MAHNOB-HCI/images/'
    labelarr,ecgarr,timearr,frequencyarr=load_all_data(ecg_label_path,imgsdatapath)

    ecgarr=ecg_preprocessing(ecgarr)

    labelarr=label_preprocessing(labelarr)
    timearr,frequencyarr=img_preprocessing(timearr,frequencyarr)
    history=models(labelarr,ecgarr,timearr,frequencyarr)
    drawlines(history)