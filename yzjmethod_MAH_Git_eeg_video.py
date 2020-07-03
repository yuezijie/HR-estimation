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

os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"


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
    print('正在读',num)
    foreheadarr=[]
    nosearr=[]
    imgsfolder=imgspath+'/'+num

    count=0
    while True:
        i=300
        foreheadimgpath = os.path.join(imgsfolder, 'forehead_{0}.jpg'.format(i))
        noseimgpath = os.path.join(imgsfolder, 'nose_{0}.jpg'.format(i))
        if count<1500:
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

    return timeseries,frequencyseries

def load_eeg(eegpath,num):
    eegarr=[]
    eegsfile = eegpath+ '/' + num+'/eeg5-25s.csv'
    with open(eegsfile, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            eachrow = [float(i) for i in row]
            eegarr.append(eachrow)
    eegarr=np.array(eegarr)
    return eegarr

def load_all_data(hrdatapath,img_folderpath,EEG_folderpath):
    labelarr= []
    timearr=[]
    frequencyarr = []
    eegarr = []
    count=0
    with open(hrdatapath, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # if count<30:
            labelarr.append(float(row[1]))
            sample_num=row[0]
            eegdata = load_eeg(EEG_folderpath, str(int(sample_num)))

            timedata,frequencydata= load_imgs(img_folderpath, str(int(sample_num)))
            timearr.append(timedata)
            frequencyarr.append(frequencydata)
            eegarr.append(eegdata)
                # count+=1

    labelarr=np.array(labelarr)
    print("标签维度是",labelarr.shape)
    timearr=np.array(timearr)
    print("图像时序数据维度是", timearr.shape)
    frequencyarr=np.array(frequencyarr)
    print("图像频域数据维度是", frequencyarr.shape)
    eegarr=np.array(eegarr)
    print("eeg维度是",eegarr.shape)

    return labelarr,timearr,frequencyarr,eegarr

def label_preprocessing(labelarr):

    newlabel = np.zeros((labelarr.shape[0], 1))
    for i in range(labelarr.shape[0]):
        newlabel[i,:] = labelarr[i]
    print("新label维度", newlabel.shape)
    # newlabel = normalize(newlabel, axis=0, norm='max')

    return newlabel


def models(labelarr,timearr,frequencyarr,eegarr):


    time_train, time_test, y_train, y_test = train_test_split(timearr, labelarr, test_size=0.3, random_state=3)
    eeg_train, eeg_test, y_train, y_test = train_test_split(eegarr, labelarr, test_size=0.3, random_state=3)
    frequency_train, frequency_test, y_train, y_test = train_test_split(frequencyarr, labelarr, test_size=0.3,
                                                                        random_state=3)
    eeginput = Input(shape=(eegarr.shape[1], eegarr.shape[2]))
    # input
    timeinput = Input(shape=(timearr.shape[1], timearr.shape[2], timearr.shape[3],timearr.shape[4]))
    frequencyinput = Input(
        shape=(frequencyarr.shape[1], frequencyarr.shape[2], frequencyarr.shape[3], frequencyarr.shape[4]))

    conv1 = Conv3D(filters=64, kernel_size=(1, 11, 11), padding='same', activation='relu', strides=(1, 4, 4))(timeinput)
    bn1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn1)
    conv2 = Conv3D(filters=96, kernel_size=(1, 5, 5), padding='same', activation='relu', strides=(1, 1, 1))(pool1)
    bn2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 4, 2))(bn2)
    conv3 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(pool2)
    bn3 = BatchNormalization()(conv3)
    conv4 = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn3)
    bn4 = BatchNormalization()(conv4)
    conv5 = Conv3D(filters=32, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn4)
    bn5 = BatchNormalization()(conv5)
    pool3 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 4, 2))(bn5)
    fla1 = Flatten()(pool3)
    fc1 = Dense(units=1024, activation='relu')(fla1)

    conv6 = Conv3D(filters=64, kernel_size=(1, 11, 11), padding='same', activation='relu', strides=(1, 4, 4))(
        frequencyinput)
    bn6 = BatchNormalization()(conv6)
    pool4 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 2, 2))(bn6)
    conv7 = Conv3D(filters=96, kernel_size=(1, 5, 5), padding='same', activation='relu', strides=(1, 1, 1))(pool4)
    bn7 = BatchNormalization()(conv7)
    pool5 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 4, 2))(bn7)
    conv8 = Conv3D(filters=128, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(pool5)
    bn8 = BatchNormalization()(conv8)
    conv9 = Conv3D(filters=64, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn8)
    bn9 = BatchNormalization()(conv9)
    conv10 = Conv3D(filters=32, kernel_size=(1, 3, 3), padding='same', activation='relu', strides=(1, 1, 1))(bn9)
    bn10 = BatchNormalization()(conv10)
    pool6 = MaxPooling3D(pool_size=(1, 3, 3), strides=(1, 4, 2))(bn10)
    fla2 = Flatten()(pool6)
    fc2 = Dense(units=1024, activation='relu')(fla2)

    conv11 = Conv1D(filters=16, kernel_size=(8), strides=1, activation='relu')(eeginput)
    conv12 = Conv1D(filters=64, kernel_size=(4), strides=1, activation='relu')(conv11)
    pool7 = MaxPooling1D(pool_size=2)(conv12)
    conv13 = Conv1D(filters=128, kernel_size=(4), strides=1, activation='relu')(pool7)
    conv14 = Conv1D(filters=64, kernel_size=(4), strides=1, activation='relu')(conv13)
    pool8 = MaxPooling1D(pool_size=2)(conv14)
    # conv15=Conv1D(filters=128,kernel_size=(4), strides=1, activation='relu')(pool8)
    # conv16 = Conv1D(filters=128, kernel_size=(4), strides=1, activation='relu')(conv15)
    # pool9 = MaxPooling1D(pool_size=2)(conv16)
    # conv17=Conv1D(filters=256,kernel_size=(2), strides=1, activation='relu')(pool9)
    # conv18 = Conv1D(filters=256, kernel_size=(2), strides=1, activation='relu')(conv17)
    # pool10 = MaxPooling1D(pool_size=2)(conv18)
    fla3 = Flatten()(pool8)
    fc3 = Dense(units=1024, activation='relu')(fla3)

    add1 = add([fc1, fc2, fc3])

    # fla4 = Flatten()(add1)
    fc4=Dense(units=1024, activation='relu')(add1)
    fc5 = Dense(units=512, activation='relu')(fc4)
    fc6 = Dense(units=256, activation='relu')(fc5)
    fc7 = Dense(units=128, activation='relu')(fc6)
    fc8 = Dense(units=64, activation='relu')(fc7)
    fc9=Dense(units=labelarr.shape[1])(fc8)

    model = Model(input=[timeinput,frequencyinput,eeginput], output=fc9)
    model.summary()

    model = multi_gpu_model(model, gpus=3)

    model.compile(optimizer=Adam(lr=1e-4),loss=losses.mean_squared_error, metrics=[metrics.MeanSquaredError(),
                                                                metrics.MeanAbsoluteError(),
                                                                metrics.MeanAbsolutePercentageError(),
                                                                metrics.RootMeanSquaredError(),pearson_r])

    # model_checkpoint = ModelCheckpoint('/home/som/lab/seed-yzj/paper4/model/eeg+videonet.hdf5', monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=True)

    history = model.fit([time_train, frequency_train, eeg_train], y_train, batch_size=32, epochs=300, verbose=1,
                        validation_data=([time_test, frequency_test, eeg_test], y_test))

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
    csv_file = open('/home/som/lab/seed-yzj/paper4/model/eeg+videonet.csv', 'w', newline='')
    writer = csv.writer(csv_file)
    for key in history_dict:
        writer.writerow([key, history_dict[key]])
    csv_file.close()

def img_preprocessing(timearr,frequencyarr):
    timearr = timearr.astype('float32')
    timearr /= 255

    frequencyarr= frequencyarr.astype('float32')
    frequencyarr /= np.max(frequencyarr,axis=0)

    return timearr,frequencyarr

if __name__ == '__main__':
    hr_label_path='/home/som/lab/seed-yzj/paper5/data/MAH_sr_exp/heartrate_gt.csv'

    EEG_folderpath='/home/som/lab-data/seed-yzj/MAHNOB-HCI/Sessions/'
    imgsdatapath = '/home/som/lab-data/seed-yzj/MAHNOB-HCI/roi_images/'

    labelarr,timearr, frequencyarr,eegarr=load_all_data(hr_label_path,imgsdatapath,EEG_folderpath)

    labelarr=label_preprocessing(labelarr)
    print(labelarr)
    timearr, frequencyarr = img_preprocessing(timearr, frequencyarr)
    history=models(labelarr,timearr, frequencyarr,eegarr)
    # drawlines(history)