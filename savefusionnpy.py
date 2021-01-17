from cv2 import cv2
import os
import numpy as np
from scipy import signal
from keras.models import *
from keras.optimizers import *
from keras import backend as K
from keras import metrics, losses
from config import Fusionconfig

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

source_matrix = np.array([[1, 1, 1, 1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 0, 0, 0, 0],
                          [1, 1, 1, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 1, 1, 1, 1],
                          [0, 0, 0, 0, 1, 1, 1, 1]])

location_matrix = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1]])


def pearson_r(y_true, y_pred):
    """
        皮尔逊相关系数，模型编译用
        """
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

def img_preprocessing(timearr):

    timearr = timearr.astype('float32')
    timearr /= 255
    return timearr

class Fusion(Fusionconfig):
    def __init__(self):
        self.nosearr=np.load(self.nosearrpath)
        self.nosefrearr=np.load(self.nosefrearrpath)
        self.nosemodel=load_model(self.nosemodelpath,compile=False)
        self.nosemodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])
        # self.nosemodel.summary()

        self.foreheadarr=np.load(self.foreheadarrpath)
        self.foreheadfrearr=np.load(self.foreheadfrearrpath)
        self.foreheadmodel=load_model(self.foreheadmodelpath,compile=False)
        self.foreheadmodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])
        # self.foreheadmodel.summary()

        self.leftarr=np.load(self.leftarrpath)
        self.leftfrearr=np.load(self.leftfrearrpath)
        self.leftmodel=load_model(self.leftmodelpath,compile=False)
        self.leftmodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])

        self.rightarr=np.load(self.rightarrpath)
        self.rightfrearr=np.load(self.rightfrearrpath)
        self.rightmodel=load_model(self.rightmodelpath,compile=False)
        self.rightmodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])

        self.thermalnosearr=np.load(self.thermalnosearrpath)
        self.thermalnosefrearr=np.load(self.thermalnosefrearrpath)
        self.thermalnosemodel=load_model(self.thermalnosemodelpath,compile=False)
        self.thermalnosemodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])

        self.thermalforeheadarr=np.load(self.thermalforeheadarrpath)
        self.thermalforeheadfrearr=np.load(self.thermalforeheadfrearrpath)
        self.thermalforeheadmodel=load_model(self.thermalforeheadmodelpath,compile=False)
        self.thermalforeheadmodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])

        self.thermalleftarr=np.load(self.thermalleftarrpath)
        self.thermalleftfrearr=np.load(self.thermalleftfrearrpath)
        self.thermalleftmodel=load_model(self.thermalleftmodelpath,compile=False)
        self.thermalleftmodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])

        self.thermalrightarr=np.load(self.thermalrightarrpath)
        self.thermalrightfrearr=np.load(self.thermalrightfrearrpath)
        self.thermalrightmodel=load_model(self.thermalrightmodelpath,compile=False)
        self.thermalrightmodel.compile(optimizer=Adam(lr=1e-4), loss=losses.mean_squared_error, metrics=[metrics.MeanAbsoluteError(),
                                                                                  metrics.MeanAbsolutePercentageError(),
                                                                                  metrics.RootMeanSquaredError(),
                                                                                  pearson_r])
    def feature_extraction(self,model,arr):
        layer_model = Model(input=model.input,
                                    output=model.get_layer('add_42').output)
        arr=img_preprocessing(arr)
        output = layer_model.predict(arr)

        return output

    def cal_feature_sim(self,arr1,arr2):
        dis=np.sqrt(np.sum(np.power((arr1 - arr2), 2)))
        # print(dis.shape)
        # print(dis)
        return dis

    def pipeline(self):

        #空间特征计算
        self.nose_feature=self.feature_extraction(self.nosemodel,self.nosearr)
        self.forehead_feature=self.feature_extraction(self.foreheadmodel,self.foreheadarr)
        self.left_feature=self.feature_extraction(self.leftmodel,self.leftarr)
        self.right_feature=self.feature_extraction(self.rightmodel,self.rightarr)

        self.thermal_nose_feature=self.feature_extraction(self.thermalnosemodel,self.thermalnosearr)
        self.thermal_forehead_feature=self.feature_extraction(self.thermalforeheadmodel,self.thermalforeheadarr)
        self.thermal_left_feature=self.feature_extraction(self.thermalleftmodel,self.thermalleftarr)
        self.thermal_right_feature=self.feature_extraction(self.thermalrightmodel,self.thermalrightarr)

        #空间特征相似度
        integrate_feature_arr=np.zeros((self.nose_feature.shape[0],8,self.nose_feature.shape[1],
                                        self.nose_feature.shape[2],self.nose_feature.shape[3]))

        integrate_feature_arr[:,0,:,:,:]=self.nose_feature
        integrate_feature_arr[ :,1, :, :,:] = self.forehead_feature
        integrate_feature_arr[ :,2, :, :,:] = self.left_feature
        integrate_feature_arr[ :,3, :, :,:] = self.right_feature
        integrate_feature_arr[:,4,:,:,:]=self.thermal_nose_feature
        integrate_feature_arr[ :,5, :, :,:] = self.thermal_forehead_feature
        integrate_feature_arr[ :,6, :, :,:] = self.thermal_left_feature
        integrate_feature_arr[ :,7, :, :,:] = self.thermal_right_feature
        # print(integrate_feature_arr)
        feature_arr = np.zeros((self.nose_feature.shape[0], 8, 8))

        for i in range(integrate_feature_arr.shape[0]):
            for j in range(integrate_feature_arr.shape[1]):
                for m in range(j+1,integrate_feature_arr.shape[1]):
                    feature_arr[i,j,m]=feature_arr[i,m,j]=self.cal_feature_sim(integrate_feature_arr[i,j],
                                                                               integrate_feature_arr[i,m])
        #空间特征相似度归一化
        feature_arr=1-(feature_arr-309.7)*0.8/(645.27-309.7)
        for i in range(feature_arr.shape[0]):
            for j in range(feature_arr.shape[1]):
                feature_arr[i, j, j] = 1
        # print(feature_arr)

        #位置和源相似度
        source_arr=[]
        location_arr=[]
        for i in range(self.nose_feature.shape[0]):
            source_arr.append(source_matrix)
            location_arr.append(location_matrix)
        source_arr=np.array(source_arr)
        location_arr=np.array(location_arr)

        #频率相似度
        integrate_fre_arr = np.zeros((self.nosefrearr.shape[0], 8, self.nosefrearr.shape[1],
                                          self.nosefrearr.shape[2], self.nosefrearr.shape[3]))

        integrate_fre_arr[:, 0, :, :, :] = self.nosefrearr
        integrate_fre_arr[:, 1, :, :, :] = self.foreheadfrearr
        integrate_fre_arr[:, 2, :, :, :] = self.leftfrearr
        integrate_fre_arr[:, 3, :, :, :] = self.rightfrearr
        integrate_fre_arr[:, 4, :, :, :] = self.thermalnosefrearr
        integrate_fre_arr[:, 5, :, :, :] = self.thermalforeheadfrearr
        integrate_fre_arr[:, 6, :, :, :] = self.thermalleftfrearr
        integrate_fre_arr[:, 7, :, :, :] = self.thermalrightfrearr


        fre_feature_arr = np.zeros((self.nosefrearr.shape[0], 8, 8))

        for i in range(integrate_fre_arr.shape[0]):
            for j in range(integrate_fre_arr.shape[1]):
                for m in range(j + 1, integrate_fre_arr.shape[1]):
                    fre_feature_arr[i, j, m] = fre_feature_arr[i, m, j] = self.cal_feature_sim(integrate_fre_arr[i, j],
                                                                                       integrate_fre_arr[i, m])
        #频率相似度归一化
        fre_feature_arr=1-(fre_feature_arr-10.47)*0.8/(431.09-10.47)
        for i in range(fre_feature_arr.shape[0]):
            for j in range(fre_feature_arr.shape[1]):
                fre_feature_arr[i, j, j] = 1
        # print(fre_feature_arr)

        final_sim_arr=0.25*feature_arr+0.25*source_arr+0.25*location_arr+0.25*fre_feature_arr
        print(final_sim_arr)

        np.save('/home/som/lab/seed-yzj/newpaper4/laboratory/featurenpy/feature.npy', integrate_feature_arr)
        np.save('/home/som/lab/seed-yzj/newpaper4/laboratory/featurenpy/simlarity.npy', final_sim_arr)


if __name__ == '__main__':
    fusion_similarity = Fusion()
    fusion_similarity.pipeline()